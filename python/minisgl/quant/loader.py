"""Weight loading utilities for quantized models."""

from __future__ import annotations

import glob
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import safetensors
import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_ceil, download_hf_weight

if TYPE_CHECKING:
    from .base import BaseQuantConfig


QUANTIZED_SUFFIXES = ("qweight", "qzeros", "scales", "g_idx")


def _is_quantized_weight(key: str) -> bool:
    return any(key.endswith(suffix) for suffix in QUANTIZED_SUFFIXES)


def _parse_awq_config(model_path: str, state_dict: Dict[str, torch.Tensor]) -> Optional[Dict[str, Any]]:
    """Parse AWQ configuration from model state dict.
    
    AWQ models typically have keys like:
    - model.layers.0.self_attn.q_proj.qweight
    - model.layers.0.self_attn.q_proj.scales
    - model.layers.0.self_attn.q_proj.qzeros
    
    Args:
        model_path: Path to the model.
        state_dict: Model state dictionary.
        
    Returns:
        AWQ configuration dict if detected, None otherwise.
    """
    has_qweight = any(k.endswith(".qweight") for k in state_dict.keys())
    has_scales = any(k.endswith(".scales") for k in state_dict.keys())
    
    if not (has_qweight and has_scales):
        return None
    
    config: Dict[str, Any] = {"quant_type": "awq", "bits": 4, "group_size": 128}
    
    for key, tensor in state_dict.items():
        if key.endswith(".qweight"):
            weight_key = key.replace(".qweight", ".weight")
            if weight_key in state_dict:
                original_shape = state_dict[weight_key].shape
                quantized_shape = tensor.shape
                if len(original_shape) >= 2 and len(quantized_shape) >= 2:
                    ratio = original_shape[1] / quantized_shape[1]
                    if ratio >= 6:
                        config["bits"] = 4
                    elif ratio >= 3:
                        config["bits"] = 8
            break
    
    return config


def detect_quant_config(model_path: str) -> Optional[Dict[str, Any]]:
    """Detect quantization configuration from a model path.
    
    This function examines the model weights to determine if they are
    quantized and what quantization scheme is used.
    
    Args:
        model_path: Path to the model (HuggingFace model ID or local path).
        
    Returns:
        Quantization configuration dictionary if quantized model detected,
        None for unquantized models.
        
    Raises:
        ValueError: If multiple conflicting quantization schemes are detected.
    """
    model_folder = download_hf_weight(model_path)
    
    files = glob.glob(f"{model_folder}/*.safetensors")
    if not files:
        return None
    
    sample_state_dict: Dict[str, torch.Tensor] = {}
    for file in sorted(files)[:1]:
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in list(f.keys())[:100]:
                sample_state_dict[name] = f.get_tensor(name)
    
    awq_config = _parse_awq_config(model_path, sample_state_dict)
    if awq_config:
        return awq_config
    
    return None


def load_quantized_weight(
    model_path: str,
    device: torch.device,
    quant_config: Optional[BaseQuantConfig] = None,
) -> Tuple[Dict[str, torch.Tensor], Optional[BaseQuantConfig]]:
    """Load quantized weights from a model path.
    
    This function handles loading of quantized model weights, including:
    - AWQ quantized weights (qweight, scales, qzeros)
    - Future: GPTQ, bitsandbytes
    
    Args:
        model_path: Path to the model.
        device: Target device for the weights.
        quant_config: Optional pre-configured quantization config.
            If None, will attempt to auto-detect from model.
            
    Returns:
        Tuple of (state_dict, quant_config):
        - state_dict: Dictionary mapping weight names to tensors
        - quant_config: Quantization configuration (None if not quantized)
        
    Raises:
        ValueError: If quantized format is not supported.
        RuntimeError: If weight loading fails.
    """
    model_folder = download_hf_weight(model_path)
    files = glob.glob(f"{model_folder}/*.safetensors")
    
    if not files:
        raise RuntimeError(f"No safetensors files found in {model_folder}")
    
    state_dict: Dict[str, torch.Tensor] = {}
    
    for file in sorted(files):
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                state_dict[name] = f.get_tensor(name)
    
    if quant_config is None:
        detected = _parse_awq_config(model_path, state_dict)
        if detected:
            from .awq import AWQConfig
            quant_config = AWQConfig(
                bits=detected.get("bits", 4),
                group_size=detected.get("group_size", 128),
            )
    
    return state_dict, quant_config


def shard_quantized_state_dict(
    state_dict: Dict[str, torch.Tensor],
    quant_config: BaseQuantConfig,
) -> Dict[str, torch.Tensor]:
    """Shard quantized state dict for tensor parallelism.
    
    Args:
        state_dict: Full state dictionary.
        quant_config: Quantization configuration.
        
    Returns:
        Sharded state dictionary for current TP rank.
    """
    tp_info = get_tp_info()
    if tp_info.size == 1:
        return state_dict
    
    r = tp_info.rank
    n = tp_info.size
    
    sharded_dict: Dict[str, torch.Tensor] = {}
    
    for key, value in state_dict.items():
        if not any(key.endswith(suffix) for suffix in QUANTIZED_SUFFIXES):
            sharded_dict[key] = value
            continue
        
        base_key = key
        for suffix in (".qweight", ".scales", ".qzeros", ".g_idx"):
            if key.endswith(suffix):
                base_key = key[: -len(suffix)]
                break
        
        if any(pattern in base_key for pattern in (".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj")):
            sharded_dict[key] = value.chunk(n, dim=0)[r]
        elif any(pattern in base_key for pattern in (".o_proj", ".down_proj")):
            sharded_dict[key] = value.chunk(n, dim=1)[r] if value.dim() > 1 else value
        elif "lm_head" in key or "embed_tokens" in key:
            if value.dim() >= 1:
                num_embeddings = value.shape[0]
                num_embeddings_per_partition = div_ceil(num_embeddings, n)
                vocab_start_idx = r * num_embeddings_per_partition
                vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
                sharded_dict[key] = value[vocab_start_idx:vocab_end_idx]
            else:
                sharded_dict[key] = value
        else:
            sharded_dict[key] = value
    
    return sharded_dict
