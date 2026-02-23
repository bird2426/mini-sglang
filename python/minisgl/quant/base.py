"""Quantization support for mini-sglang.

This module provides a pluggable quantization framework supporting multiple
quantization schemes including AWQ, GPTQ, and bitsandbytes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, Type, runtime_checkable

import torch

if TYPE_CHECKING:
    from minisgl.engine.config import EngineConfig


@runtime_checkable
class QuantConfig(Protocol):
    """Protocol defining the interface for quantization configurations.
    
    All quantization schemes must implement this protocol to be compatible
    with the quantization framework.
    """
    
    @property
    def quant_type(self) -> str:
        """Return the type of quantization (e.g., 'awq', 'gptq', 'bnb')."""
        ...
    
    @property
    def bits(self) -> int:
        """Return the number of bits for quantization."""
        ...
    
    def get_layer_config(self, layer_name: str) -> Optional[Dict[str, Any]]:
        """Get quantization configuration for a specific layer.
        
        Args:
            layer_name: The name of the layer (e.g., 'model.layers.0.self_attn.q_proj')
            
        Returns:
            Dictionary with layer-specific quantization config, or None if not quantized.
        """
        ...
    
    def is_layer_quantized(self, layer_name: str) -> bool:
        """Check if a specific layer should be quantized.
        
        Args:
            layer_name: The name of the layer.
            
        Returns:
            True if the layer should be quantized.
        """
        ...


class BaseQuantConfig:
    """Base class for quantization configurations.
    
    Provides common functionality for all quantization schemes.
    Subclasses should implement scheme-specific logic.
    """
    
    def __init__(
        self,
        quant_type: str,
        bits: int,
        group_size: int = 128,
        desc_act: bool = False,
        static_groups: bool = False,
        sym: bool = True,
        true_sequential: bool = True,
        lm_head: bool = False,
    ):
        """Initialize base quantization configuration.
        
        Args:
            quant_type: Type of quantization (e.g., 'awq', 'gptq')
            bits: Number of bits for quantization (4, 8)
            group_size: Group size for quantization (default: 128)
            desc_act: Whether to use descending activation order (GPTQ-specific)
            static_groups: Whether to use static groups (GPTQ-specific)
            sym: Whether to use symmetric quantization
            true_sequential: Whether to use true sequential quantization
            lm_head: Whether to quantize the LM head
        """
        self._quant_type = quant_type
        self._bits = bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.static_groups = static_groups
        self.sym = sym
        self.true_sequential = true_sequential
        self.lm_head = lm_head
        
        # Set of layer names that should NOT be quantized
        self._excluded_layers: set[str] = set()
    
    @property
    def quant_type(self) -> str:
        """Return the quantization type."""
        return self._quant_type
    
    @property
    def bits(self) -> int:
        """Return the number of bits."""
        return self._bits
    
    def exclude_layer(self, layer_name: str) -> None:
        """Add a layer to the exclusion list.
        
        Args:
            layer_name: Name of the layer to exclude from quantization.
        """
        self._excluded_layers.add(layer_name)
    
    def include_layer(self, layer_name: str) -> None:
        """Remove a layer from the exclusion list.
        
        Args:
            layer_name: Name of the layer to include in quantization.
        """
        self._excluded_layers.discard(layer_name)
    
    def is_layer_quantized(self, layer_name: str) -> bool:
        """Check if a layer should be quantized.
        
        By default, all linear projection layers are quantized except:
        - LM head (unless lm_head=True)
        - Explicitly excluded layers
        
        Args:
            layer_name: Name of the layer to check.
            
        Returns:
            True if the layer should be quantized.
        """
        if layer_name in self._excluded_layers:
            return False
        
        # Check if it's a quantizable linear layer
        quantizable_patterns = (
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "qkv_proj", "gate_up_proj",
        )
        
        is_linear = any(pattern in layer_name for pattern in quantizable_patterns)
        
        if not is_linear:
            return False
        
        return True
    
    def get_layer_config(self, layer_name: str) -> Optional[Dict[str, Any]]:
        """Get quantization configuration for a layer.
        
        Args:
            layer_name: Name of the layer.
            
        Returns:
            Dictionary with quantization config if quantized, None otherwise.
        """
        if not self.is_layer_quantized(layer_name):
            return None
        
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "desc_act": self.desc_act,
            "sym": self.sym,
        }
    
    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"quant_type='{self.quant_type}', "
            f"bits={self.bits}, "
            f"group_size={self.group_size}, "
            f"sym={self.sym}"
            f")"
        )


# Registry for quantization configurations
_QUANT_CONFIG_REGISTRY: Dict[str, Type[BaseQuantConfig]] = {}


def register_quant_config(quant_type: str, config_class: Type[BaseQuantConfig]) -> None:
    """Register a quantization configuration class.
    
    Args:
        quant_type: The quantization type identifier (e.g., 'awq', 'gptq')
        config_class: The configuration class to register
        
    Raises:
        ValueError: If quant_type is already registered.
    """
    if quant_type in _QUANT_CONFIG_REGISTRY:
        raise ValueError(f"Quantization type '{quant_type}' is already registered")
    _QUANT_CONFIG_REGISTRY[quant_type] = config_class


def get_quant_config_class(quant_type: str) -> Optional[Type[BaseQuantConfig]]:
    """Get the configuration class for a quantization type.
    
    Args:
        quant_type: The quantization type identifier.
        
    Returns:
        The configuration class, or None if not registered.
    """
    return _QUANT_CONFIG_REGISTRY.get(quant_type)


def create_quant_config(quant_type: str, **kwargs: Any) -> BaseQuantConfig:
    """Create a quantization configuration instance.
    
    Args:
        quant_type: The quantization type identifier.
        **kwargs: Additional arguments passed to the config constructor.
        
    Returns:
        A quantization configuration instance.
        
    Raises:
        ValueError: If quant_type is not registered.
    """
    config_class = get_quant_config_class(quant_type)
    if config_class is None:
        available = list(_QUANT_CONFIG_REGISTRY.keys())
        raise ValueError(
            f"Unknown quantization type '{quant_type}'. "
            f"Available types: {available}"
        )
    return config_class(quant_type=quant_type, **kwargs)


def list_quant_types() -> list[str]:
    """List all registered quantization types.
    
    Returns:
        List of registered quantization type identifiers.
    """
    return list(_QUANT_CONFIG_REGISTRY.keys())
