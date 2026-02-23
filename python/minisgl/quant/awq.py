"""AWQ (Activation-aware Weight Quantization) configuration and support."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BaseQuantConfig, register_quant_config


class AWQConfig(BaseQuantConfig):
    """Configuration for AWQ (Activation-aware Weight Quantization).
    
    AWQ identifies 1% of salient weights in each layer and scales them
    before quantization to reduce quantization error. This results in
    better accuracy compared to standard quantization methods.
    
    Reference: https://arxiv.org/abs/2306.00978
    """
    
    def __init__(
        self,
        quant_type: str = "awq",
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        version: str = "gemm",
        **kwargs: Any,
    ):
        """Initialize AWQ configuration.
        
        Args:
            quant_type: Must be 'awq' for this config.
            bits: Number of bits for quantization (4 or 8). Default: 4
            group_size: Group size for quantization. Default: 128
            zero_point: Whether to use zero-point quantization. Default: True
            version: AWQ kernel version ('gemm' or 'gemv'). Default: 'gemm'
            **kwargs: Additional arguments passed to BaseQuantConfig.
        """
        if bits not in (4, 8):
            raise ValueError(f"AWQ only supports 4 or 8 bits, got {bits}")
        
        if version not in ("gemm", "gemv"):
            raise ValueError(f"AWQ version must be 'gemm' or 'gemv', got {version}")
        
        super().__init__(
            quant_type=quant_type,
            bits=bits,
            group_size=group_size,
            sym=not zero_point,
            **kwargs,
        )
        
        self.zero_point = zero_point
        self.version = version
    
    def get_layer_config(self, layer_name: str) -> Optional[Dict[str, Any]]:
        """Get AWQ-specific configuration for a layer.
        
        Args:
            layer_name: Name of the layer.
            
        Returns:
            Dictionary with AWQ config if quantized, None otherwise.
        """
        base_config = super().get_layer_config(layer_name)
        if base_config is None:
            return None
        
        base_config.update({
            "zero_point": self.zero_point,
            "version": self.version,
        })
        return base_config
    
    def __repr__(self) -> str:
        return (
            f"AWQConfig("
            f"bits={self.bits}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point}, "
            f"version='{self.version}'"
            f")"
        )


# Register AWQ configuration
register_quant_config("awq", AWQConfig)
