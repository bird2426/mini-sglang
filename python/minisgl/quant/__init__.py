"""Quantization module for mini-sglang.

This module provides a pluggable quantization framework supporting multiple
quantization schemes including AWQ, GPTQ, and bitsandbytes.
"""

from __future__ import annotations

from .awq import AWQConfig
from .base import (
    BaseQuantConfig,
    QuantConfig,
    create_quant_config,
    get_quant_config_class,
    list_quant_types,
    register_quant_config,
)
from .loader import detect_quant_config, load_quantized_weight, shard_quantized_state_dict

__all__ = [
    # Base classes and registry
    "BaseQuantConfig",
    "QuantConfig",
    "register_quant_config",
    "get_quant_config_class",
    "create_quant_config",
    "list_quant_types",
    # Quantization implementations
    "AWQConfig",
    # Weight loading utilities
    "load_quantized_weight",
    "detect_quant_config",
    "shard_quantized_state_dict",
]
