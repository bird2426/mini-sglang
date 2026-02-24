from .base import BaseLLMModel
from .config import ModelConfig, RotaryConfig
from .register import get_model_class
from .weight import load_weight
from minisgl.quant import detect_quant_config


def create_model(model_config: ModelConfig, quant_config=None) -> BaseLLMModel:
    model = get_model_class(model_config.architectures[0], model_config)
    
    # If quant_config is provided, set it on the model for later use
    if quant_config is not None:
        model.quant_config = quant_config
    
    return model
from .config import ModelConfig, RotaryConfig
from .register import get_model_class
from .weight import load_weight


def create_model(model_config: ModelConfig) -> BaseLLMModel:
    return get_model_class(model_config.architectures[0], model_config)


__all__ = ["create_model", "load_weight", "RotaryConfig"]
