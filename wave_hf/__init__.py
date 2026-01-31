"""
WAVe: Word-Aligned Verification of Synthetic Speech for ASR

This package provides HuggingFace Transformers integration for WAVe models.
"""

from .configuration_wave import WAVeConfig
from .modeling_wave import (
    WAVeModel,
    WAVeForQualityVerification,
    WAVeModelOutput,
    WAVeForQualityOutput,
    WAVePreTrainedModel,
)
from .processing_wave import WAVeProcessor

# Auto-registration with transformers
try:
    from transformers import AutoConfig, AutoModel, AutoProcessor

    AutoConfig.register("wave", WAVeConfig)
    AutoModel.register(WAVeConfig, WAVeModel)
    AutoModel.register(WAVeConfig, WAVeForQualityVerification)
    # Note: AutoProcessor registration happens via processing_wave.py

except ImportError:
    # Transformers not installed, skip auto-registration
    pass

__version__ = "1.0.0"

__all__ = [
    "WAVeConfig",
    "WAVeModel",
    "WAVeForQualityVerification",
    "WAVeModelOutput",
    "WAVeForQualityOutput",
    "WAVePreTrainedModel",
    "WAVeProcessor",
]
