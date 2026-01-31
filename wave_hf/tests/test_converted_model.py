#!/usr/bin/env python3
"""Test script for converted WAVe model."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoConfig, AutoModel, AutoProcessor
from configuration_wave import WAVeConfig
from modeling_wave import WAVeForQualityVerification
import torch
import numpy as np

# Register the model
AutoConfig.register("wave", WAVeConfig)
AutoModel.register(WAVeConfig, WAVeForQualityVerification)

print("=" * 80)
print("Testing Converted WAVe Model")
print("=" * 80)

# Load processor
print("\n1. Loading processor...")
from processing_wave import WAVeProcessor
from transformers import AutoTokenizer, AutoFeatureExtractor

tokenizer = AutoTokenizer.from_pretrained('./wave-portuguese')
feature_extractor = AutoFeatureExtractor.from_pretrained('./wave-portuguese')
processor = WAVeProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
print("   ✓ Processor loaded")

# Load model
print("\n2. Loading model...")
model = AutoModel.from_pretrained('./wave-portuguese')
model.eval()
print("   ✓ Model loaded")

# Print config
print("\n3. Model configuration:")
print(f"   - Projection dim: {model.config.projection_dim}")
print(f"   - Use word alignment: {model.config.use_word_alignment}")
print(f"   - Use cross modal: {model.config.use_cross_modal}")
print(f"   - Use attentive pooling: {model.config.use_attentive_pooling}")

# Print architecture
print("\n4. Model architecture:")
print(f"   - Type: {type(model).__name__}")
print(f"   - Has wave attribute: {hasattr(model, 'wave')}")
if hasattr(model, 'wave'):
    print(f"   - Has text_encoder: {hasattr(model.wave, 'text_encoder')}")
    print(f"   - Has audio_encoder: {hasattr(model.wave, 'audio_encoder')}")
    print(f"   - Has word_level_alignment: {hasattr(model.wave, 'word_level_alignment')}")

# Test inference
print("\n5. Testing inference...")
text = "Olá, como você está?"
audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz

print(f"   - Text: '{text}'")
print(f"   - Audio shape: {audio.shape}")

inputs = processor(
    text=text,
    audio=audio,
    sampling_rate=16000,
    return_tensors="pt"
)

print(f"   - Input keys: {list(inputs.keys())}")

with torch.no_grad():
    outputs = model(**inputs)

print("\n6. Model outputs:")
print(f"   - Quality score: {outputs.quality_score.item():.4f}")
print(f"   - Cosine similarity: {outputs.cosine_similarity.item():.4f}")
if outputs.mean_alignment_score is not None:
    print(f"   - Mean alignment score: {outputs.mean_alignment_score.item():.4f}")
print(f"   - Text embeds shape: {outputs.text_embeds.shape}")
print(f"   - Audio embeds shape: {outputs.audio_embeds.shape}")
if outputs.alignment_scores is not None:
    print(f"   - Alignment scores shape: {outputs.alignment_scores.shape}")

print("\n" + "=" * 80)
print("✓ All tests passed!")
print("=" * 80)
