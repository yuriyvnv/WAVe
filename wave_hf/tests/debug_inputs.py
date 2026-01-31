#!/usr/bin/env python3
"""
Debug script to see exactly what inputs each model receives.
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Static test data
np.random.seed(42)
audio_array = np.random.randn(16000).astype(np.float32)
audio_array = audio_array * 0.1
test_text = "Olá, como você está?"

print("=" * 80)
print("Input Comparison")
print("=" * 80)

# ===== ORIGINAL PROCESSING =====
print("\n1. ORIGINAL Processing:")

from transformers import AutoTokenizer, AutoFeatureExtractor

tokenizer_orig = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
feature_extractor_orig = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

# Text
text_inputs = tokenizer_orig(test_text, padding=True, truncation=True, return_tensors="pt")
print(f"\nText inputs:")
print(f"  input_ids shape: {text_inputs['input_ids'].shape}")
print(f"  input_ids: {text_inputs['input_ids']}")
print(f"  attention_mask: {text_inputs['attention_mask']}")

# Audio
audio_features = feature_extractor_orig(audio_array, sampling_rate=16000, return_tensors="pt")
print(f"\nAudio features keys: {list(audio_features.keys())}")

if "input_features" in audio_features:
    audio_input = audio_features["input_features"]
    print(f"  input_features shape: {audio_input.shape}")
    print(f"  input_features mean: {audio_input.mean():.6f}, std: {audio_input.std():.6f}")
    print(f"  input_features first 5 values: {audio_input.flatten()[:5]}")
elif "input_values" in audio_features:
    audio_input = audio_features["input_values"]
    print(f"  input_values shape: {audio_input.shape}")
    print(f"  input_values mean: {audio_input.mean():.6f}, std: {audio_input.std():.6f}")
    print(f"  input_values first 5 values: {audio_input.flatten()[:5]}")

if "attention_mask" in audio_features:
    print(f"  audio attention_mask shape: {audio_features['attention_mask'].shape}")
    print(f"  audio attention_mask sum: {audio_features['attention_mask'].sum()}")

# ===== HF PROCESSING =====
print("\n2. HF Processing:")

from processing_wave import WAVeProcessor

tokenizer_hf = AutoTokenizer.from_pretrained('./wave-portuguese')
feature_extractor_hf = AutoFeatureExtractor.from_pretrained('./wave-portuguese')
processor_hf = WAVeProcessor(tokenizer=tokenizer_hf, feature_extractor=feature_extractor_hf)

inputs_hf = processor_hf(text=test_text, audio=audio_array, sampling_rate=16000, return_tensors="pt")

print(f"\nHF processor output keys: {list(inputs_hf.keys())}")
print(f"\nText inputs:")
print(f"  input_ids shape: {inputs_hf['input_ids'].shape}")
print(f"  input_ids: {inputs_hf['input_ids']}")
print(f"  attention_mask: {inputs_hf['attention_mask']}")

print(f"\nAudio inputs:")
if "input_values" in inputs_hf:
    print(f"  input_values shape: {inputs_hf['input_values'].shape}")
    print(f"  input_values mean: {inputs_hf['input_values'].mean():.6f}, std: {inputs_hf['input_values'].std():.6f}")
    print(f"  input_values first 5 values: {inputs_hf['input_values'].flatten()[:5]}")

if "audio_attention_mask" in inputs_hf:
    print(f"  audio_attention_mask shape: {inputs_hf['audio_attention_mask'].shape}")
    print(f"  audio_attention_mask sum: {inputs_hf['audio_attention_mask'].sum()}")

# ===== COMPARISON =====
print("\n" + "=" * 80)
print("Comparison:")
print("=" * 80)

print(f"\nText input_ids match: {torch.equal(text_inputs['input_ids'], inputs_hf['input_ids'])}")
print(f"Text attention_mask match: {torch.equal(text_inputs['attention_mask'], inputs_hf['attention_mask'])}")

if "input_features" in audio_features:
    if "input_values" in inputs_hf:
        print(f"\\nAudio inputs:")
        print(f"  Original uses: input_features")
        print(f"  HF uses: input_values")
        print(f"  Shapes match: {audio_features['input_features'].shape == inputs_hf['input_values'].shape}")
        print(f"  Values match: {torch.allclose(audio_features['input_features'], inputs_hf['input_values'], atol=1e-5)}")
        if not torch.allclose(audio_features['input_features'], inputs_hf['input_values'], atol=1e-5):
            diff = (audio_features['input_features'] - inputs_hf['input_values']).abs()
            print(f"  Max difference: {diff.max():.6f}")
            print(f"  Mean difference: {diff.mean():.6f}")

print()
