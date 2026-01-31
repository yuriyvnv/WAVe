#!/usr/bin/env python3
"""
Trace forward pass step-by-step to find where models diverge.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoConfig, AutoModel
from configuration_wave import WAVeConfig
from modeling_wave import WAVeForQualityVerification
from original_model_standalone import EnhancedAudioTextModel

# Register
AutoConfig.register('wave', WAVeConfig)
AutoModel.register(WAVeConfig, WAVeForQualityVerification)

# Static test data
np.random.seed(42)
audio_array = np.random.randn(16000).astype(np.float32) * 0.1
test_text = "Olá, como você está?"

print("=" * 80)
print("Forward Pass Trace")
print("=" * 80)

# Prepare inputs
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

text_inputs = tokenizer(test_text, padding=True, truncation=True, return_tensors="pt")
audio_features = feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")
audio_input = audio_features["input_features"]
audio_mask = audio_features.get("attention_mask", None)

# Load checkpoint
checkpoint = torch.load('../training_multimodal/3_alignment_MHGLU_Portuguese/best_model_gap.pt', map_location='cpu', weights_only=False)

# ===== ORIGINAL MODEL =====
print("\n1. ORIGINAL MODEL")
orig_model = EnhancedAudioTextModel(
    text_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    audio_model_name="facebook/w2v-bert-2.0",
    projection_dim=checkpoint.get('projection_dim', 768),
    use_cross_modal=False,
    use_attentive_pooling=False,
    use_word_alignment=True,
    freeze_encoders="none"
)
orig_model.load_state_dict(checkpoint['model_state_dict'])
orig_model.eval()

with torch.no_grad():
    # Text encoding
    text_outputs = orig_model.text_encoder(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"])
    text_hidden = text_outputs.last_hidden_state
    text_cls = text_hidden[:, 0, :]
    print(f"  Text CLS: shape={text_cls.shape}, mean={text_cls.mean():.6f}, std={text_cls.std():.6f}")

    # Audio encoding
    audio_outputs = orig_model.audio_encoder(input_features=audio_input, attention_mask=audio_mask)
    audio_hidden = audio_outputs.last_hidden_state
    audio_pooled = audio_hidden.mean(dim=1)
    print(f"  Audio pooled: shape={audio_pooled.shape}, mean={audio_pooled.mean():.6f}, std={audio_pooled.std():.6f}")

    # Text projection
    text_projected = orig_model.text_projection(text_cls)
    print(f"  Text projected: mean={text_projected.mean():.6f}, std={text_projected.std():.6f}")

    # Audio projection
    audio_projected = orig_model.audio_projection(audio_pooled)
    print(f"  Audio projected: mean={audio_projected.mean():.6f}, std={audio_projected.std():.6f}")

    # Normalize
    text_normalized = torch.nn.functional.normalize(text_projected, p=2, dim=-1)
    audio_normalized = torch.nn.functional.normalize(audio_projected, p=2, dim=-1)
    print(f"  Text normalized: mean={text_normalized.mean():.6f}, std={text_normalized.std():.6f}")
    print(f"  Audio normalized: mean={audio_normalized.mean():.6f}, std={audio_normalized.std():.6f}")

    # Cosine sim
    cosine_sim = torch.nn.functional.cosine_similarity(text_normalized, audio_normalized)
    print(f"  Cosine similarity: {cosine_sim.item():.6f}")

# ===== HF MODEL =====
print("\n2. HF MODEL")
hf_model = AutoModel.from_pretrained('./wave-portuguese')
hf_model.eval()

with torch.no_grad():
    # Text encoding
    text_outputs = hf_model.wave.text_encoder(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"])
    text_hidden = text_outputs.last_hidden_state
    text_cls = text_hidden[:, 0, :]
    print(f"  Text CLS: shape={text_cls.shape}, mean={text_cls.mean():.6f}, std={text_cls.std():.6f}")

    # Audio encoding
    audio_outputs = hf_model.wave.audio_encoder(input_features=audio_input, attention_mask=audio_mask)
    audio_hidden = audio_outputs.last_hidden_state
    audio_pooled = audio_hidden.mean(dim=1)
    print(f"  Audio pooled: shape={audio_pooled.shape}, mean={audio_pooled.mean():.6f}, std={audio_pooled.std():.6f}")

    # Text projection
    text_projected = hf_model.wave.text_projection(text_cls)
    print(f"  Text projected: mean={text_projected.mean():.6f}, std={text_projected.std():.6f}")

    # Audio projection
    audio_projected = hf_model.wave.audio_projection(audio_pooled)
    print(f"  Audio projected: mean={audio_projected.mean():.6f}, std={audio_projected.std():.6f}")

    # Normalize
    text_normalized = torch.nn.functional.normalize(text_projected, p=2, dim=-1)
    audio_normalized = torch.nn.functional.normalize(audio_projected, p=2, dim=-1)
    print(f"  Text normalized: mean={text_normalized.mean():.6f}, std={text_normalized.std():.6f}")
    print(f"  Audio normalized: mean={audio_normalized.mean():.6f}, std={audio_normalized.std():.6f}")

    # Cosine sim
    cosine_sim = torch.nn.functional.cosine_similarity(text_normalized, audio_normalized)
    print(f"  Cosine similarity: {cosine_sim.item():.6f}")

print("\n" + "=" * 80)
