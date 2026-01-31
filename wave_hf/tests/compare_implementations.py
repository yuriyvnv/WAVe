#!/usr/bin/env python3
"""
Compare original implementation vs HuggingFace implementation.

This script runs inference on the same audio and text using both:
1. Original PyTorch implementation (from trainer_unfreeze.py)
2. HuggingFace converted implementation

We compare:
- Cosine similarity scores
- Text embeddings
- Audio embeddings
- Quality scores (if available)
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("WAVe Implementation Comparison")
print("=" * 80)

# ===== SETUP STATIC TEST DATA =====
print("\n1. Setting up test data...")

# Static audio array (1 second at 16kHz, Portuguese-like formants)
# This is a synthetic audio signal that mimics speech characteristics
np.random.seed(42)  # For reproducibility
audio_array = np.random.randn(16000).astype(np.float32)
audio_array = audio_array * 0.1  # Normalize amplitude

# Portuguese test text
test_text = "Olá, como você está?"

print(f"   Text: '{test_text}'")
print(f"   Audio: {audio_array.shape}, mean={audio_array.mean():.4f}, std={audio_array.std():.4f}")

# ===== ORIGINAL IMPLEMENTATION =====
print("\n2. Running ORIGINAL implementation...")

try:
    from original_model_standalone import EnhancedAudioTextModel
    from transformers import AutoTokenizer, AutoFeatureExtractor

    # Load checkpoint
    checkpoint_path = "../training_multimodal/3_alignment_MHGLU_Portuguese/best_model_gap.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Create model
    original_model = EnhancedAudioTextModel(
        text_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        audio_model_name="facebook/w2v-bert-2.0",
        projection_dim=checkpoint.get('projection_dim', 768),
        use_cross_modal=checkpoint.get('use_cross_modal', False),
        use_attentive_pooling=checkpoint.get('use_attentive_pooling', False),
        use_word_alignment=checkpoint.get('use_word_alignment', True),
        freeze_encoders="none"
    )

    # Load weights
    original_model.load_state_dict(checkpoint['model_state_dict'])
    original_model.eval()

    # Prepare inputs
    tokenizer_orig = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    feature_extractor_orig = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    # Tokenize text
    text_inputs = tokenizer_orig(
        test_text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # Process audio
    audio_features = feature_extractor_orig(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    )

    # Get audio features (handle both possible keys)
    if "input_features" in audio_features:
        audio_input = audio_features["input_features"]
    elif "input_values" in audio_features:
        audio_input = audio_features["input_values"]

    audio_mask = audio_features.get("attention_mask", None)

    # Run inference
    with torch.no_grad():
        original_outputs = original_model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            input_values=audio_input,
            audio_attention_mask=audio_mask
        )

    # Extract results
    orig_text_embeds = original_outputs["text_embeds"]
    orig_audio_embeds = original_outputs["audio_embeds"]
    orig_cosine_sim = torch.nn.functional.cosine_similarity(
        orig_text_embeds, orig_audio_embeds, dim=-1
    )

    print("   ✓ Original model inference complete")
    print(f"   - Text embeds shape: {orig_text_embeds.shape}")
    print(f"   - Audio embeds shape: {orig_audio_embeds.shape}")
    print(f"   - Cosine similarity: {orig_cosine_sim.item():.6f}")

    if "alignment_scores" in original_outputs and original_outputs["alignment_scores"] is not None:
        orig_alignment = torch.sigmoid(original_outputs["alignment_scores"]).mean().item()
        print(f"   - Mean alignment score: {orig_alignment:.6f}")
    else:
        orig_alignment = None

    original_success = True

except Exception as e:
    print(f"   ✗ Error with original implementation: {e}")
    import traceback
    traceback.print_exc()
    original_success = False

# ===== HUGGINGFACE IMPLEMENTATION =====
print("\n3. Running HUGGINGFACE implementation...")

try:
    from transformers import AutoConfig, AutoModel
    from configuration_wave import WAVeConfig
    from modeling_wave import WAVeForQualityVerification
    from processing_wave import WAVeProcessor
    from transformers import AutoTokenizer, AutoFeatureExtractor

    # Register model
    AutoConfig.register("wave", WAVeConfig)
    AutoModel.register(WAVeConfig, WAVeForQualityVerification)

    # Load processor
    tokenizer_hf = AutoTokenizer.from_pretrained('./wave-portuguese')
    feature_extractor_hf = AutoFeatureExtractor.from_pretrained('./wave-portuguese')
    processor_hf = WAVeProcessor(tokenizer=tokenizer_hf, feature_extractor=feature_extractor_hf)

    # Load model
    hf_model = AutoModel.from_pretrained('./wave-portuguese')
    hf_model.eval()

    # Process inputs
    inputs_hf = processor_hf(
        text=test_text,
        audio=audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    )

    # Run inference
    with torch.no_grad():
        hf_outputs = hf_model(**inputs_hf)

    # Extract results
    hf_text_embeds = hf_outputs.text_embeds
    hf_audio_embeds = hf_outputs.audio_embeds
    hf_cosine_sim = hf_outputs.cosine_similarity
    hf_quality_score = hf_outputs.quality_score

    print("   ✓ HuggingFace model inference complete")
    print(f"   - Text embeds shape: {hf_text_embeds.shape}")
    print(f"   - Audio embeds shape: {hf_audio_embeds.shape}")
    print(f"   - Cosine similarity: {hf_cosine_sim.item():.6f}")
    print(f"   - Quality score: {hf_quality_score.item():.6f}")

    if hf_outputs.mean_alignment_score is not None:
        hf_alignment = hf_outputs.mean_alignment_score.item()
        print(f"   - Mean alignment score: {hf_alignment:.6f}")
    else:
        hf_alignment = None

    hf_success = True

except Exception as e:
    print(f"   ✗ Error with HuggingFace implementation: {e}")
    import traceback
    traceback.print_exc()
    hf_success = False

# ===== COMPARISON =====
print("\n" + "=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)

if original_success and hf_success:
    print("\n✓ Both implementations ran successfully!")

    # Compare cosine similarity
    print("\n📊 Cosine Similarity Comparison:")
    print(f"   Original:     {orig_cosine_sim.item():.6f}")
    print(f"   HuggingFace:  {hf_cosine_sim.item():.6f}")
    cosine_diff = abs(orig_cosine_sim.item() - hf_cosine_sim.item())
    print(f"   Difference:   {cosine_diff:.6f}")

    if cosine_diff < 0.001:
        print("   ✓ EXCELLENT: Differences < 0.001 (essentially identical)")
    elif cosine_diff < 0.01:
        print("   ✓ GOOD: Differences < 0.01 (minor numerical differences)")
    elif cosine_diff < 0.05:
        print("   ⚠ ACCEPTABLE: Differences < 0.05 (acceptable range)")
    else:
        print("   ✗ WARNING: Differences > 0.05 (investigate)")

    # Compare embeddings
    print("\n📊 Embedding Comparison:")
    text_embed_diff = torch.abs(orig_text_embeds - hf_text_embeds).mean().item()
    audio_embed_diff = torch.abs(orig_audio_embeds - hf_audio_embeds).mean().item()

    print(f"   Text embedding MAE:  {text_embed_diff:.6f}")
    print(f"   Audio embedding MAE: {audio_embed_diff:.6f}")

    if text_embed_diff < 0.001 and audio_embed_diff < 0.001:
        print("   ✓ EXCELLENT: Embeddings are essentially identical")
    elif text_embed_diff < 0.01 and audio_embed_diff < 0.01:
        print("   ✓ GOOD: Minor numerical differences in embeddings")
    else:
        print("   ⚠ Check: Embeddings have some differences")

    # Compare alignment scores
    if orig_alignment is not None and hf_alignment is not None:
        print("\n📊 Alignment Score Comparison:")
        print(f"   Original:     {orig_alignment:.6f}")
        print(f"   HuggingFace:  {hf_alignment:.6f}")
        alignment_diff = abs(orig_alignment - hf_alignment)
        print(f"   Difference:   {alignment_diff:.6f}")

        if alignment_diff < 0.001:
            print("   ✓ EXCELLENT: Alignment scores match perfectly")
        elif alignment_diff < 0.01:
            print("   ✓ GOOD: Minor differences in alignment")
        else:
            print("   ⚠ Check: Notable differences in alignment")

    # Overall verdict
    print("\n" + "=" * 80)
    print("VERDICT:")
    if cosine_diff < 0.01 and text_embed_diff < 0.01 and audio_embed_diff < 0.01:
        print("✅ HuggingFace implementation is CORRECT!")
        print("The converted model produces essentially identical results to the original.")
    elif cosine_diff < 0.05:
        print("✅ HuggingFace implementation is ACCEPTABLE!")
        print("Minor numerical differences are within acceptable range.")
    else:
        print("⚠️  HuggingFace implementation needs investigation!")
        print("Differences exceed acceptable thresholds.")
    print("=" * 80)

elif not original_success:
    print("\n✗ Original implementation failed to run")
    print("Cannot perform comparison without both implementations working")

elif not hf_success:
    print("\n✗ HuggingFace implementation failed to run")
    print("The conversion may need additional fixes")

print()
