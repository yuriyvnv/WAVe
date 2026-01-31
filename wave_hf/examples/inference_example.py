"""
Basic inference example with WAVe model.

This script demonstrates how to:
1. Load a WAVe model from HuggingFace Hub
2. Process audio-text pairs
3. Get quality scores and detailed outputs
"""

import torch
import numpy as np
from transformers import AutoProcessor, AutoModel


def main():
    print("=" * 80)
    print("WAVe Inference Example")
    print("=" * 80)

    # ===== LOAD MODEL =====
    print("\n1. Loading model and processor...")

    model_name = "yuriyvnv/WAVe-1B-Multimodal-PT"  
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"   Model loaded on: {device}")

    # ===== PREPARE INPUT =====
    print("\n2. Preparing inputs...")

    # Example: Synthetic Portuguese audio
    text = "Olá, como você está hoje?"

    # Simulated audio (replace with real audio)
    # In practice: audio, sr = librosa.load("your_audio.wav", sr=16000)
    audio = np.random.randn(16000)  # 1 second at 16kHz

    print(f"   Text: '{text}'")
    print(f"   Audio shape: {audio.shape}")

    # Process inputs
    inputs = processor(
        text=text,
        audio=audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ===== INFERENCE =====
    print("\n3. Running inference...")

    with torch.no_grad():
        outputs = model(**inputs)

    # ===== RESULTS =====
    print("\n4. Results:")
    print("-" * 80)

    # Main quality score
    quality_score = outputs.quality_score.item()
    print(f"\n   Quality Score: {quality_score:.4f}")

    # Interpret
    if quality_score >= 0.8:
        print("   Status: ✓ HIGH QUALITY - Safe for ASR training")
    elif quality_score >= 0.5:
        print("   Status: ⚠ MEDIUM QUALITY - Use with caution")
    else:
        print("   Status: ✗ LOW QUALITY - Discard")

    # Detailed outputs
    print(f"\n   Detailed Metrics:")
    print(f"   - Cosine Similarity: {outputs.cosine_similarity.item():.4f}")

    if outputs.mean_alignment_score is not None:
        print(f"   - Mean Alignment Score: {outputs.mean_alignment_score.item():.4f}")

    # Embeddings
    print(f"\n   Embeddings:")
    print(f"   - Text embedding shape: {outputs.text_embeds.shape}")
    print(f"   - Audio embedding shape: {outputs.audio_embeds.shape}")

    # Word-level information
    if outputs.alignment_scores is not None:
        word_scores = torch.sigmoid(outputs.alignment_scores).cpu().numpy()[0]
        print(f"\n   Word-level Alignment Scores:")
        print(f"   - Number of tokens: {len(word_scores)}")
        print(f"   - Min score: {word_scores.min():.4f}")
        print(f"   - Max score: {word_scores.max():.4f}")
        print(f"   - Mean score: {word_scores.mean():.4f}")

        # Show per-token scores
        print(f"\n   Per-token scores (first 10):")
        for i, score in enumerate(word_scores[:10]):
            print(f"      Token {i}: {score:.4f}")

    # Alignment matrix
    if outputs.alignment_matrix is not None:
        align_matrix = outputs.alignment_matrix.cpu().numpy()[0]
        print(f"\n   Alignment Matrix:")
        print(f"   - Shape: {align_matrix.shape} (text_tokens × audio_frames)")
        print(f"   - Each row shows how a word attends to audio frames")

    print("\n" + "=" * 80)
    print("Inference complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
