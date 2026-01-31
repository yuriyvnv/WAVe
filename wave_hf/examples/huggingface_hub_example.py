#!/usr/bin/env python3
"""
Example: Using WAVe directly from HuggingFace Hub

This example shows how to:
1. Load the model directly from HuggingFace Hub (no local files needed)
2. Process real audio files with their transcripts
3. Get quality scores for synthetic speech assessment
4. Filter datasets based on quality thresholds

Requirements:
    pip install transformers torch torchaudio datasets
"""

import torch
import torchaudio
from transformers import AutoModel, AutoProcessor
from pathlib import Path


def load_model_from_hub():
    """Load WAVe model and processor from HuggingFace Hub."""
    print("Loading WAVe model from HuggingFace Hub...")
    print("Model: yuriyvnv/WAVe-1B-Multimodal-PT")

    # Load processor and model (trust_remote_code required for custom architecture)
    processor = AutoProcessor.from_pretrained(
        "yuriyvnv/WAVe-1B-Multimodal-PT",
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        "yuriyvnv/WAVe-1B-Multimodal-PT",
        trust_remote_code=True
    )

    model.eval()  # Set to evaluation mode

    print("✓ Model loaded successfully!\n")
    return processor, model


def assess_single_audio(processor, model, audio_path, text):
    """
    Assess quality of a single audio-text pair.

    Args:
        processor: WAVe processor
        model: WAVe model
        audio_path: Path to audio file (.wav, .mp3, .flac, etc.)
        text: Transcript text

    Returns:
        Dictionary with quality metrics
    """
    # Load audio file
    audio, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        audio = resampler(audio)
        sample_rate = 16000

    # Convert to numpy array (processor expects this)
    audio = audio.squeeze().numpy()

    # Process inputs
    inputs = processor(
        text=text,
        audio=audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True
    )

    # Get quality assessment
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract metrics
    quality_score = outputs.quality_score.item()
    cosine_similarity = outputs.cosine_similarity.item()
    mean_alignment = outputs.mean_alignment_score.item()

    # Per-word alignment scores
    word_scores = outputs.alignment_scores.squeeze().numpy()

    return {
        "quality_score": quality_score,
        "cosine_similarity": cosine_similarity,
        "mean_alignment": mean_alignment,
        "word_scores": word_scores,
        "text": text,
        "audio_path": str(audio_path)
    }


def interpret_quality(quality_score):
    """Interpret quality score and provide recommendation."""
    if quality_score >= 0.8:
        return "HIGH", "✅ Safe for ASR training", "green"
    elif quality_score >= 0.5:
        return "MEDIUM", "⚠️  Use with caution / Review manually", "yellow"
    else:
        return "LOW", "❌ Discard from training set", "red"


def example_1_single_file():
    """Example 1: Assess a single audio file."""
    print("=" * 80)
    print("Example 1: Assessing a Single Audio File")
    print("=" * 80)

    # Load model
    processor, model = load_model_from_hub()

    # Example: Replace with your audio file path
    audio_path = "path/to/your/audio.wav"  # e.g., "sample_speech.wav"
    text = "Olá, como você está?"  # Portuguese transcript

    print(f"Audio file: {audio_path}")
    print(f"Text: {text}\n")

    # Check if file exists (for demo purposes)
    if not Path(audio_path).exists():
        print("⚠️  Warning: Example audio file not found!")
        print("Replace 'path/to/your/audio.wav' with your actual audio file.\n")
        return

    # Assess quality
    results = assess_single_audio(processor, model, audio_path, text)

    # Display results
    print("Results:")
    print("-" * 80)
    print(f"Quality Score:       {results['quality_score']:.3f}")
    print(f"Cosine Similarity:   {results['cosine_similarity']:.3f}")
    print(f"Mean Alignment:      {results['mean_alignment']:.3f}")

    # Interpretation
    quality_level, recommendation, _ = interpret_quality(results['quality_score'])
    print(f"\nQuality Level: {quality_level}")
    print(f"Recommendation: {recommendation}")
    print("=" * 80)


def example_2_batch_processing():
    """Example 2: Process multiple audio files in batch."""
    print("\n" + "=" * 80)
    print("Example 2: Batch Processing Multiple Files")
    print("=" * 80)

    # Load model
    processor, model = load_model_from_hub()

    # Example dataset: list of (audio_path, text) pairs
    dataset = [
        ("audio1.wav", "Olá, como você está?"),
        ("audio2.wav", "Este é um teste de qualidade."),
        ("audio3.wav", "A síntese de voz está funcionando bem."),
    ]

    results = []
    high_quality_count = 0

    print(f"Processing {len(dataset)} audio files...\n")

    for i, (audio_path, text) in enumerate(dataset, 1):
        print(f"[{i}/{len(dataset)}] {audio_path}")

        # Check if file exists
        if not Path(audio_path).exists():
            print(f"  ⚠️  File not found, skipping...\n")
            continue

        # Assess quality
        result = assess_single_audio(processor, model, audio_path, text)
        results.append(result)

        # Display result
        quality_level, recommendation, _ = interpret_quality(result['quality_score'])
        print(f"  Quality: {result['quality_score']:.3f} ({quality_level})")
        print(f"  {recommendation}\n")

        if result['quality_score'] >= 0.8:
            high_quality_count += 1

    # Summary
    if results:
        print("-" * 80)
        print(f"Summary:")
        print(f"  Total processed: {len(results)}")
        print(f"  High quality (≥0.8): {high_quality_count} ({100*high_quality_count/len(results):.1f}%)")
        print(f"  Average quality: {sum(r['quality_score'] for r in results)/len(results):.3f}")

    print("=" * 80)


def example_3_huggingface_dataset():
    """Example 3: Filter a HuggingFace dataset."""
    print("\n" + "=" * 80)
    print("Example 3: Filtering a HuggingFace Dataset")
    print("=" * 80)

    try:
        from datasets import load_dataset
    except ImportError:
        print("⚠️  Please install datasets: pip install datasets")
        return

    # Load model
    processor, model = load_model_from_hub()

    # Example: Load a small Portuguese speech dataset
    # Replace with your actual dataset
    print("Loading dataset from HuggingFace...")
    print("(Using CommonVoice Portuguese as example)\n")

    # Load a small subset
    dataset = load_dataset(
        "mozilla-foundation/common_voice_16_1",
        "pt",
        split="test[:10]",  # Only 10 samples for demo
        trust_remote_code=True
    )

    print(f"Dataset loaded: {len(dataset)} samples")

    # Filter function
    def quality_filter(example):
        """Filter function that adds quality score to each example."""
        audio_array = example["audio"]["array"]
        text = example["sentence"]
        sample_rate = example["audio"]["sampling_rate"]

        # Process
        inputs = processor(
            text=text,
            audio=audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )

        # Get quality
        with torch.no_grad():
            outputs = model(**inputs)

        quality = outputs.quality_score.item()

        # Add quality to example
        example["quality_score"] = quality

        # Return True if high quality
        return quality >= 0.8

    # Apply filter
    print("\nFiltering dataset (keeping only quality ≥ 0.8)...")
    filtered_dataset = dataset.filter(quality_filter)

    print(f"\nResults:")
    print(f"  Original size: {len(dataset)}")
    print(f"  Filtered size: {len(filtered_dataset)}")
    print(f"  Kept: {100*len(filtered_dataset)/len(dataset):.1f}%")

    print("=" * 80)


def example_4_with_generated_audio():
    """Example 4: Using with generated/synthetic audio (no file needed)."""
    print("\n" + "=" * 80)
    print("Example 4: Working with Generated Audio Arrays")
    print("=" * 80)

    # Load model
    processor, model = load_model_from_hub()

    # Simulate synthetic audio (e.g., from TTS system)
    # In practice, this would come from your TTS model
    sample_rate = 16000
    duration = 2.0  # seconds
    audio_array = torch.randn(int(sample_rate * duration)).numpy()  # Random for demo

    text = "Este é um exemplo de áudio sintético."

    print("Processing synthetic audio array...")
    print(f"Text: {text}")
    print(f"Audio: {len(audio_array)} samples @ {sample_rate}Hz ({duration}s)\n")

    # Process
    inputs = processor(
        text=text,
        audio=audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt"
    )

    # Assess quality
    with torch.no_grad():
        outputs = model(**inputs)

    quality_score = outputs.quality_score.item()

    print(f"Quality Score: {quality_score:.3f}")

    quality_level, recommendation, _ = interpret_quality(quality_score)
    print(f"Quality Level: {quality_level}")
    print(f"Recommendation: {recommendation}")

    print("\nNote: This example uses random noise, so quality will be low.")
    print("Replace with actual TTS output for realistic assessment.")
    print("=" * 80)


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "WAVe: HuggingFace Hub Usage Examples" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Example 4 doesn't require any files
    example_4_with_generated_audio()

    # Uncomment these when you have audio files:
    # example_1_single_file()
    # example_2_batch_processing()
    # example_3_huggingface_dataset()

    print("\n✅ All examples completed!")
    print("\nTo run examples 1-3, you need to:")
    print("  1. Prepare audio files (.wav, .mp3, .flac)")
    print("  2. Update the file paths in the code")
    print("  3. Uncomment the example function calls in main()")
    print()


if __name__ == "__main__":
    main()
