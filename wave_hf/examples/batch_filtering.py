"""
Batch filtering example for synthetic datasets.

This script shows how to filter a large dataset of synthetic audio-text pairs
using WAVe quality scores.
"""

import torch
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from datasets import load_from_disk, Dataset
import numpy as np


def filter_dataset(
    dataset,
    model,
    processor,
    threshold=0.8,
    batch_size=32,
    device="cuda",
):
    """
    Filter dataset based on WAVe quality scores.

    Args:
        dataset: HuggingFace dataset with 'text' and 'audio' columns
        model: WAVe model
        processor: WAVe processor
        threshold: Quality threshold (0-1)
        batch_size: Batch size for processing
        device: Device to run on

    Returns:
        filtered_dataset: Dataset with only high-quality samples
        statistics: Dictionary with filtering statistics
    """
    model.eval()
    model.to(device)

    all_scores = []
    all_indices = []

    print(f"Processing {len(dataset)} samples...")

    # Process in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Filtering"):
        batch_end = min(i + batch_size, len(dataset))
        batch = dataset[i:batch_end]

        # Process batch
        inputs = processor(
            text=batch["text"],
            audio=batch["audio"],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get quality scores
        with torch.no_grad():
            outputs = model(**inputs)

        scores = outputs.quality_score.cpu().numpy()
        all_scores.extend(scores)

        # Track indices of good samples
        for j, score in enumerate(scores):
            if score >= threshold:
                all_indices.append(i + j)

    all_scores = np.array(all_scores)

    # Filter dataset
    filtered_dataset = dataset.select(all_indices)

    # Compute statistics
    statistics = {
        "total_samples": len(dataset),
        "filtered_samples": len(filtered_dataset),
        "filtered_percentage": 100 * len(filtered_dataset) / len(dataset),
        "threshold": threshold,
        "quality_scores": {
            "mean": float(all_scores.mean()),
            "median": float(np.median(all_scores)),
            "std": float(all_scores.std()),
            "min": float(all_scores.min()),
            "max": float(all_scores.max()),
        },
        "quality_distribution": {
            "high (≥0.8)": int((all_scores >= 0.8).sum()),
            "medium (0.5-0.8)": int(((all_scores >= 0.5) & (all_scores < 0.8)).sum()),
            "low (<0.5)": int((all_scores < 0.5).sum()),
        }
    }

    return filtered_dataset, statistics, all_scores


def main():
    parser = argparse.ArgumentParser(description="Filter synthetic dataset with WAVe")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for filtered dataset")
    parser.add_argument("--model_name", type=str, default="yuriyvnv/wave-portuguese", help="Model name")
    parser.add_argument("--threshold", type=float, default=0.8, help="Quality threshold")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--save_scores", action="store_true", help="Save all quality scores")

    args = parser.parse_args()

    print("=" * 80)
    print("WAVe Batch Filtering")
    print("=" * 80)

    # Load dataset
    print(f"\nLoading dataset from: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Load model
    print(f"\nLoading model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Filter dataset
    print(f"\nFiltering with threshold: {args.threshold}")
    filtered_dataset, statistics, all_scores = filter_dataset(
        dataset=dataset,
        model=model,
        processor=processor,
        threshold=args.threshold,
        batch_size=args.batch_size,
        device=device
    )

    # Print statistics
    print("\n" + "=" * 80)
    print("Filtering Results")
    print("=" * 80)
    print(f"\nTotal samples: {statistics['total_samples']}")
    print(f"Filtered samples: {statistics['filtered_samples']}")
    print(f"Kept: {statistics['filtered_percentage']:.2f}%")

    print(f"\nQuality Scores:")
    print(f"  Mean: {statistics['quality_scores']['mean']:.4f}")
    print(f"  Median: {statistics['quality_scores']['median']:.4f}")
    print(f"  Std: {statistics['quality_scores']['std']:.4f}")
    print(f"  Range: [{statistics['quality_scores']['min']:.4f}, {statistics['quality_scores']['max']:.4f}]")

    print(f"\nQuality Distribution:")
    print(f"  High (≥0.8): {statistics['quality_distribution']['high (≥0.8)']} samples")
    print(f"  Medium (0.5-0.8): {statistics['quality_distribution']['medium (0.5-0.8)']} samples")
    print(f"  Low (<0.5): {statistics['quality_distribution']['low (<0.5)']} samples")

    # Save filtered dataset
    print(f"\nSaving filtered dataset to: {args.output_path}")
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_dataset.save_to_disk(str(output_dir / "dataset"))

    # Save statistics
    with open(output_dir / "statistics.json", "w") as f:
        json.dump(statistics, f, indent=2)

    # Optionally save all scores
    if args.save_scores:
        np.save(output_dir / "quality_scores.npy", all_scores)
        print(f"Saved quality scores to: {output_dir / 'quality_scores.npy'}")

    print("\n" + "=" * 80)
    print("Filtering complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
