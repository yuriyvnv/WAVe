# Multimodal Training Logs

This directory contains training logs and visualization scripts for the multimodal embedding experiments.

## Overview

- **Training Logs**: Detailed logs from multimodal embedding training sessions
- **Visualization Scripts**: Tools for plotting training progress and metrics
- **Performance Analysis**: Scripts for analyzing model performance across languages
- **Similarity Transformation**: Special handling for Portuguese scaled probability logs

## Files

### Training Logs

**`training_embedding_dutch.log`**
- Training log for Dutch multimodal embedding model with word alignment
- Contains loss progression, validation metrics, and training statistics

**`training_embedding_noAlignment_dutch.log`**
- Training log for Dutch baseline model without word alignment
- Used for comparison with alignment-based approach

**`training_embedding_noAlignment_PT.log`**
- Training log for Portuguese baseline model without word alignment
- Baseline comparison for Portuguese experiments

**`training_log_PT.log`**
- Training log for Portuguese multimodal embedding model with word alignment
- Contains Portuguese-specific training metrics and progress
- **Note**: Similarities stored as scaled probabilities with T=0.1

### Visualization Scripts

**`plotting_dutch.py`**
- Generates training progress plots for Dutch experiments
- Creates visualizations comparing alignment vs. non-alignment approaches
- Plots loss curves, similarity distributions, and alignment metrics

**`plotting_portuguese.py`**
- Generates training progress plots for Portuguese experiments
- Creates comparative visualizations for Portuguese models
- Includes similarity gap analysis and alignment effectiveness plots
- **Special Feature**: Transforms scaled probabilities back to normalized similarities (0-1)

## Portuguese Similarity Transformation

The Portuguese logs contain similarities stored as scaled probabilities with temperature T=0.1. The plotting script includes this transformation function:

```python
T = 0.1
def to_plain(arr):
    arr = np.asarray(arr)
    cos = T * np.log(arr / (1 - arr))
    return np.clip((cos + 1) / 2, 0, 1)
```

## Usage

### Generate Training Plots

```bash
# Generate Dutch training visualizations
python plotting_dutch.py

# Generate Portuguese training visualizations (with similarity transformation)
python plotting_portuguese.py
```

## Log Format

Training logs contain:
- **Epoch Information**: Training progress by epoch
- **Loss Values**: Training and validation losses
- **Similarity Metrics**: Clean vs. corrupt audio similarity scores
- **Alignment Scores**: Word-level alignment effectiveness
- **Memory Usage**: GPU memory consumption tracking
- **Timing Information**: Training duration and performance metrics

**Portuguese-Specific**: Similarity values require transformation from scaled probabilities (T=0.1) to normalized similarities (0-1).

## Visualization Output

Generated plots include:
- **Loss Progression**: Training and validation loss over epochs
- **Similarity Gap**: Difference between clean and corrupt audio similarities
- **Alignment Effectiveness**: Word alignment quality over training
- **Comparative Analysis**: Alignment vs. non-alignment model performance
- **Normalized Similarities**: Properly transformed Portuguese similarity distributions

## Requirements

```
matplotlib>=3.5.0
numpy>=1.21.0
pandas>=1.3.0
seaborn>=0.11.0
```