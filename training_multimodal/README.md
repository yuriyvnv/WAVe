# Multimodal Speech-Text Embedding Training

This directory contains the core implementation of our multimodal embedding model that learns aligned representations between speech audio and text transcripts using **word-level alignment mechanisms**. The primary goal is to learn embeddings that can effectively filter bad synthetic audio samples by measuring speech-transcript alignment quality.

## Overview

- **Word-Level Alignment**: Core mechanism that aligns word-level text representations with temporal segments in audio
- **Synthetic Data Filtering**: Learning embeddings that can identify and filter poor-quality synthetic audio samples
- **Separate Language Models**: Independent training for Dutch and Portuguese languages
- **Real Data Evaluation**: Models evaluated on Common Voice 17.0 validation and test sets (real audio data)

## Directory Structure

```
training_multimodal/
├── trainer_multimodal/           # Core training implementation
├── subsets_creation/             # Dataset preparation and management
├── inference_synthetic_data/     # Model inference and evaluation
├── 3_alignment_MHGLU_Dutch/      # Training results with word alignment (Dutch)
├── 3_alignment_MHGLU_Portuguese/ # Training results with word alignment (Portuguese)
├── 3layers_NO_Alignment_Dutch/   # Baseline without alignment (Dutch)
└── 3layers_NO_Alignment_Portuguese/  # Baseline without alignment (Portuguese)
```

## Core Architecture

### Word-Level Alignment Module

```python
class WordLevelAlignmentModule(nn.Module):
    """
    Module to align word-level representations from text with temporal segments in audio.
    Critical for identifying poor-quality synthetic audio samples.
    """
```

**Key Features:**
- Attention-based alignment between text tokens and audio frames
- Quality assessment through alignment scoring
- Synthetic data filtering capability

### Enhanced Audio-Text Model

1. **Text Encoder**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
2. **Audio Encoder**: `facebook/w2v-bert-2.0`
3. **Word Alignment Module**: Core contribution for synthetic data quality assessment
4. **Projection Layers**: Map to shared embedding space

## Training Configuration

### Language-Specific Training

**Dutch Model:**
- Dataset: `ANONYMOUS_USER/synthetic_transcript_nl`
- Evaluation: Common Voice 17.0 Dutch validation/test sets
- Output: `3_alignment_MHGLU_Dutch/`

**Portuguese Model:**
- Dataset: `ANONYMOUS_USER/synthetic_transcript_pt`
- Evaluation: Common Voice 17.0 Portuguese validation/test sets
- Output: `3_alignment_MHGLU_Portuguese/`

### Training Parameters
- Batch Size: 8 (with gradient accumulation)
- Learning Rate: 1e-5 with linear warmup
- Mixed Precision: FP16
- Encoder Freezing: Partial (last 5 layers unfrozen)

## Usage

### Training

```bash
# Dutch model
python trainer_unfreeze.py \
    --output_dir="3_alignment_MHGLU_Dutch" \
    --dataset_name="ANONYMOUS_USER/synthetic_transcript_nl" \
    --use_word_alignment \
    --language="nl"

# Portuguese model
python trainer_unfreeze.py \
    --output_dir="3_alignment_MHGLU_Portuguese" \
    --dataset_name="ANONYMOUS_USER/synthetic_transcript_pt" \
    --use_word_alignment \
    --language="pt"
```

### Synthetic Data Filtering

```bash
# Filter synthetic Dutch data
python inference_on_synthetic_data_ours.py

# Filter synthetic Portuguese data
python inference_on_synthetic_capes.py
```

## Key Components

### `trainer_multimodal/`
- `trainer_unfreeze.py`: Main training script with word alignment
- Training automation and model upload scripts

### `subsets_creation/`
- Dataset creation and mixing utilities
- HuggingFace dataset management
- Quality filtering for synthetic data

### `inference_synthetic_data/`
- Model inference on synthetic datasets
- Quality assessment and filtering tools
- Batch processing for large datasets

### Model Results
Each results directory contains:
- `test_metrics.json`: Evaluation metrics on Common Voice 17.0
- Training progress visualizations
- Model checkpoints and saved states

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
datasets>=2.12.0
librosa>=0.10.0
```