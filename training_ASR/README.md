# Automatic Speech Recognition (ASR) Training

This directory contains the implementation for fine-tuning Whisper models on various languages and datasets, serving as baseline experiments for the main multimodal alignment work.

## Overview

The ASR training component focuses on:
- Fine-tuning Whisper models for Dutch and Portuguese languages
- Training on both real and synthetic audio data
- Comprehensive evaluation across multiple test sets
- Supporting the main multimodal embedding alignment research

## Directory Structure

```
training_ASR/
├── trainer_whisper/          # Core training scripts
│   ├── hf_trainer_complete.py         # Main Whisper training script for Dutch
│   ├── hf_trainer_complete_capes.py   # Whisper training for Portuguese (CAPES dataset)
│   └── train_and_push.sh              # Training automation script
├── evaluate_on_subsets/      # Evaluation frameworks
│   ├── evaluate_librispeech_nl.py     # Dutch LibriSpeech evaluation
│   ├── evaluate_librispeech_pt.py     # Portuguese LibriSpeech evaluation
│   ├── evaluate_results.py            # Single model evaluation
│   ├── evaluate_cv.sh                 # Common Voice evaluation automation
│   ├── evaluate_multi_nl.sh           # Multi-model Dutch evaluation
│   └── evaluate_multi_pt.sh           # Multi-model Portuguese evaluation
└── README.md                 # This file
```

## Training Scripts

### Core Training (`trainer_whisper/`)

**`hf_trainer_complete.py`**
- Fine-tunes Whisper models on Dutch synthetic transcript data
- Supports various Whisper model sizes (tiny, small, large-v3)
- Uses mixed training data combining synthetic and real audio
- Implements comprehensive evaluation during training

**`hf_trainer_complete_capes.py`**
- Fine-tunes Whisper models on Portuguese CAPES dataset
- Specialized for Portuguese language with synthetic audio
- Includes data preprocessing and augmentation

**Key Features:**
- Gradient checkpointing for memory efficiency
- Mixed precision training (bf16)
- Automatic model pushing to HuggingFace Hub
- Comprehensive logging and metrics tracking

### Training Configuration

```python
# Example training parameters
training_args = Seq2SeqTrainingArguments(
    per_device_train_batch_size=256,
    learning_rate=5e-6,
    max_steps=1000,
    warmup_ratio=0.1,
    bf16=True,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50
)
```

## Evaluation Framework (`evaluate_on_subsets/`)

### Multi-Dataset Evaluation

**`evaluate_librispeech_nl.py` / `evaluate_librispeech_pt.py`**
- Evaluate multiple Whisper models on LibriSpeech test sets
- Support for various model variants:
  - `ANONYMOUS_USER/whisper-tiny-mixed-nl`
  - `ANONYMOUS_USER/whisper-tiny-cv-only-nl`
  - `ANONYMOUS_USER/whisper-small-mixed-pt`
  - And more...

**`evaluate_results.py`**
- Single model evaluation script
- Detailed WER (Word Error Rate) computation
- Supports custom model paths and datasets

### Automation Scripts

**`evaluate_cv.sh`**
```bash
# Automated evaluation with git tracking
python evaluate_results.py
git add .
git commit -m "Evaluation done for model ANONYMOUS_USER/whisper-model: $(date)"
git push
```

## Usage Examples

### Training a New Model

```bash
# Dutch model training
cd trainer_whisper/
python hf_trainer_complete.py

# Portuguese model training  
python hf_trainer_complete_capes.py
```

### Evaluating Models

```bash
# Evaluate on LibriSpeech Dutch
cd evaluate_on_subsets/
python evaluate_librispeech_nl.py

# Evaluate single model
python evaluate_results.py

# Automated evaluation with tracking
./evaluate_cv.sh
```

### Batch Evaluation

```bash
# Evaluate multiple Dutch models
./evaluate_multi_nl.sh

# Evaluate multiple Portuguese models
./evaluate_multi_pt.sh
```

## Model Variants

The training produces several model variants:

### Dutch Models
- **whisper-tiny-mixed-nl**: Tiny model trained on mixed synthetic/real data
- **whisper-tiny-cv-only-nl**: Tiny model trained only on Common Voice
- **whisper-small-mixed-nl**: Small model with mixed training data

### Portuguese Models  
- **whisper-small-mixed-pt**: Small model trained on mixed Portuguese data
- **whisper-large-v3-pt**: Large model fine-tuned for Portuguese

## Dataset Integration

### Training Data Sources
- **Common Voice 17.0**: Real speech recordings
- **CAPES**: Portuguese speech corpus
- **Synthetic Audio**: Generated using TTS (see `synthetic_audio/`)

### Data Processing
- Automatic resampling to 16kHz
- Text normalization and punctuation handling
- Audio-transcript alignment verification

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
librosa>=0.10.0
jiwer>=3.0.0
wandb>=0.15.0
```

## Key Features

- **Mixed Training Paradigm**: Combines synthetic and real audio data
- **Multi-language Support**: Dutch and Portuguese implementations
- **Comprehensive Evaluation**: Multiple test sets and metrics
- **Memory Efficient**: Gradient checkpointing and optimized batch sizes
- **Experiment Tracking**: Weights & Biases integration
- **Model Hub Integration**: Automatic model uploads

## Results Integration

These ASR models serve as baselines and comparison points for the main multimodal embedding alignment work. The trained models are used in:

1. **Downstream Task Evaluation**: Testing multimodal embeddings on ASR tasks
2. **Performance Comparison**: Comparing alignment-based vs. traditional ASR
3. **Data Quality Assessment**: Evaluating synthetic vs. real audio effectiveness

## Notes

- Models are automatically pushed to HuggingFace Hub during training
- All personal identifiers have been anonymized to `ANONYMOUS_USER`
- Training logs and metrics are preserved for reproducibility
- GPU memory optimization included for large model training