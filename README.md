# Speech-Transcript Embedding Alignment

This repository contains the implementation and experiments for learning aligned embeddings between speech audio and text transcripts using word-level alignment mechanisms.

## Repository Structure

This repository is organized into several main directories, each containing specific components of our research:

### 📁 [training_multimodal/](./training_multimodal/)
Contains the core multimodal embedding training implementation with word-level alignment:
- **Core model architecture** with word alignment module
- **Training scripts** for the alignment-based embedding model
- **Inference and evaluation** scripts for synthetic data
- **Dataset creation and processing** utilities
- **Trained model results** for Dutch and Portuguese languages

### 📁 [training_ASR/](./training_ASR/)
Contains Automatic Speech Recognition (ASR) training and evaluation:
- **Whisper fine-tuning** scripts for multiple languages
- **Evaluation frameworks** for ASR models on various datasets
- **Baseline ASR experiments** supporting the main multimodal work

### 📁 [synthetic_audio/](./synthetic_audio/)
Contains synthetic audio generation and verification:
- **Transcript and Text-to-Speech (TTS) generation** for creating synthetic training data
- **Dataset verification** and quality control scripts
- **Audio processing utilities** for synthetic data creation

### 📁 [multimodal_training_logs/](./multimodal_training_logs/)
Contains training logs and visualization scripts:
- **Training progress logs** for all multimodal experiments
- **Plotting and visualization** scripts for training metrics
- **Performance analysis** tools

## Key Features

- **Word-Level Alignment**: Core mechanism for aligning speech and text representations at the word level
- **Multilingual Support**: Experiments on Dutch and Portuguese languages
- **Synthetic Data Integration**: Leveraging synthetic audio for improved training
- **Comprehensive Evaluation**: Both intrinsic alignment metrics and downstream ASR performance

## Quick Start

1. **For multimodal embedding training**: See [`training_multimodal/README.md`](./training_multimodal/README.md)
2. **For ASR training and evaluation**: See [`training_ASR/README.md`](./training_ASR/README.md)
3. **For synthetic audio generation**: See [`synthetic_audio/README.md`](./synthetic_audio/README.md)

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-compatible GPU (recommended)

Detailed requirements are provided in each subfolder's README.
