# WAVe: Word-Aligned Verification of Synthetic Speech for ASR

**Solving the Synthetic Audio Quality Bottleneck**

Training robust ASR systems requires massive amounts of high-quality audio data. Synthetic audio generation offers a promising solution, but introduces a critical challenge: *how do you automatically identify when synthesized audio is good enough to train on?*

## Why WAVe?

**The Problem:** Not all synthetic audio is created equal. Poor-quality synthetic samples can degrade ASR model performance, yet manually filtering thousands of hours of audio is impractical. Traditional approaches rely on ASR metrics (WER/CER) or simple heuristics, which often fail to capture subtle quality issues.

**Our Solution:** WAVe introduces a novel multimodal embedding model that learns to measure speech-transcript alignment quality at the **word level**. By learning what "good alignment" looks like from real data, WAVe effectively identifies synthetic samples that deviate from natural speech patterns.

**The Results:**
- **34% reduction** in training steps while maintaining or improving ASR performance
- **Up to 50% improvement** in cross-domain generalization (e.g., MLS benchmark: 13.54% → 6.89% WER)
- **30% less synthetic data** needed compared to previous filtering methods, with superior results
- Effective detection of localized synthesis errors that sentence-level methods miss

### Key Innovation: Word-Level Alignment

Unlike conventional sentence-level filtering, WAVe operates at finer granularity through attention-based word-level alignment. This enables detection of:
- Unnatural prosody or timing
- Mispronunciations or incorrect audio synthesis
- Text-audio mismatches
- Poor audio quality that would hurt ASR training

The result? **Cleaner synthetic training data → Better downstream ASR models → Faster training.**

---

## 🔗 Trained Models

All trained models are available on Hugging Face: **[yuriyvnv](https://huggingface.co/yuriyvnv)**

Including:
- WAVe multimodal embedding models (Dutch & Portuguese)
- Fine-tuned Whisper models (Tiny, Small, Large-v3) across multiple configurations
- Models trained on filtered synthetic data, unfiltered data, and CommonVoice-only baselines

---

## Supplementary Materials

The following pdf files help guide the user how the models were trained to replicate our results. 
A static seed of 42 was used across all runs.

- **Corruption_Strategies.pdf**: Details the corruption strategies used for training WAVe model
- **synthetic_data_generation_prompts.pdf**: Describes the prompts and methodology for generating synthetic training data
- **WAVe_Configurations_training.pdf**: Configuration details for WAVe model training
- **Whisper_Finetuning_config.pdf**: Configuration specifications for Whisper model fine-tuning

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
