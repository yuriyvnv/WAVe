#!/usr/bin/env python3
"""
Convert native PyTorch WAVe checkpoint to HuggingFace format.

This script converts checkpoints from the original trainer_unfreeze.py format
to the HuggingFace Transformers format, enabling use with AutoModel and AutoProcessor.

Usage:
    python convert_checkpoint.py \\
        --checkpoint_path path/to/best_model_loss.pt \\
        --output_dir ./wave-hf-portuguese \\
        --language pt \\
        --push_to_hub

Example:
    # Convert Portuguese model
    python convert_checkpoint.py \\
        --checkpoint_path ../training_multimodal/portuguese_results/best_model_loss.pt \\
        --output_dir ./wave-portuguese \\
        --language pt

    # Convert and push to Hub
    python convert_checkpoint.py \\
        --checkpoint_path ../training_multimodal/dutch_results/best_model_loss.pt \\
        --output_dir ./wave-dutch \\
        --language nl \\
        --push_to_hub \\
        --hub_model_id yuriyvnv/wave-dutch
"""

import torch
import argparse
from pathlib import Path
from typing import Dict, Any
import logging

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from configuration_wave import WAVeConfig
from modeling_wave import WAVeModel, WAVeForQualityVerification
from processing_wave import WAVeProcessor
from transformers import AutoTokenizer, AutoFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_config_from_checkpoint(checkpoint: Dict[str, Any]) -> WAVeConfig:
    """
    Extract WAVeConfig from native checkpoint metadata.

    The original checkpoints store hyperparameters as top-level keys.
    We extract them to create a proper WAVeConfig.

    Args:
        checkpoint: Full checkpoint dict with model_state_dict and metadata

    Returns:
        WAVeConfig instance with parameters from checkpoint
    """
    logger.info("Extracting configuration from checkpoint...")

    # Extract config parameters (use defaults if not present)
    config = WAVeConfig(
        # Encoder paths (these are fixed for trained models)
        text_model_name_or_path="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        audio_model_name_or_path="facebook/w2v-bert-2.0",

        # Projection parameters
        projection_dim=checkpoint.get("projection_dim", 768),
        text_embedding_dim=checkpoint.get("text_embedding_dim", 768),
        audio_embedding_dim=checkpoint.get("audio_embedding_dim", 1024),

        # Alignment parameters
        alignment_dim=checkpoint.get("alignment_dim", 768),
        num_alignment_heads=checkpoint.get("num_alignment_heads", 6),
        n_glu_heads=checkpoint.get("n_glu_heads", 4),

        # Feature flags (CRITICAL: must match training config)
        use_word_alignment=checkpoint.get("use_word_alignment", True),
        use_cross_modal=checkpoint.get("use_cross_modal", False),  # NOT used in training
        use_attentive_pooling=checkpoint.get("use_attentive_pooling", False),  # NOT used in training

        # Training parameters
        dropout=checkpoint.get("dropout", 0.1),
        freeze_encoders=checkpoint.get("freeze_encoders", "partial"),
        text_layers_to_unfreeze=checkpoint.get("text_layers_to_unfreeze", 3),
        audio_layers_to_unfreeze=checkpoint.get("audio_layers_to_unfreeze", 3),
        temperature=checkpoint.get("temperature", 0.1),
    )

    logger.info(f"Configuration extracted:")
    logger.info(f"  - Projection dim: {config.projection_dim}")
    logger.info(f"  - Use word alignment: {config.use_word_alignment}")
    logger.info(f"  - Use cross modal: {config.use_cross_modal}")
    logger.info(f"  - Use attentive pooling: {config.use_attentive_pooling}")

    return config


def convert_state_dict(
    checkpoint_state_dict: Dict[str, torch.Tensor],
    model_type: str = "quality"
) -> Dict[str, torch.Tensor]:
    """
    Convert PyTorch checkpoint state dict to HuggingFace format.

    The native checkpoint has keys like:
        "text_encoder.encoder.layer.0.attention.self.query.weight"

    For WAVeForQualityVerification, we need:
        "wave.text_encoder.encoder.layer.0.attention.self.query.weight"

    Args:
        checkpoint_state_dict: State dict from native checkpoint
        model_type: "base" for WAVeModel, "quality" for WAVeForQualityVerification

    Returns:
        Converted state dict compatible with HF model
    """
    logger.info(f"Converting state dict for {model_type} model...")

    new_state_dict = {}

    for key, value in checkpoint_state_dict.items():
        # For quality model, add "wave." prefix
        if model_type == "quality" and not key.startswith("wave."):
            new_key = f"wave.{key}"
        else:
            new_key = key

        new_state_dict[new_key] = value

    logger.info(f"Converted {len(new_state_dict)} parameters")

    return new_state_dict


def create_model_card(
    output_dir: str,
    checkpoint_info: dict,
    config: WAVeConfig,
    language: str,
):
    """Create README.md model card for HuggingFace Hub."""

    language_names = {
        "pt": "Portuguese",
        "nl": "Dutch",
        "en": "English",
    }

    lang_name = language_names.get(language, language)

    # Get training metrics if available
    epoch = checkpoint_info.get("epoch", "N/A")
    train_metrics = checkpoint_info.get("train_metrics", {})
    val_metrics = checkpoint_info.get("val_metrics", {})

    # Format metrics for display
    val_loss = val_metrics.get("loss", "N/A")
    val_gap = val_metrics.get("similarity_gap", "N/A")

    readme_content = f"""---
language:
- {language}
license: apache-2.0
tags:
- audio
- speech
- multimodal
- synthetic-speech
- quality-assessment
- asr
- word-alignment
library_name: transformers
pipeline_tag: audio-classification
---

# WAVe: Word-Aligned Verification for {lang_name}

This model was trained to assess the quality of synthetic speech at the word level for {lang_name}.

## Model Description

WAVe (Word-Aligned Verification) is a multimodal embedding model that evaluates synthetic speech quality by:

1. **Computing text-audio similarity** at the sentence level using contrastive learning
2. **Aligning words with audio frames** using multi-head attention mechanisms
3. **Scoring alignment quality** with multi-head Gated Linear Units (GLU)

The model produces:
- **Quality scores** in range [0, 1] for each audio-text pair
- **Per-word alignment scores** indicating local synthesis quality
- **Full alignment matrices** showing word-to-frame correspondences

## Intended Use

This model is designed for **quality filtering of synthetic speech data** before ASR training. It can:

- Filter synthetic audio generated by TTS systems
- Detect synthesis errors (mispronunciations, omissions, unnatural prosody)
- Assess audio-text correspondence at word level

## Training Details

- **Trained on**: Common Voice {lang_name}
- **Epoch**: {epoch}
- **Validation Loss**: {val_loss}
- **Validation Similarity Gap**: {val_gap}

### Model Architecture

- **Text Encoder**: RoBERTa-large (sentence-transformers/all-roberta-large-v1)
- **Audio Encoder**: Wav2Vec2-BERT 2.0 (facebook/w2v-bert-2.0)
- **Projection Dim**: {config.projection_dim}
- **Alignment Heads**: {config.num_alignment_heads}
- **GLU Heads**: {config.n_glu_heads}

## Usage

```python
from transformers import AutoProcessor, AutoModel
import torch

# Load model and processor
processor = AutoProcessor.from_pretrained("{Path(output_dir).name}")
model = AutoModel.from_pretrained("{Path(output_dir).name}")

# Prepare inputs
text = "Your text here"
audio = torch.randn(16000)  # Your audio array (16kHz)

inputs = processor(
    text=text,
    audio=audio,
    sampling_rate=16000,
    return_tensors="pt"
)

# Get quality score
with torch.no_grad():
    outputs = model(**inputs)

quality_score = outputs.quality_score.item()
print(f"Quality: {{quality_score:.3f}}")

# Interpret score
if quality_score >= 0.8:
    print("✓ High quality - safe for ASR training")
elif quality_score >= 0.5:
    print("⚠ Medium quality - use with caution")
else:
    print("✗ Low quality - discard")
```

### Batch Filtering

```python
# Filter a dataset of synthetic samples
scores, high_quality_mask = model.verify_quality(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    input_values=batch["audio"],
    threshold=0.8
)

# Get indices of high-quality samples
good_indices = high_quality_mask.nonzero(as_tuple=True)[0]
```

### Access Detailed Outputs

```python
outputs = model(**inputs)

# All available outputs:
quality = outputs.quality_score              # Overall quality [0, 1]
cosine_sim = outputs.cosine_similarity       # Text-audio similarity [0, 1]
mean_align = outputs.mean_alignment_score    # Average alignment score
text_emb = outputs.text_embeds               # Text embeddings (768-dim)
audio_emb = outputs.audio_embeds             # Audio embeddings (768-dim)
word_scores = outputs.alignment_scores       # Per-word scores (seq_len,)
align_matrix = outputs.alignment_matrix      # Full matrix (text_len, audio_len)
```

## Evaluation Results

According to the paper, WAVe achieves:

- **34% reduction** in ASR training steps while maintaining performance
- **50% improvement** in cross-domain generalization (MLS benchmark)
- **Better filtering** with 30% less synthetic data vs. sentence-level methods

## Citation

```bibtex
@article{{wave2025,
  title={{WAVe: Word-Aligned Verification of Synthetic Speech for ASR}},
  author={{Perezhohin, Yuriy and Castelli, Mauro}},
  journal={{Preprint}},
  year={{2025}}
}}
```

## License

Apache 2.0

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
"""

    output_path = Path(output_dir) / "README.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    logger.info(f"Model card created at {output_path}")


def convert_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    model_type: str = "quality",
    language: str = "pt",
    push_to_hub: bool = False,
    hub_model_id: str = None,
):
    """
    Convert native PyTorch checkpoint to HuggingFace format.

    Args:
        checkpoint_path: Path to native .pt checkpoint
        output_dir: Directory to save converted model
        model_type: "base" or "quality" model variant
        language: Language code ("pt", "nl", etc.)
        push_to_hub: Whether to push to HuggingFace Hub
        hub_model_id: Model ID for Hub (e.g., "yuriyvnv/wave-portuguese")
    """
    logger.info("=" * 80)
    logger.info("WAVe Checkpoint Conversion")
    logger.info("=" * 80)

    # Load checkpoint
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract config
    config = extract_config_from_checkpoint(checkpoint)

    # Create model
    logger.info(f"Creating {model_type} model...")
    if model_type == "quality":
        model = WAVeForQualityVerification(config)
    else:
        model = WAVeModel(config)

    # Convert state dict
    converted_state_dict = convert_state_dict(
        checkpoint["model_state_dict"],
        model_type=model_type
    )

    # Load weights
    logger.info("Loading weights into model...")
    missing_keys, unexpected_keys = model.load_state_dict(
        converted_state_dict,
        strict=False
    )

    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys[:5]}...")

    logger.info("✓ Weights loaded successfully")

    # Create processor
    logger.info("Creating processor...")
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name_or_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.audio_model_name_or_path)
    processor = WAVeProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

    # Save to disk
    logger.info(f"Saving to: {output_dir}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    # Copy processing_wave.py to output directory for custom processor class
    import shutil
    processing_file = Path(__file__).parent / "processing_wave.py"
    if processing_file.exists():
        shutil.copy(processing_file, output_path / "processing_wave.py")

    # Create model card
    create_model_card(
        output_dir=output_dir,
        checkpoint_info=checkpoint,
        config=config,
        language=language
    )

    logger.info("✓ Conversion complete!")
    logger.info(f"  Model saved to: {output_path}")
    logger.info(f"  Files created:")
    logger.info(f"    - config.json")
    logger.info(f"    - pytorch_model.bin")
    logger.info(f"    - preprocessor_config.json")
    logger.info(f"    - tokenizer files")
    logger.info(f"    - README.md")

    # Push to Hub
    if push_to_hub:
        if hub_model_id is None:
            raise ValueError("Must provide hub_model_id when push_to_hub=True")

        logger.info(f"Pushing to HuggingFace Hub: {hub_model_id}")
        model.push_to_hub(hub_model_id, use_auth_token=True)
        processor.push_to_hub(hub_model_id, use_auth_token=True)

        logger.info(f"✓ Pushed to Hub: https://huggingface.co/{hub_model_id}")

    logger.info("=" * 80)

    return model, processor, config


def main():
    parser = argparse.ArgumentParser(
        description="Convert WAVe checkpoint to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to .pt checkpoint file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for converted model"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="quality",
        choices=["base", "quality"],
        help="Model type: 'base' for WAVeModel, 'quality' for WAVeForQualityVerification (default: quality)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="pt",
        help="Language code (pt, nl, en, etc.)"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push converted model to HuggingFace Hub"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="Model ID for Hub (e.g., 'yuriyvnv/wave-portuguese')"
    )

    args = parser.parse_args()

    convert_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        model_type=args.model_type,
        language=args.language,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )


if __name__ == "__main__":
    main()
