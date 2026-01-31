# WAVe HuggingFace Implementation

Official HuggingFace Transformers integration for **WAVe** (Word-Aligned Verification), a multimodal model for assessing synthetic speech quality at the word level.

## 🌐 Model on HuggingFace Hub

**Model**: [yuriyvnv/WAVe-1B-Multimodal-PT](https://huggingface.co/yuriyvnv/WAVe-1B-Multimodal-PT)

- 879M parameters (1B)
- Portuguese language
- Word-level speech quality assessment
- Multimodal: Audio (Wav2Vec2-BERT) + Text (XLM-RoBERTa)

## 🚀 Quick Start

```python
from transformers import AutoModel, AutoProcessor
import torch

# Load from HuggingFace Hub (no local files needed)
processor = AutoProcessor.from_pretrained(
    "yuriyvnv/WAVe-1B-Multimodal-PT",
    trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "yuriyvnv/WAVe-1B-Multimodal-PT",
    trust_remote_code=True
)

# Process audio and text
text = "Olá, como você está?"
audio = ...  # numpy array, 16kHz mono

inputs = processor(text=text, audio=audio, sampling_rate=16000, return_tensors="pt")

# Get quality score
with torch.no_grad():
    outputs = model(**inputs)

quality = outputs.quality_score.item()  # 0.0 to 1.0

# Interpret: ≥0.8 = High quality, 0.5-0.8 = Medium, <0.5 = Low
```

## 📁 Directory Structure

```
wave_hf/
├── README.md                   # This file
├── STRUCTURE_EXPLAINED.md      # Explains modules/ vs modules.py
│
├── Core Implementation
│   ├── configuration_wave.py  # WAVeConfig
│   ├── modeling_wave.py        # WAVe model
│   ├── processing_wave.py      # WAVeProcessor
│   └── modules/                # Architecture components
│       ├── alignment.py        # Word-level alignment ⭐
│       ├── attention.py        # Cross-modal attention
│       ├── pooling.py          # Attentive pooling
│       └── projection.py       # Enhanced projections
│
├── scripts/                    # Utilities
│   ├── convert_checkpoint.py  # Convert .pt to HuggingFace format
│   └── upload_to_hub.py        # Upload to HuggingFace Hub
│
├── examples/                   # Usage examples
│   ├── huggingface_hub_example.py  # Complete Hub usage guide ⭐
│   ├── inference_example.py        # Basic inference
│   └── batch_filtering.py          # Dataset filtering
│
├── tests/                      # Testing utilities
└── docs/                       # Documentation
```

**Note**: The `wave-portuguese/` directory contains the converted model but is in `.gitignore` (large files). The model is available on HuggingFace Hub instead.

## 📖 Examples

### Example 1: Single Audio Assessment

```python
# See: examples/huggingface_hub_example.py
import torchaudio

# Load audio file
audio, sr = torchaudio.load("speech.wav")
audio = audio.mean(dim=0).numpy()  # Convert to mono

# Process
inputs = processor(text="Transcript here", audio=audio, sampling_rate=sr, return_tensors="pt")
outputs = model(**inputs)

print(f"Quality: {outputs.quality_score.item():.3f}")
print(f"Cosine Similarity: {outputs.cosine_similarity.item():.3f}")
print(f"Mean Alignment: {outputs.mean_alignment_score.item():.3f}")
```

### Example 2: Batch Dataset Filtering

```python
from datasets import load_dataset

dataset = load_dataset("mozilla-foundation/common_voice_16_1", "pt", split="test[:100]")

def quality_filter(example):
    inputs = processor(
        text=example["sentence"],
        audio=example["audio"]["array"],
        sampling_rate=16000,
        return_tensors="pt"
    )
    with torch.no_grad():
        quality = model(**inputs).quality_score.item()
    return quality >= 0.8

filtered = dataset.filter(quality_filter)
print(f"Kept {len(filtered)}/{len(dataset)} high-quality samples")
```

### Example 3: TTS Quality Assessment

```python
# Assess synthetic audio from your TTS system
tts_audio = your_tts_model.synthesize("Text to speak")  # numpy array

inputs = processor(text="Text to speak", audio=tts_audio, sampling_rate=16000, return_tensors="pt")
quality = model(**inputs).quality_score.item()

if quality >= 0.8:
    print("✅ High quality TTS output")
else:
    print("❌ Low quality, regenerate")
```

## 🔧 Converting Your Own Checkpoints

If you trained WAVe using the original training code:

```bash
# Convert checkpoint to HuggingFace format
python scripts/convert_checkpoint.py \
    --checkpoint_path path/to/your/checkpoint.pt \
    --output_dir ./wave-converted \
    --language pt

# Upload to HuggingFace Hub
python scripts/upload_to_hub.py
```

This creates a directory with:
- `config.json` - Model configuration
- `model.safetensors` - Model weights (~3.3GB)
- `README.md` - Model card
- Tokenizer and processor files

## 📊 Model Outputs

```python
outputs = model(**inputs)

# Available attributes:
outputs.quality_score           # Overall quality [0, 1] - PRIMARY METRIC
outputs.cosine_similarity       # Sentence-level similarity [-1, 1]
outputs.mean_alignment_score    # Average word alignment [0, 1]
outputs.text_embeds            # Text embeddings (batch, 768)
outputs.audio_embeds           # Audio embeddings (batch, 768)
outputs.alignment_scores       # Per-word scores (batch, seq_len)
outputs.alignment_matrix       # Attention matrix (batch, tokens, frames)
```

## 🎯 Quality Thresholds

Based on experimental results:

| Score | Quality | Recommendation |
|-------|---------|----------------|
| 0.8 - 1.0 | **High** | ✅ Safe for ASR training |
| 0.5 - 0.8 | **Medium** | ⚠️ Review manually or use with caution |
| 0.0 - 0.5 | **Low** | ❌ Discard from training set |

## 🏗️ Architecture

```
WAVe Model (879M parameters)
├── Text Encoder: XLM-RoBERTa (278M)
├── Audio Encoder: Wav2Vec2-BERT 2.0 (581M)
├── Text Projection: Enhanced MLP (2.4M)
├── Audio Projection: Enhanced MLP (2.8M)
└── Word-Level Alignment: Multi-head attention + GLU (14M) ⭐
    └── Core Innovation: Word-by-word quality scoring
```

**Key Innovation**: The word-level alignment module compares word representations before and after audio alignment, enabling precise detection of synthesis errors.

## 📝 Files Explanation

### Why `modules/` directory AND `modules.py` file?

See [STRUCTURE_EXPLAINED.md](STRUCTURE_EXPLAINED.md) for detailed explanation.

**TL;DR:**
- `modules/` = Development (edit these files)
- `modules.py` = HuggingFace Hub distribution (auto-generated)
- `modeling_wave.py` = Smart imports (works with both)

## 🔗 Links

- **HuggingFace Model**: https://huggingface.co/yuriyvnv/WAVe-1B-Multimodal-PT
- **Paper**: [Coming soon]
- **Training Code**: See parent repository

## 📄 Citation

```bibtex
@article{perezhohin2024wave,
  title={WAVe: Word-Aligned Verification of Synthetic Speech for ASR},
  author={Perezhohin, Yuriy and Castelli, Mauro},
  journal={arXiv preprint},
  year={2024}
}
```

## 📜 License

Apache 2.0

---

**Questions?** Check [examples/huggingface_hub_example.py](examples/huggingface_hub_example.py) for complete usage examples or the [STRUCTURE_EXPLAINED.md](STRUCTURE_EXPLAINED.md) for architecture details.
