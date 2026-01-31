# WAVe Examples

This directory contains practical examples for using WAVe models.

## Examples

### 1. `huggingface_hub_example.py` ⭐ **NEW**

Complete examples for using WAVe directly from HuggingFace Hub with real audio.

```bash
python huggingface_hub_example.py
```

**What it includes:**
- **Example 1**: Assess a single audio file with transcript
- **Example 2**: Batch process multiple audio files
- **Example 3**: Filter HuggingFace datasets (CommonVoice, etc.)
- **Example 4**: Work with generated/synthetic audio arrays (TTS output)

**Key features:**
- No local model files needed (loads from Hub directly)
- Handles real audio files (.wav, .mp3, .flac)
- Shows complete workflow from audio loading to quality assessment
- Includes quality interpretation and recommendations

### 2. `inference_example.py`

Basic inference on a single audio-text pair.

```bash
python inference_example.py
```

**What it does:**
- Loads a WAVe model from HuggingFace Hub
- Processes a sample audio-text pair
- Computes quality score and displays all outputs
- Shows how to interpret results

### 3. `batch_filtering.py`

Filter a dataset of synthetic audio samples.

```bash
python batch_filtering.py --dataset_path /path/to/dataset --output_path ./filtered
```

**What it does:**
- Loads synthetic dataset
- Computes quality scores for all samples
- Filters based on threshold
- Saves filtered dataset and statistics

### 3. `quality_analysis.py`

Analyze quality distribution of a synthetic dataset.

```bash
python quality_analysis.py --dataset_path /path/to/dataset
```

**What it does:**
- Computes quality scores for entire dataset
- Generates quality distribution plots
- Shows per-word alignment statistics
- Identifies common failure patterns

## Usage Patterns

### Single Sample Inference

```python
from transformers import AutoProcessor, AutoModel

# Load from HuggingFace Hub
processor = AutoProcessor.from_pretrained(
    "yuriyvnv/WAVe-1B-Multimodal-PT",
    trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "yuriyvnv/WAVe-1B-Multimodal-PT",
    trust_remote_code=True
)

# Process audio and text
inputs = processor(text="Olá, como você está?", audio=audio_array, sampling_rate=16000, return_tensors="pt")
outputs = model(**inputs)

# Get quality score
quality = outputs.quality_score.item()
```

### Batch Processing

```python
from torch.utils.data import DataLoader

# Create dataloader
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# Process in batches
for batch in dataloader:
    inputs = processor(
        text=batch["text"],
        audio=batch["audio"],
        sampling_rate=16000,
        padding=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    quality_scores = outputs.quality_score
    # Filter or save based on quality_scores
```

### Filtering Pipeline

```python
def filter_synthetic_dataset(dataset, model, processor, threshold=0.8):
    \"\"\"Filter dataset by quality.\"\"\"

    good_samples = []

    for i, sample in enumerate(dataset):
        inputs = processor(
            text=sample["text"],
            audio=sample["audio"],
            sampling_rate=16000,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)

        if outputs.quality_score.item() >= threshold:
            good_samples.append(sample)

    return good_samples
```

## Tips

1. **Batch Processing**: Always use batching for better performance
2. **GPU Usage**: Move model to GPU with `model.to("cuda")`
3. **Mixed Precision**: Use `torch.amp` for faster inference
4. **Thresholds**:
   - 0.8+ for high quality (safe for training)
   - 0.5-0.8 for medium quality (inspect manually)
   - <0.5 for low quality (discard)
