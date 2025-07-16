# create_mixed_synthetic_cv_dataset.py
"""
Create a mixed dataset combining:
- Training: Synthetic Portuguese audio (yuriyvnv/synthetic_transcript_pt)
- Validation: Common Voice 17 Portuguese validation split
- Test: Common Voice 17 Portuguese test split

This allows training on synthetic data while evaluating on real speech data.
"""

import os
from datasets import load_dataset, DatasetDict, Audio, concatenate_datasets
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ───────────────────────── CONFIG ─────────────────────────
SYNTHETIC_DATASET = "yuriyvnv/synthetic_transcript_pt"
COMMON_VOICE_DATASET = "mozilla-foundation/common_voice_17_0"
LANGUAGE = "pt"

# New dataset configuration
NEW_DATASET_NAME = "yuriyvnv/synthetic_transcript_pt"
NEW_DATASET_DESCRIPTION = "Mixed Portuguese dataset: Synthetic training + Common Voice validation/test"

def standardize_columns(dataset, dataset_type="synthetic"):
    """Standardize column names across datasets"""
    
    if dataset_type == "synthetic":
        # Synthetic dataset columns: text, audio, voice, model, etc.
        # Keep all columns, just ensure we have the right text column
        if "text" not in dataset.column_names:
            raise ValueError("Synthetic dataset missing 'text' column")
        return dataset
    
    elif dataset_type == "common_voice":
        # Common Voice columns: sentence, audio, etc.
        # Rename 'sentence' to 'text' and keep essential columns
        def rename_columns(batch):
            # Rename sentence to text for consistency
            batch["text"] = batch["sentence"]
            return batch
        
        # Apply renaming and keep only essential columns
        dataset = dataset.map(rename_columns, remove_columns=["sentence"])
        
        # Keep only the columns we need: text, audio, and maybe some metadata
        columns_to_keep = ["text", "audio"]
        
        # Keep additional useful columns if they exist
        optional_columns = ["age", "gender", "accent", "locale", "client_id"]
        for col in optional_columns:
            if col in dataset.column_names:
                columns_to_keep.append(col)
        
        # Remove unwanted columns
        columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)
            
        return dataset
    
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

def create_mixed_dataset():
    """Create the mixed dataset with synthetic training + CV validation/test"""
    
    logger.info("🚀 Creating mixed synthetic-CommonVoice dataset...")
    
    # Load synthetic dataset
    logger.info("📊 Loading synthetic dataset for training...")
    synthetic_dataset = load_dataset(SYNTHETIC_DATASET, token=HF_TOKEN)
    synthetic_train = synthetic_dataset["train"]
    logger.info(f"✅ Loaded {len(synthetic_train):,} synthetic training samples")
    
    # Load Common Voice dataset  
    logger.info("📊 Loading Common Voice dataset for validation/test...")
    cv_dataset = load_dataset(COMMON_VOICE_DATASET, LANGUAGE, token=HF_TOKEN)
    cv_validation = cv_dataset["validation"]
    cv_test = cv_dataset["test"]
    logger.info(f"✅ Loaded {len(cv_validation):,} CV validation + {len(cv_test):,} CV test samples")
    
    # Ensure audio is at 16kHz for all datasets
    logger.info("🔄 Standardizing audio to 16kHz...")
    synthetic_train = synthetic_train.cast_column("audio", Audio(sampling_rate=16000))
    cv_validation = cv_validation.cast_column("audio", Audio(sampling_rate=16000))
    cv_test = cv_test.cast_column("audio", Audio(sampling_rate=16000))
    
    # Standardize column names
    logger.info("🔄 Standardizing column names...")
    synthetic_train = standardize_columns(synthetic_train, "synthetic")
    cv_validation = standardize_columns(cv_validation, "common_voice")
    cv_test = standardize_columns(cv_test, "common_voice")
    
    # Add dataset source labels
    logger.info("🏷️ Adding dataset source labels...")
    
    def add_source_label(batch, source):
        batch["dataset_source"] = [source] * len(batch["text"])
        return batch
    
    synthetic_train = synthetic_train.map(
        lambda batch: add_source_label(batch, "synthetic"),
        batched=True
    )
    cv_validation = cv_validation.map(
        lambda batch: add_source_label(batch, "common_voice"),
        batched=True
    )
    cv_test = cv_test.map(
        lambda batch: add_source_label(batch, "common_voice"),
        batched=True
    )
    
    # Create the mixed dataset
    logger.info("🔧 Creating mixed dataset structure...")
    mixed_dataset = DatasetDict({
        "train": synthetic_train,
        "validation": cv_validation,
        "test": cv_test
    })
    
    logger.info("📊 Mixed dataset summary:")
    for split, dataset in mixed_dataset.items():
        source_counts = {}
        for item in dataset:
            source = item["dataset_source"]
            source_counts[source] = source_counts.get(source, 0) + 1
        
        logger.info(f"   {split}: {len(dataset):,} samples")
        for source, count in source_counts.items():
            logger.info(f"     - {source}: {count:,} samples")
    
    return mixed_dataset

def upload_to_hub(mixed_dataset):
    """Upload the mixed dataset to Hugging Face Hub"""
    
    logger.info(f"🚀 Uploading mixed dataset to: {NEW_DATASET_NAME}")
    
    try:
        mixed_dataset.push_to_hub(
            NEW_DATASET_NAME,
            commit_message="Create mixed dataset: synthetic training + Common Voice validation/test",
            private=False,
            token=HF_TOKEN
        )
        logger.info(f"✅ Successfully uploaded to: https://huggingface.co/datasets/{NEW_DATASET_NAME}")
        
    except Exception as e:
        logger.error(f"❌ Failed to upload dataset: {e}")
        logger.info("💾 Dataset is still available locally for manual upload")
        raise

def create_dataset_card():
    """Create a dataset card for the mixed dataset"""
    
    dataset_card = f"""---
license: apache-2.0
task_categories:
- automatic-speech-recognition
- text-to-speech
language:
- pt
tags:
- synthetic
- common-voice
- portuguese
- mixed-dataset
- speech
size_categories:
- 10K<n<100K
---

# 🇧🇷 Mixed Synthetic-CommonVoice Portuguese Dataset

A **mixed Portuguese speech dataset** combining synthetic training data with real evaluation data for robust ASR model development.

## 📊 Dataset Composition

### Training Split (Synthetic)
- **Source**: OpenAI TTS-generated audio from `{SYNTHETIC_DATASET}`
- **Samples**: ~22,000 synthetic Portuguese sentences
- **Voices**: 9 different OpenAI TTS voices
- **Quality**: High-fidelity synthetic speech (24kHz MP3)

### Validation & Test Splits (Real Speech)
- **Source**: Mozilla Common Voice 17 Portuguese
- **Validation**: Real human speech for model validation
- **Test**: Real human speech for final evaluation
- **Quality**: Diverse speakers, accents, and recording conditions

## 🎯 Use Case

This dataset is designed for **domain adaptation** scenarios where you want to:

1. **Train efficiently** on high-quality synthetic data
2. **Evaluate realistically** on actual human speech
3. **Test generalization** from synthetic to real speech
4. **Benchmark** synthetic vs. real data performance

## 📋 Dataset Structure

```python
# All splits have consistent structure:
{{
    "text": "O tempo está muito bom hoje.",
    "audio": <audio_array>,
    "dataset_source": "synthetic" | "common_voice"
    # Additional columns vary by source
}}
```

## 🚀 Quick Start

```python
from datasets import load_dataset

# Load the mixed dataset
dataset = load_dataset("{NEW_DATASET_NAME}")

# Train on synthetic data
train_data = dataset["train"]  # All synthetic samples

# Evaluate on real speech
val_data = dataset["validation"]   # Real CV validation
test_data = dataset["test"]        # Real CV test

# Check data sources
print("Training sources:", set(train_data["dataset_source"]))
print("Validation sources:", set(val_data["dataset_source"]))
```

## 🔬 Research Applications

- **Domain Adaptation**: Study synthetic→real speech transfer
- **Data Efficiency**: Compare synthetic vs. real training data
- **Robustness Testing**: Evaluate across different speech types
- **Zero-shot Evaluation**: Test models trained only on synthetic data

## 🤝 Acknowledgments

- **Synthetic Data**: Generated using OpenAI GPT-4o-mini + TTS-1
- **Real Data**: Mozilla Common Voice 17 Portuguese contributors
- **Inspiration**: Common Voice word distribution patterns

---

**Perfect for researchers studying synthetic speech data and domain adaptation!** 🎉
"""
    
    return dataset_card

def save_dataset_info(mixed_dataset):
    """Save dataset information for reference"""
    
    dataset_info = {
        "dataset_name": NEW_DATASET_NAME,
        "creation_date": "2025-07-13",
        "splits": {},
        "total_samples": 0
    }
    
    for split_name, dataset in mixed_dataset.items():
        split_info = {
            "samples": len(dataset),
            "columns": dataset.column_names,
            "sources": {}
        }
        
        # Count sources
        for item in dataset:
            source = item["dataset_source"]
            split_info["sources"][source] = split_info["sources"].get(source, 0) + 1
        
        dataset_info["splits"][split_name] = split_info
        dataset_info["total_samples"] += len(dataset)
    
    # Save to file
    import json
    with open("mixed_dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    logger.info("📋 Dataset info saved to: mixed_dataset_info.json")
    return dataset_info

def main():
    """Main function to create and upload the mixed dataset"""
    
    logger.info("🚀 Starting mixed dataset creation process...")
    
    try:
        # Create the mixed dataset
        mixed_dataset = create_mixed_dataset()
        
        # Save dataset information
        dataset_info = save_dataset_info(mixed_dataset)
        
        # Create dataset card
        #dataset_card = create_dataset_card()
        logger.info("📝 Dataset card created")
        
        # Confirm upload
        print("\\n" + "="*60)
        print("📊 DATASET SUMMARY:")
        print(f"   Train: {{dataset_info['splits']['train']['samples']:,}} samples (synthetic)")
        print(f"   Validation: {{dataset_info['splits']['validation']['samples']:,}} samples (real)")
        print(f"   Test: {{dataset_info['splits']['test']['samples']:,}} samples (real)")
        print(f"   Total: {{dataset_info['total_samples']:,}} samples")
        print("="*60)
        
        confirm = input(f"\\n🚀 Upload to '{NEW_DATASET_NAME}'? (y/N): ")
        if confirm.lower() == 'y':
            upload_to_hub(mixed_dataset)
            
            # Save dataset card to file
            #with open("README.md", "w", encoding="utf-8") as f:
            #    f.write(dataset_card)
            #logger.info("📝 Dataset card saved to README.md")
            
            logger.info("\\n✅ Mixed dataset creation completed successfully!")
            logger.info(f"🔗 Dataset URL: https://huggingface.co/datasets/{NEW_DATASET_NAME}")
            
        else:
            logger.info("❌ Upload cancelled by user")
            logger.info("💾 Mixed dataset available locally for manual processing")
            
    except Exception as e:
        logger.error(f"🚨 Error creating mixed dataset: {e}")
        raise

if __name__ == "__main__":
    main()