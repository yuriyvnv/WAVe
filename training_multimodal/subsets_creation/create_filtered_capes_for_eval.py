#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datasets import load_dataset, Dataset, DatasetDict, Audio, concatenate_datasets
from tqdm import tqdm
from huggingface_hub import login
import os
from collections import Counter

class IEEESubsetCreator:
    def __init__(self, hf_token=None):
        """
        Initialize the IEEE subset creator
        
        Args:
            hf_token: HuggingFace token for pushing dataset
        """
        self.hf_token = hf_token
        
        # Dataset names
        self.synthetic_dataset_name = "my-north-ai/capes_synthetic_audio_PT"
        self.reference_dataset_name = "my-north-ai/cv_mls_psfb_fs0_24"
        self.cv_dataset_name = "mozilla-foundation/common_voice_13_0"
        self.output_dataset_name = "yuriyvnv/capes_synthetic_audio_filtered"
        
    def load_reference_sentences(self):
        """Load unique sentences from reference dataset"""
        print(f"\n📥 Loading reference dataset: {self.reference_dataset_name}")
        
        try:
            # Load the reference dataset
            reference_dataset = load_dataset(
                self.reference_dataset_name, 
                split='train',
                trust_remote_code=True
            )
            
            print(f"Reference dataset size: {len(reference_dataset)}")
            print(f"Reference dataset columns: {reference_dataset.column_names}")
            
            # Extract sentences - try different possible column names
            sentence_column = None
            for col in ['sentence', 'text', 'transcription', 'transcript']:
                if col in reference_dataset.column_names:
                    sentence_column = col
                    break
            
            if sentence_column is None:
                raise ValueError(f"Could not find text column in reference dataset. Available columns: {reference_dataset.column_names}")
            
            print(f"Using column '{sentence_column}' for text matching")
            
            # Get all sentences and normalize them
            sentences = []
            for example in tqdm(reference_dataset, desc="Extracting sentences"):
                sentence = example[sentence_column]
                if sentence:
                    # Normalize: strip whitespace and convert to lowercase for matching
                    normalized = sentence.strip().lower()
                    sentences.append(normalized)
            
            # Get unique sentences
            unique_sentences = set(sentences)
            
            print(f"Total sentences extracted: {len(sentences)}")
            print(f"Unique sentences: {len(unique_sentences)}")
            
            # Show some examples
            print("\nExample sentences from reference dataset:")
            for i, sent in enumerate(list(unique_sentences)[:5]):
                print(f"  {i+1}. {sent[:100]}...")  # Truncate long sentences
            
            return unique_sentences, sentence_column
            
        except Exception as e:
            print(f"Error loading reference dataset: {e}")
            raise
    
    def filter_synthetic_by_text_matching(self, reference_sentences):
        """Load full synthetic dataset and filter by text matching"""
        print(f"\n📥 Loading FULL synthetic dataset: {self.synthetic_dataset_name}")
        synthetic_dataset = load_dataset(self.synthetic_dataset_name, split='train')
        
        print(f"Full synthetic dataset size: {len(synthetic_dataset)}")
        print(f"Columns: {synthetic_dataset.column_names}")
        
        # Find text column in synthetic dataset
        text_column = None
        for col in ['translation', 'sentence', 'text', 'transcription']:
            if col in synthetic_dataset.column_names:
                text_column = col
                break
        
        if text_column is None:
            raise ValueError(f"Could not find text column. Available: {synthetic_dataset.column_names}")
        
        print(f"Using column '{text_column}' from synthetic dataset for text matching")
        
        # Apply text matching filter to FULL dataset
        print("\n🔍 Filtering full dataset by text matching...")
        matched_indices = []
        excluded_indices = []
        matched_sentences = []
        
        for idx in tqdm(range(len(synthetic_dataset)), desc="Matching sentences"):
            sample = synthetic_dataset[idx]
            sentence = sample[text_column]
            
            if sentence:
                # Normalize for comparison
                normalized = sentence.strip().lower()
                
                if normalized in reference_sentences:
                    matched_indices.append(idx)
                    matched_sentences.append(sentence)
                else:
                    excluded_indices.append(idx)
            else:
                excluded_indices.append(idx)
        
        # Create filtered dataset with only matched samples
        filtered_dataset = synthetic_dataset.select(matched_indices)
        
        # Rename translation to sentence if needed
        if text_column == 'translation':
            filtered_dataset = filtered_dataset.rename_column('translation', 'sentence')
        
        # Keep only audio and sentence columns
        columns_to_keep = ['audio', 'sentence']
        filtered_dataset = filtered_dataset.select_columns(columns_to_keep)
        
        # Ensure audio is 16kHz
        filtered_dataset = filtered_dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        # Print statistics
        print("\n📊 TEXT MATCHING STATISTICS (FULL DATASET):")
        print("=" * 60)
        print(f"Total samples in full dataset: {len(synthetic_dataset):,}")
        print(f"Samples matching reference sentences: {len(matched_indices):,}")
        print(f"Samples excluded (no match): {len(excluded_indices):,}")
        print(f"Match rate: {len(matched_indices)/len(synthetic_dataset)*100:.2f}%")
        
        # Analyze matched sentences
        sentence_counts = Counter(matched_sentences)
        print(f"\nUnique matched sentences: {len(sentence_counts):,}")
        print("\nTop 10 most frequent matched sentences:")
        for sent, count in sentence_counts.most_common(10):
            print(f"  [{count:,}x] {sent[:80]}...")  # Truncate long sentences
        
        return filtered_dataset, len(matched_indices), len(excluded_indices)
    
    def load_common_voice_portuguese(self):
        """Load Common Voice Portuguese dataset"""
        print(f"\n📥 Loading Common Voice Portuguese dataset...")
        
        try:
            # Try loading with language filter
            cv_dataset = load_dataset(
                self.cv_dataset_name, 
                "pt",  # Portuguese
                split=None,  # Load all splits
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Error loading with language filter: {e}")
            # Alternative: load specific Portuguese config if available
            cv_dataset = load_dataset(
                "mozilla-foundation/common_voice_11_0",  # Try older version
                "pt",
                split=None,
                trust_remote_code=True
            )
        
        # Print dataset info
        for split in cv_dataset:
            print(f"Common Voice {split}: {len(cv_dataset[split])} samples")
            print(f"Columns: {cv_dataset[split].column_names}")
        
        # Normalize each split to only have audio and sentence columns
        normalized_cv = {}
        columns_to_keep = ['audio', 'sentence']
        
        for split in cv_dataset:
            # Keep only audio and sentence columns
            normalized_cv[split] = cv_dataset[split].select_columns(columns_to_keep)
            # Ensure audio is 16kHz
            normalized_cv[split] = normalized_cv[split].cast_column("audio", Audio(sampling_rate=16000))
            print(f"Normalized {split} columns: {normalized_cv[split].column_names}")
        
        return DatasetDict(normalized_cv)
    
    def create_ieee_subset_with_splits(self):
        """Create the IEEE subset with train/validation/test splits"""
        # Load reference sentences
        reference_sentences, _ = self.load_reference_sentences()
        
        # Filter full synthetic dataset by text matching
        filtered_synthetic, matched_count, excluded_count = self.filter_synthetic_by_text_matching(
            reference_sentences
        )
        
        # Add source identifier to synthetic data
        filtered_synthetic = filtered_synthetic.add_column(
            'source', ['synthetic_matched'] * len(filtered_synthetic)
        )
        
        # Load Common Voice dataset (already normalized)
        cv_dataset = self.load_common_voice_portuguese()
        
        # Add source identifier to CV data
        for split in cv_dataset:
            cv_dataset[split] = cv_dataset[split].add_column(
                'source', ['common_voice'] * len(cv_dataset[split])
            )
        
        # Create train split: filtered synthetic + CV train
        print("\n🔀 Creating IEEE train split...")
        ieee_train = concatenate_datasets([filtered_synthetic, cv_dataset['train']])
        ieee_train = ieee_train.shuffle(seed=42)
        
        # Validation and test are CV only
        ieee_validation = cv_dataset['validation']
        ieee_test = cv_dataset['test']
        
        # Create final IEEE subset with all splits
        ieee_subset = DatasetDict({
            'train': ieee_train,
            'validation': ieee_validation,
            'test': ieee_test
        })
        
        print("\n📊 IEEE SUBSET STATISTICS:")
        print("=" * 60)
        for split, data in ieee_subset.items():
            print(f"\n{split}: {len(data)} samples")
            if split == 'train':
                # Count sources
                source_values = data['source']
                synthetic_count = sum(1 for s in source_values if s == 'synthetic_matched')
                cv_count = sum(1 for s in source_values if s == 'common_voice')
                print(f"  - Synthetic (text-matched): {synthetic_count:,}")
                print(f"  - Common Voice: {cv_count:,}")
        
        return ieee_subset, matched_count, excluded_count
    
    def push_ieee_subset_to_hub(self, ieee_subset, matched_count, excluded_count):
        """Push the IEEE subset to HuggingFace Hub"""
        if self.hf_token:
            login(token=self.hf_token)
        
        print(f"\n🚀 Pushing IEEE subset to {self.output_dataset_name}...")
        
        # Calculate statistics for dataset card
        train_source_values = ieee_subset['train']['source']
        train_synthetic_count = sum(1 for s in train_source_values if s == 'synthetic_matched')
        train_cv_count = sum(1 for s in train_source_values if s == 'common_voice')
        
        # Option 1: Push as separate splits with ieee_ prefix
        # This will show as additional splits in the dataset viewer
        ieee_splits = DatasetDict({
            'ieee_train': ieee_subset['train'],
            'ieee_validation': ieee_subset['validation'], 
            'ieee_test': ieee_subset['test']
        })
        
        # Load existing dataset to merge
        try:
            print("Loading existing dataset...")
            existing_dataset = load_dataset(self.output_dataset_name)
            
            # Merge with existing splits
            final_dataset = DatasetDict({
                'train': existing_dataset['train'],
                'validation': existing_dataset['validation'],
                'test': existing_dataset['test'],
                'ieee_train': ieee_subset['train'],
                'ieee_validation': ieee_subset['validation'],
                'ieee_test': ieee_subset['test']
            })
            
            # If there's an existing original_filtered_ieee split, we can keep or remove it
            if 'original_filtered_ieee' in existing_dataset:
                print("Note: Removing old 'original_filtered_ieee' split, replacing with ieee_train/val/test")
            
        except Exception as e:
            print(f"Could not load existing dataset: {e}")
            print("Creating new dataset with IEEE splits only")
            final_dataset = ieee_splits
        
        # Create comprehensive dataset card
        dataset_card = f"""
# CAPES Synthetic Audio Filtered Dataset

This dataset contains filtered synthetic Portuguese audio samples with multiple configurations.

## Dataset Structure

### Main Configuration (High-quality filtered)
- `train`: High-quality synthetic (similarity > 0.8) + Common Voice train
- `validation`: Common Voice validation only
- `test`: Common Voice test only

### IEEE Configuration (Text-matched filtered)
- `ieee_train`: Synthetic (text-matched) + Common Voice train
- `ieee_validation`: Common Voice validation only
- `ieee_test`: Common Voice test only

## IEEE Configuration Details

The IEEE configuration contains synthetic samples filtered by exact text matching with the reference dataset `my-north-ai/cv_mls_psfb_fs0_24`.

### Filtering Statistics:
- Original synthetic dataset: 110,224 samples
- Samples with matching text: {matched_count:,}
- Samples excluded: {excluded_count:,}
- Match rate: {matched_count/110224*100:.2f}%

### IEEE Split Composition:
- **ieee_train**: {len(ieee_subset['train']):,} samples
  - Synthetic (text-matched): {train_synthetic_count:,} samples  
  - Common Voice: {train_cv_count:,} samples
- **ieee_validation**: {len(ieee_subset['validation']):,} samples (Common Voice only)
- **ieee_test**: {len(ieee_subset['test']):,} samples (Common Voice only)

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("yuriyvnv/capes_synthetic_audio_filtered")

# Access main configuration
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Access IEEE configuration
ieee_train = dataset['ieee_train']
ieee_val = dataset['ieee_validation']
ieee_test = dataset['ieee_test']
```

## Fields

All splits contain:
- `audio`: Audio data (16kHz sampling rate)
- `sentence`: Text transcription
- `source`: Data source identifier

## Source Datasets

- Synthetic: my-north-ai/capes_synthetic_audio_PT
- Common Voice: mozilla-foundation/common_voice_13_0 (Portuguese)
- IEEE Reference: my-north-ai/cv_mls_psfb_fs0_24
"""
        
        # Push the complete dataset with all splits
        final_dataset.push_to_hub(
            self.output_dataset_name,
            private=False,
            commit_message="Add IEEE configuration splits (ieee_train/validation/test)",
        )
        
        print(f"\n✅ IEEE subset successfully pushed!")
        print(f"   Dataset: {self.output_dataset_name}")
        print(f"   New splits: ieee_train, ieee_validation, ieee_test")
        print(f"\n📊 IEEE splits added:")
        print(f"   - ieee_train: {len(ieee_subset['train']):,} samples")
        print(f"   - ieee_validation: {len(ieee_subset['validation']):,} samples")  
        print(f"   - ieee_test: {len(ieee_subset['test']):,} samples")

def main():
    # Configuration
    HF_TOKEN = None  # Set your HuggingFace token or use HF_TOKEN env variable
    
    # Use environment variable if token not provided
    if HF_TOKEN is None:
        HF_TOKEN = os.getenv("HF_TOKEN")
    
    # Create subset creator
    creator = IEEESubsetCreator(hf_token=HF_TOKEN)
    
    # Create the IEEE subset with all splits
    ieee_subset, matched_count, excluded_count = creator.create_ieee_subset_with_splits()
    
    # Push to HuggingFace Hub
    creator.push_ieee_subset_to_hub(ieee_subset, matched_count, excluded_count)
    
    print("\n🎉 IEEE subset creation and upload complete!")
    print(f"\n📊 FINAL SUMMARY:")
    print(f"   Full synthetic dataset: 110,224 samples")
    print(f"   Synthetic samples matched: {matched_count:,}")
    print(f"   Synthetic samples excluded: {excluded_count:,}")
    print(f"   Match rate: {matched_count/110224*100:.2f}%")

if __name__ == "__main__":
    main()