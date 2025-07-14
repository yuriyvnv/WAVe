#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, Audio, concatenate_datasets
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class DatasetSubsetCreator:
    def __init__(self):
        self.target_sr = 16000  # Standardize on 16kHz
        
    def load_filtered_synthetic_data(self):
        """Load high and medium quality synthetic samples"""
        print("📊 Loading filtered synthetic data...")
        
        filtered_dir = Path("filtered_datasets")
        high_quality_path = filtered_dir / "high_quality_samples.json"
        medium_quality_path = filtered_dir / "medium_quality_samples.json"
        
        # Load high quality samples
        with open(high_quality_path, 'r') as f:
            high_quality = json.load(f)
        print(f"  High quality: {len(high_quality)} samples")
        
        # Load medium quality samples  
        with open(medium_quality_path, 'r') as f:
            medium_quality = json.load(f)
        print(f"  Medium quality: {len(medium_quality)} samples")
        
        # Combine and sort by sample_index
        combined = high_quality + medium_quality
        combined.sort(key=lambda x: x['sample_index'])
        
        print(f"  📊 Total filtered: {len(combined)} samples")
        return combined
    
    def load_current_dataset(self):
        """Load current HuggingFace dataset"""
        print("📥 Loading current HuggingFace dataset...")
        dataset = load_dataset("yuriyvnv/synthetic_transcript_pt")
        
        print(f"  Train: {len(dataset['train'])} samples")
        print(f"  Validation: {len(dataset['validation'])} samples") 
        print(f"  Test: {len(dataset['test'])} samples")
        
        return dataset
    
    def load_common_voice_data(self):
        """Load Common Voice 17 Portuguese data and standardize format"""
        print("📥 Loading Common Voice 17 Portuguese...")
        cv_dataset = load_dataset("mozilla-foundation/common_voice_17_0", "pt")
        
        print(f"  Original train: {len(cv_dataset['train'])} samples (48kHz)")
        print(f"  Original validation: {len(cv_dataset['validation'])} samples (48kHz)")
        print(f"  Original test: {len(cv_dataset['test'])} samples (48kHz)")
        
        print("🔧 Standardizing Common Voice format...")
        
        # 🚀 FAST: Cast audio column to 16kHz (automatic resampling!)
        print("  Resampling audio 48kHz → 16kHz...")
        cv_dataset = cv_dataset.cast_column("audio", Audio(sampling_rate=self.target_sr))
        
        # 🚀 FAST: Remove extra CV-only columns first
        print("  Removing CV-specific columns...")
        columns_to_remove = ['segment', 'variant', 'path']
        existing_columns = cv_dataset['train'].column_names
        columns_to_remove = [col for col in columns_to_remove if col in existing_columns]
        
        if columns_to_remove:
            cv_dataset = cv_dataset.remove_columns(columns_to_remove)
            print(f"    Removed: {columns_to_remove}")
        
        # 🚀 FAST: Rename sentence → text
        print("  Renaming 'sentence' → 'text'...")
        cv_dataset = cv_dataset.rename_column("sentence", "text")
        
        # 🚀 FAST: Add new columns with constant values (much faster than map!)
        print("  Adding synthetic-specific fields...")
        
        def add_constant_fields(split_dataset):
            """Add constant fields efficiently"""
            # Add new columns with constant values - much faster!
            split_dataset = split_dataset.add_column('voice', ['common_voice'] * len(split_dataset))
            split_dataset = split_dataset.add_column('model', ['common_voice_17'] * len(split_dataset))
            split_dataset = split_dataset.add_column('dataset_source', ['common_voice'] * len(split_dataset))
            return split_dataset
        
        # Apply to each split
        cv_dataset['train'] = add_constant_fields(cv_dataset['train'])
        cv_dataset['validation'] = add_constant_fields(cv_dataset['validation'])
        cv_dataset['test'] = add_constant_fields(cv_dataset['test'])
        
        # 🚀 FAST: Handle missing fields with default values (only if needed)
        print("  Ensuring all required fields exist...")
        
        def ensure_field_exists(split_dataset, field_name, default_value):
            """Add field if it doesn't exist"""
            if field_name not in split_dataset.column_names:
                split_dataset = split_dataset.add_column(field_name, [default_value] * len(split_dataset))
            return split_dataset
        
        # Check and add missing fields for each split
        for split_name in ['train', 'validation', 'test']:
            split_data = cv_dataset[split_name]
            split_data = ensure_field_exists(split_data, 'age', '')
            split_data = ensure_field_exists(split_data, 'gender', '')
            split_data = ensure_field_exists(split_data, 'accent', '')
            split_data = ensure_field_exists(split_data, 'locale', 'pt')
            split_data = ensure_field_exists(split_data, 'client_id', '')
            split_data = ensure_field_exists(split_data, 'up_votes', 0)
            split_data = ensure_field_exists(split_data, 'down_votes', 0)
            cv_dataset[split_name] = split_data
        
        print(f"✅ Standardized Common Voice (FAST METHOD):")
        print(f"  Train: {len(cv_dataset['train'])} samples (16kHz)")
        print(f"  Validation: {len(cv_dataset['validation'])} samples (16kHz)")
        print(f"  Test: {len(cv_dataset['test'])} samples (16kHz)")
        print(f"  Schema: {list(cv_dataset['train'].features.keys())}")
        
        return cv_dataset
    
    def create_filtered_synthetic_dataset(self, current_train, filtered_indices):
        """Create filtered synthetic dataset from indices"""
        print("🔧 Creating filtered synthetic dataset...")
        
        # Extract indices
        indices = [item['sample_index'] for item in filtered_indices]
        print(f"  Filtering {len(indices)} samples from {len(current_train)} total")
        
        # Use select() to filter by indices - MUCH faster!
        filtered_dataset = current_train.select(indices)
        
        print(f"  ✅ Filtered synthetic dataset: {len(filtered_dataset)} samples")
        return filtered_dataset
    
    def create_subset_1_fully_synthetic(self, current_dataset):
        """Create subset 1: fully_synthetic (reuse existing)"""
        print("\n🎯 Creating Subset 1: fully_synthetic")
        print("-" * 50)
        
        # Current dataset is already perfect for this subset!
        subset_dict = DatasetDict({
            'train': current_dataset['train'],
            'validation': current_dataset['validation'], 
            'test': current_dataset['test']
        })
        
        print(f"  Train: {len(subset_dict['train'])} (all synthetic)")
        print(f"  Validation: {len(subset_dict['validation'])} (CV17)")
        print(f"  Test: {len(subset_dict['test'])} (CV17)")
        print(f"  Total: {sum(len(split) for split in subset_dict.values())} samples")
        
        return subset_dict
    
    def create_subset_2_mixed(self, filtered_synthetic, cv_dataset, current_validation, current_test):
        """Create subset 2: mixed_cv_synthetic"""
        print("\n🎯 Creating Subset 2: mixed_cv_synthetic")
        print("-" * 50)
        
        print(f"  Combining datasets...")
        print(f"    Filtered synthetic: {len(filtered_synthetic)} samples")
        print(f"    CV17 train: {len(cv_dataset['train'])} samples")
        
        # 🚀 FAST: Use concatenate_datasets to combine
        mixed_train = concatenate_datasets([filtered_synthetic, cv_dataset['train']])
        
        print(f"  ✅ Combined train: {len(mixed_train)} samples")
        
        subset_dict = DatasetDict({
            'train': mixed_train,
            'validation': current_validation,  # Reuse existing (already CV17)
            'test': current_test  # Reuse existing (already CV17)
        })
        
        print(f"  Train: {len(subset_dict['train'])} (filtered synthetic + CV17)")
        print(f"  Validation: {len(subset_dict['validation'])} (CV17)")
        print(f"  Test: {len(subset_dict['test'])} (CV17)")
        print(f"  Total: {sum(len(split) for split in subset_dict.values())} samples")
        
        return subset_dict
    
    def create_subset_3_cv_only(self, cv_dataset):
        """Create subset 3: cv_only"""
        print("\n🎯 Creating Subset 3: cv_only") 
        print("-" * 50)
        
        # Just use Common Voice dataset directly
        subset_dict = DatasetDict({
            'train': cv_dataset['train'],
            'validation': cv_dataset['validation'],
            'test': cv_dataset['test']
        })
        
        print(f"  Train: {len(subset_dict['train'])} (CV17 only)")
        print(f"  Validation: {len(subset_dict['validation'])} (CV17)")
        print(f"  Test: {len(subset_dict['test'])} (CV17)")
        print(f"  Total: {sum(len(split) for split in subset_dict.values())} samples")
        
        return subset_dict
    
    def verify_subset_consistency(self, subsets):
        """Verify all subsets have consistent schemas"""
        print("\n🔍 Verifying subset consistency...")
        
        reference_features = None
        reference_name = None
        
        for subset_name, subset_data in subsets.items():
            train_features = subset_data['train'].features
            
            if reference_features is None:
                reference_features = train_features
                reference_name = subset_name
                print(f"  Using {subset_name} as reference schema")
                continue
            
            # Compare features
            ref_keys = set(reference_features.keys())
            subset_keys = set(train_features.keys())
            
            if ref_keys == subset_keys:
                print(f"  ✅ {subset_name}: Schema matches reference")
            else:
                print(f"  ⚠️ {subset_name}: Schema differences detected")
                missing = ref_keys - subset_keys
                extra = subset_keys - ref_keys
                if missing:
                    print(f"    Missing fields: {missing}")
                if extra:
                    print(f"    Extra fields: {extra}")
            
            # Check audio consistency
            sample = subset_data['train'][0]
            audio_sr = sample['audio']['sampling_rate']
            if audio_sr == self.target_sr:
                print(f"  ✅ {subset_name}: Audio sampling rate correct ({audio_sr}Hz)")
            else:
                print(f"  ❌ {subset_name}: Wrong sampling rate ({audio_sr}Hz)")
    
    def create_all_subsets(self):
        """Create all three subsets efficiently"""
        print("🚀 CREATING ALL DATASET SUBSETS (FAST VERSION)")
        print("=" * 60)
        
        # Load all required data
        print("📂 Loading all datasets...")
        current_dataset = self.load_current_dataset()
        cv_dataset = self.load_common_voice_data()
        filtered_indices = self.load_filtered_synthetic_data()
        
        # Create filtered synthetic dataset
        filtered_synthetic = self.create_filtered_synthetic_dataset(
            current_dataset['train'], 
            filtered_indices
        )
        
        print("\n🎯 Creating all subsets...")
        subsets = {}
        
        # Subset 1: fully_synthetic (instant - reuse existing)
        subsets['fully_synthetic'] = self.create_subset_1_fully_synthetic(current_dataset)
        
        # Subset 2: mixed_cv_synthetic (fast concatenation)
        subsets['mixed_cv_synthetic'] = self.create_subset_2_mixed(
            filtered_synthetic,
            cv_dataset,
            current_dataset['validation'],
            current_dataset['test']
        )
        
        # Subset 3: cv_only (instant - use CV dataset)
        subsets['cv_only'] = self.create_subset_3_cv_only(cv_dataset)
        
        # Verify consistency
        self.verify_subset_consistency(subsets)
        
        return subsets
    
    def save_subsets_locally(self, subsets, output_dir="dataset_subsets"):
        """Save subsets locally as HuggingFace datasets"""
        print(f"\n💾 Saving subsets locally to {output_dir}")
        print("-" * 50)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for subset_name, subset_data in subsets.items():
            subset_path = output_path / subset_name
            
            print(f"  Saving {subset_name}...")
            
            # Save as HuggingFace dataset format
            subset_data.save_to_disk(str(subset_path))
            
            # Also create summary JSON for quick inspection
            summary = {}
            for split_name, split_data in subset_data.items():
                summary[split_name] = {
                    'num_samples': len(split_data),
                    'features': list(split_data.features.keys()),
                    'audio_sampling_rate': split_data[0]['audio']['sampling_rate'],
                    'sample_text': split_data[0]['text'][:100] + "..."
                }
            
            summary_path = subset_path / "summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            print(f"    ✅ Saved to {subset_path}")
            print(f"       Train: {len(subset_data['train'])} samples")
            print(f"       Val: {len(subset_data['validation'])} samples") 
            print(f"       Test: {len(subset_data['test'])} samples")
        
        print(f"✅ All subsets saved to {output_path}")
        return output_path

def main():
    """Main execution function"""
    print("🎯 FAST DATASET SUBSET CREATOR")
    print("=" * 60)
    
    creator = DatasetSubsetCreator()
    
    try:
        # Create all subsets (should be much faster now!)
        subsets = creator.create_all_subsets()
        
        # Save locally for inspection
        output_path = creator.save_subsets_locally(subsets)
        
        # Print summary
        print(f"\n📊 SUBSET CREATION SUMMARY")
        print("=" * 60)
        
        total_all_subsets = 0
        for subset_name, subset_data in subsets.items():
            total_samples = sum(len(split) for split in subset_data.values())
            total_all_subsets += total_samples
            
            print(f"\n{subset_name}:")
            print(f"  Total: {total_samples:,} samples")
            for split_name, split_data in subset_data.items():
                print(f"    {split_name}: {len(split_data):,} samples")
        
        print(f"\n🎉 SUCCESS! All subsets created and saved to {output_path}")
        print(f"📊 Total across all subsets: {total_all_subsets:,} samples")
        print(f"⚡ Estimated processing time: ~5-10 minutes (much faster!)")
        
        print(f"\n📋 Next steps:")
        print(f"  1. Review the saved datasets in {output_path}")
        print(f"  2. Check summary.json files for quick overview")
        print(f"  3. Run upload script to push to HuggingFace")
        print(f"  4. Test loading each subset")
        
        return subsets, output_path
        
    except Exception as e:
        print(f"❌ Error creating subsets: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    subsets, output_path = main()