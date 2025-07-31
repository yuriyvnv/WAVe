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
        
        filtered_dir = Path("filtered_datasets_nl")
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
        return combined, high_quality
    
    def load_current_dataset(self):
        """Load current HuggingFace dataset"""
        print("📥 Loading current HuggingFace dataset...")
        dataset = load_dataset("ANONYMOUS_USER/synthetic_transcript_nl")
        
        print(f"  Train: {len(dataset['train'])} samples")
        
        # Check if validation and test splits exist
        available_splits = list(dataset.keys())
        print(f"  Available splits: {available_splits}")
        
        if 'validation' in dataset:
            print(f"  Validation: {len(dataset['validation'])} samples")
        else:
            print("  Validation: Not available (will use CV17)")
            
        if 'test' in dataset:
            print(f"  Test: {len(dataset['test'])} samples")
        else:
            print("  Test: Not available (will use CV17)")
        
        return dataset
    
    def load_common_voice_data(self):
        """Load Common Voice 17 Portuguese data and standardize format"""
        print("📥 Loading Common Voice 17 Portuguese...")
        cv_dataset = load_dataset("mozilla-foundation/common_voice_17_0", "nl")
        
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
            split_data = ensure_field_exists(split_data, 'locale', 'nl')
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

    def create_high_quality_synthetic_dataset(self, current_train, high_quality_indices):
        """Create high quality only synthetic dataset from indices"""
        print("🔧 Creating high quality synthetic dataset...")
        
        # Extract indices
        indices = [item['sample_index'] for item in high_quality_indices]
        print(f"  Filtering {len(indices)} high quality samples from {len(current_train)} total")
        
        # Use select() to filter by indices
        high_quality_dataset = current_train.select(indices)
        
        print(f"  ✅ High quality synthetic dataset: {len(high_quality_dataset)} samples")
        return high_quality_dataset
    
    def create_subset_1_fully_synthetic(self, current_dataset, cv_dataset, unified_features, feature_defaults):
        """Create subset 1: fully_synthetic (train=synthetic, val/test=CV17)"""
        print("\n🎯 Creating Subset 1: fully_synthetic")
        print("-" * 50)
        
        # Use synthetic for train, CV17 for validation/test
        subset_dict = DatasetDict({
            'train': current_dataset['train'],
            'validation': cv_dataset['validation'], 
            'test': cv_dataset['test']
        })
        
        # Normalize features across all splits using unified schema
        subset_dict = self.normalize_dataset_features(subset_dict, unified_features, feature_defaults)
        
        print(f"  Train: {len(subset_dict['train'])} (all synthetic)")
        print(f"  Validation: {len(subset_dict['validation'])} (CV17)")
        print(f"  Test: {len(subset_dict['test'])} (CV17)")
        print(f"  Total: {sum(len(split) for split in subset_dict.values())} samples")
        
        return subset_dict
    
    def create_subset_2_mixed(self, filtered_synthetic, cv_dataset, unified_features, feature_defaults):
        """Create subset 2: mixed_cv_synthetic"""
        print("\n🎯 Creating Subset 2: mixed_cv_synthetic")
        print("-" * 50)
        
        print(f"  Combining datasets...")
        print(f"    Filtered synthetic: {len(filtered_synthetic)} samples")
        print(f"    CV17 train: {len(cv_dataset['train'])} samples")
        
        # Fix audio feature mismatch by standardizing both datasets
        print("  Standardizing audio features...")
        
        # Cast both to standard audio format (sampling_rate=None allows flexibility)
        from datasets import Audio
        standard_audio = Audio(sampling_rate=None, mono=True, decode=True)
        
        filtered_synthetic = filtered_synthetic.cast_column("audio", standard_audio)
        cv_train = cv_dataset['train'].cast_column("audio", standard_audio)
        
        # Now concatenate with aligned features
        mixed_train = concatenate_datasets([filtered_synthetic, cv_train])
        
        print(f"  ✅ Combined train: {len(mixed_train)} samples")
        
        subset_dict = DatasetDict({
            'train': mixed_train,
            'validation': cv_dataset['validation'],  # Use CV17
            'test': cv_dataset['test']  # Use CV17
        })
        
        # Normalize features across all splits using unified schema
        subset_dict = self.normalize_dataset_features(subset_dict, unified_features, feature_defaults)
        
        print(f"  Train: {len(subset_dict['train'])} (filtered synthetic + CV17)")
        print(f"  Validation: {len(subset_dict['validation'])} (CV17)")
        print(f"  Test: {len(subset_dict['test'])} (CV17)")
        print(f"  Total: {sum(len(split) for split in subset_dict.values())} samples")
        
        return subset_dict
    
    def create_subset_3_cv_only(self, cv_dataset, unified_features, feature_defaults):
        """Create subset 3: cv_only"""
        print("\n🎯 Creating Subset 3: cv_only") 
        print("-" * 50)
        
        # Just use Common Voice dataset directly
        subset_dict = DatasetDict({
            'train': cv_dataset['train'],
            'validation': cv_dataset['validation'],
            'test': cv_dataset['test']
        })
        
        # Normalize features across all splits using unified schema
        subset_dict = self.normalize_dataset_features(subset_dict, unified_features, feature_defaults)
        
        print(f"  Train: {len(subset_dict['train'])} (CV17 only)")
        print(f"  Validation: {len(subset_dict['validation'])} (CV17)")
        print(f"  Test: {len(subset_dict['test'])} (CV17)")
        print(f"  Total: {sum(len(split) for split in subset_dict.values())} samples")
        
        return subset_dict

    def create_subset_4_high_quality_cv(self, high_quality_synthetic, cv_dataset, unified_features, feature_defaults):
        """Create subset 4: high_quality_cv (high quality synthetic + CV train)"""
        print("\n🎯 Creating Subset 4: high_quality_cv")
        print("-" * 50)
        
        print(f"  Combining datasets...")
        print(f"    High quality synthetic: {len(high_quality_synthetic)} samples")
        print(f"    CV17 train: {len(cv_dataset['train'])} samples")
        
        # Fix audio feature mismatch by standardizing both datasets
        print("  Standardizing audio features...")
        
        from datasets import Audio
        standard_audio = Audio(sampling_rate=None, mono=True, decode=True)
        
        high_quality_synthetic = high_quality_synthetic.cast_column("audio", standard_audio)
        cv_train = cv_dataset['train'].cast_column("audio", standard_audio)
        
        # Combine high quality synthetic + CV train
        combined_train = concatenate_datasets([high_quality_synthetic, cv_train])
        
        print(f"  ✅ Combined train: {len(combined_train)} samples")
        
        subset_dict = DatasetDict({
            'train': combined_train,
            'validation': cv_dataset['validation'],  # CV17 validation
            'test': cv_dataset['test']  # CV17 test
        })
        
        # Normalize features across all splits using unified schema
        subset_dict = self.normalize_dataset_features(subset_dict, unified_features, feature_defaults)
        
        print(f"  Train: {len(subset_dict['train'])} (high quality synthetic + CV17)")
        print(f"  Validation: {len(subset_dict['validation'])} (CV17)")
        print(f"  Test: {len(subset_dict['test'])} (CV17)")
        print(f"  Total: {sum(len(split) for split in subset_dict.values())} samples")
        
        return subset_dict

    def create_subset_5_cv_fully_synthetic(self, current_dataset, cv_dataset, unified_features, feature_defaults):
        """Create subset 5: cv_fully_synthetic (CV train + all synthetic)"""
        print("\n🎯 Creating Subset 5: cv_fully_synthetic")
        print("-" * 50)
        
        print(f"  Combining datasets...")
        print(f"    CV17 train: {len(cv_dataset['train'])} samples")
        print(f"    Fully synthetic: {len(current_dataset['train'])} samples")
        
        # Fix audio feature mismatch by standardizing both datasets
        print("  Standardizing audio features...")
        
        from datasets import Audio
        standard_audio = Audio(sampling_rate=None, mono=True, decode=True)
        
        cv_train = cv_dataset['train'].cast_column("audio", standard_audio)
        synthetic_train = current_dataset['train'].cast_column("audio", standard_audio)
        
        # Combine CV train + all synthetic train
        combined_train = concatenate_datasets([cv_train, synthetic_train])
        
        print(f"  ✅ Combined train: {len(combined_train)} samples")
        
        subset_dict = DatasetDict({
            'train': combined_train,
            'validation': cv_dataset['validation'],  # CV17 validation
            'test': cv_dataset['test']  # CV17 test
        })
        
        # Normalize features across all splits using unified schema
        subset_dict = self.normalize_dataset_features(subset_dict, unified_features, feature_defaults)
        
        print(f"  Train: {len(subset_dict['train'])} (CV17 + fully synthetic)")
        print(f"  Validation: {len(subset_dict['validation'])} (CV17)")
        print(f"  Test: {len(subset_dict['test'])} (CV17)")
        print(f"  Total: {sum(len(split) for split in subset_dict.values())} samples")
        
        return subset_dict
    
    def create_unified_schema(self, subsets):
        """Create a unified schema that includes all possible features"""
        print("🔧 Creating unified schema from all subsets...")
        
        # Collect all unique features across all subsets and splits
        all_features = {}
        
        # Standard audio feature for all datasets
        from datasets import Audio, Value
        all_features['audio'] = Audio(sampling_rate=None, mono=True, decode=True)
        all_features['text'] = Value(dtype='string', id=None)
        
        # Add all other features we've seen
        feature_defaults = {
            'voice': ('string', ''),
            'model': ('string', ''),
            'text_length': ('int64', 0),
            'file_size_bytes': ('int64', 0),
            'estimated_duration': ('float64', 0.0),
            'generation_status': ('string', ''),
            'client_id': ('string', ''),
            'up_votes': ('int64', 0),
            'down_votes': ('int64', 0),
            'age': ('string', ''),
            'gender': ('string', ''),
            'accent': ('string', ''),
            'locale': ('string', 'nl'),
            'dataset_source': ('string', '')
        }
        
        for feature_name, (dtype, default_val) in feature_defaults.items():
            all_features[feature_name] = Value(dtype=dtype, id=None)
        
        print(f"  Unified schema has {len(all_features)} features: {list(all_features.keys())}")
        return all_features, feature_defaults
    def normalize_dataset_features(self, dataset_dict, unified_features=None, feature_defaults=None):
        """Normalize all splits to have the same features using unified schema"""
        print("  🔧 Normalizing dataset features across all splits...")
        
        if unified_features is None or feature_defaults is None:
            # Create unified schema if not provided
            unified_features, feature_defaults = self.create_unified_schema({})
        
        # Standardize audio feature
        from datasets import Audio
        standard_audio = Audio(sampling_rate=None, mono=True, decode=True)
        
        normalized_dict = {}
        
        for split_name, split_data in dataset_dict.items():
            print(f"    Normalizing {split_name} split...")
            
            # First, standardize audio column to fix sampling rate issues
            if 'audio' in split_data.column_names:
                split_data = split_data.cast_column("audio", standard_audio)
            
            # Add missing columns with default values
            for col_name in unified_features.keys():
                if col_name not in split_data.column_names:
                    print(f"      Adding missing column '{col_name}' to {split_name}")
                    
                    # Get default value from feature_defaults
                    if col_name in feature_defaults:
                        _, default_value = feature_defaults[col_name]
                    elif col_name == 'audio':
                        continue  # Skip audio, already handled
                    elif col_name == 'text':
                        default_value = ''
                    else:
                        default_value = ''
                    
                    # Add column with default values
                    split_data = split_data.add_column(col_name, [default_value] * len(split_data))
            
            # Remove extra columns not in unified schema
            columns_to_remove = []
            for col_name in split_data.column_names:
                if col_name not in unified_features:
                    columns_to_remove.append(col_name)
            
            if columns_to_remove:
                print(f"      Removing extra columns from {split_name}: {columns_to_remove}")
                split_data = split_data.remove_columns(columns_to_remove)
            
            # Reorder columns to match unified schema
            unified_columns = list(unified_features.keys())
            current_columns = split_data.column_names
            
            if set(current_columns) == set(unified_columns):
                if current_columns != unified_columns:
                    print(f"      Reordering columns in {split_name}")
                    split_data = split_data.select_columns(unified_columns)
            else:
                print(f"      Warning: Column mismatch in {split_name}")
                print(f"        Expected: {unified_columns}")
                print(f"        Got: {current_columns}")
            
            normalized_dict[split_name] = split_data
        
        print("  ✅ Feature normalization complete")
        return DatasetDict(normalized_dict)
    
    def verify_subset_consistency(self, subsets):
        """Verify all subsets have consistent schemas"""
        print("\n🔍 Verifying subset consistency...")
        
        # Get unified schema as reference
        unified_features, _ = self.create_unified_schema({})
        print(f"  Using unified schema as reference: {list(unified_features.keys())}")
        
        all_consistent = True
        
        for subset_name, subset_data in subsets.items():
            train_features = subset_data['train'].features
            
            # Compare with unified schema
            ref_keys = set(unified_features.keys())
            subset_keys = set(train_features.keys())
            
            if ref_keys == subset_keys:
                print(f"  ✅ {subset_name}: Schema matches unified schema")
            else:
                print(f"  ⚠️ {subset_name}: Schema differences detected")
                all_consistent = False
                missing = ref_keys - subset_keys
                extra = subset_keys - ref_keys
                if missing:
                    print(f"    Missing fields: {missing}")
                if extra:
                    print(f"    Extra fields: {extra}")
            
            # Check audio consistency
            sample = subset_data['train'][0]
            audio_sr = sample['audio']['sampling_rate']
            # Now we expect either None (flexible) or the target rate
            if audio_sr is None:
                print(f"  ✅ {subset_name}: Audio sampling rate flexible (None)")
            elif audio_sr == self.target_sr:
                print(f"  ✅ {subset_name}: Audio sampling rate correct ({audio_sr}Hz)")
            else:
                print(f"  ❌ {subset_name}: Wrong sampling rate ({audio_sr}Hz)")
                all_consistent = False
            
            # Check if all splits have same features
            for split_name in ['validation', 'test']:
                if split_name in subset_data:
                    split_features = subset_data[split_name].features
                    if split_features == train_features:
                        print(f"  ✅ {subset_name}.{split_name}: Features match train")
                    else:
                        print(f"  ❌ {subset_name}.{split_name}: Features don't match train")
                        all_consistent = False
                        # Show differences
                        train_keys = set(train_features.keys())
                        split_keys = set(split_features.keys())
                        if train_keys != split_keys:
                            missing = train_keys - split_keys
                            extra = split_keys - train_keys
                            if missing:
                                print(f"      Missing in {split_name}: {missing}")
                            if extra:
                                print(f"      Extra in {split_name}: {extra}")
        
        if all_consistent:
            print("  🎉 All subsets are fully consistent!")
        else:
            print("  ⚠️ Some inconsistencies detected - check normalization")

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
        """Create all five subsets efficiently"""
        print("🚀 CREATING ALL DATASET SUBSETS (FAST VERSION)")
        print("=" * 60)
        
        # Load all required data
        print("📂 Loading all datasets...")
        current_dataset = self.load_current_dataset()
        cv_dataset = self.load_common_voice_data()
        filtered_indices, high_quality_indices = self.load_filtered_synthetic_data()
        
        # Create unified schema for all subsets
        print("\n🔧 Creating unified feature schema...")
        unified_features, feature_defaults = self.create_unified_schema({})
        
        # Create filtered datasets
        filtered_synthetic = self.create_filtered_synthetic_dataset(
            current_dataset['train'], 
            filtered_indices
        )
        
        high_quality_synthetic = self.create_high_quality_synthetic_dataset(
            current_dataset['train'],
            high_quality_indices
        )
        
        print("\n🎯 Creating all subsets...")
        subsets = {}
        
        # Subset 1: fully_synthetic
        subsets['fully_synthetic'] = self.create_subset_1_fully_synthetic(
            current_dataset, cv_dataset, unified_features, feature_defaults
        )
        
        # Subset 2: mixed_cv_synthetic
        subsets['mixed_cv_synthetic'] = self.create_subset_2_mixed(
            filtered_synthetic, cv_dataset, unified_features, feature_defaults
        )
        
        # Subset 3: cv_only
        subsets['cv_only'] = self.create_subset_3_cv_only(
            cv_dataset, unified_features, feature_defaults
        )

        # Subset 4: high_quality_cv
        subsets['high_quality_cv'] = self.create_subset_4_high_quality_cv(
            high_quality_synthetic, cv_dataset, unified_features, feature_defaults
        )

        # Subset 5: cv_fully_synthetic
        subsets['cv_fully_synthetic'] = self.create_subset_5_cv_fully_synthetic(
            current_dataset, cv_dataset, unified_features, feature_defaults
        )
        
        # Verify consistency
        self.verify_subset_consistency(subsets)
        
        return subsets
    
    def save_subsets_locally(self, subsets, output_dir="dataset_subsets_nl"):
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