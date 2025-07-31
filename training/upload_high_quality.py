#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from datasets.features import Audio

def load_high_quality_metadata(json_path):
    """Load high quality samples metadata from JSON file"""
    print(f"📥 Loading high quality metadata from {json_path}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        high_quality_data = json.load(f)
    
    print(f"  ✅ Loaded {len(high_quality_data)} high quality sample records")
    
    # Extract sample indices and show some stats
    sample_indices = [item['sample_index'] for item in high_quality_data]
    print(f"  📊 Sample index range: {min(sample_indices)} to {max(sample_indices)}")
    print(f"  📊 Dataset source: {high_quality_data[0].get('dataset_source', 'unknown')}")
    
    return high_quality_data, sample_indices

def extract_high_quality_samples_from_source(sample_indices, source_dataset_name="yuriyvnv/synthetic_transcript_nl", source_config="fully_synthetic"):
    """Extract actual samples from source dataset using sample indices"""
    print(f"\n🎯 Extracting high quality samples from source dataset...")
    print(f"   Source: {source_dataset_name}/{source_config}")
    
    # Load the source dataset (where the high quality samples came from)
    print("📥 Loading source dataset...")
    source_dataset = load_dataset(source_dataset_name, source_config)
    source_train = source_dataset['train']
    
    print(f"  Source train size: {len(source_train)} samples")
    print(f"  Need to extract: {len(sample_indices)} samples")
    
    # Validate indices are within range
    max_index = max(sample_indices)
    if max_index >= len(source_train):
        print(f"  ⚠️ Warning: Max index {max_index} >= dataset size {len(source_train)}")
        # Filter out invalid indices
        valid_indices = [idx for idx in sample_indices if idx < len(source_train)]
        print(f"  🔧 Filtered to {len(valid_indices)} valid indices")
        sample_indices = valid_indices
    
    # Extract the high quality samples
    print("🔄 Extracting samples...")
    high_quality_samples = source_train.select(sample_indices)
    
    print(f"  ✅ Extracted {len(high_quality_samples)} high quality samples")
    
    return high_quality_samples

def add_high_quality_metadata(high_quality_samples, high_quality_metadata):
    """Add similarity and alignment scores to the extracted samples"""
    print("\n🔧 Adding similarity scores and metadata...")
    
    # Create a mapping from sample_index to metadata
    metadata_map = {item['sample_index']: item for item in high_quality_metadata}
    
    # Prepare additional columns
    similarities = []
    alignment_scores = []
    
    for i in range(len(high_quality_samples)):
        # Get the original sample index (this should match the indices we used to extract)
        original_index = high_quality_samples[i].get('sample_index', i)
        
        if original_index in metadata_map:
            meta = metadata_map[original_index]
            similarities.append(meta['similarity'])
            alignment_scores.append(meta.get('alignment_score'))
        else:
            # Fallback if index not found
            similarities.append(None)
            alignment_scores.append(None)
    
    # Add the new columns
    high_quality_samples = high_quality_samples.add_column('similarity_score', similarities)
    high_quality_samples = high_quality_samples.add_column('alignment_score', alignment_scores)
    
    # Update dataset_source to indicate these are filtered high quality
    dataset_sources = ['high_quality_filtered'] * len(high_quality_samples)
    
    # Remove old dataset_source column if it exists and add new one
    if 'dataset_source' in high_quality_samples.column_names:
        high_quality_samples = high_quality_samples.remove_columns(['dataset_source'])
    
    high_quality_samples = high_quality_samples.add_column('dataset_source', dataset_sources)
    
    print(f"  ✅ Added metadata to {len(high_quality_samples)} samples")
    
    return high_quality_samples

def create_cv_high_quality_subset():
    """Create new subset: cv_only + high_quality_samples"""
    print("🚀 Creating cv_only + high_quality subset")
    print("=" * 60)
    
    # Load cv_only subset
    print("📥 Loading cv_only subset...")
    cv_only = load_dataset("yuriyvnv/synthetic_transcript_pt", "cv_only")
    print(f"  Train: {len(cv_only['train'])} samples")
    print(f"  Validation: {len(cv_only['validation'])} samples") 
    print(f"  Test: {len(cv_only['test'])} samples")
    
    # Load high quality samples metadata
    json_path = "/home/yperezhohin/speech_transcript_embeddings/training/filtered_datasets/high_quality_samples.json"
    high_quality_metadata, sample_indices = load_high_quality_metadata(json_path)
    
    # Extract actual high quality samples from source dataset
    # Assuming they came from fully_synthetic - adjust if different
    high_quality_samples = extract_high_quality_samples_from_source(
        sample_indices, 
        source_dataset_name="yuriyvnv/synthetic_transcript_pt",
        source_config="fully_synthetic"  # Change this if your samples came from a different config
    )
    
    # Add similarity scores and metadata
    high_quality_samples = add_high_quality_metadata(high_quality_samples, high_quality_metadata)
    
    # Verify the datasets can be concatenated
    print("\n🔍 Verifying dataset compatibility...")
    cv_features = set(cv_only['train'].features.keys())
    hq_features = set(high_quality_samples.features.keys())
    
    print(f"  CV features: {sorted(cv_features)}")
    print(f"  HQ features: {sorted(hq_features)}")
    
    # Find common features
    common_features = cv_features.intersection(hq_features)
    cv_only_features = cv_features - hq_features
    hq_only_features = hq_features - cv_features
    
    print(f"  Common features: {len(common_features)}")
    if cv_only_features:
        print(f"  CV-only features: {sorted(cv_only_features)}")
    if hq_only_features:
        print(f"  HQ-only features: {sorted(hq_only_features)}")
    
    # Align features for concatenation
    if cv_features != hq_features:
        print("🔧 Aligning features for concatenation...")
        
        # Remove HQ-only features that aren't in CV
        if hq_only_features:
            print(f"  Removing HQ-only features: {sorted(hq_only_features)}")
            high_quality_samples = high_quality_samples.remove_columns(list(hq_only_features))
        
        # Add missing CV features to HQ with default values
        if cv_only_features:
            print(f"  Adding missing CV features to HQ: {sorted(cv_only_features)}")
            for feature in cv_only_features:
                # Get a sample value from CV dataset to determine type
                sample_value = cv_only['train'][0][feature]
                if isinstance(sample_value, str):
                    default_values = [''] * len(high_quality_samples)
                elif isinstance(sample_value, (int, float)):
                    default_values = [0] * len(high_quality_samples)
                else:
                    default_values = [None] * len(high_quality_samples)
                
                high_quality_samples = high_quality_samples.add_column(feature, default_values)
        
        # Reorder columns to match CV
        cv_column_order = cv_only['train'].column_names
        high_quality_samples = high_quality_samples.select_columns(cv_column_order)
        
        print("  ✅ Features aligned successfully")
    else:
        print("  ✅ Features already match perfectly")
    
    # Create combined train dataset
    print("\n🎯 Creating combined train dataset...")
    print(f"  CV train: {len(cv_only['train'])} samples")
    print(f"  High quality: {len(high_quality_samples)} samples")
    
    # Concatenate datasets
    combined_train = concatenate_datasets([cv_only['train'], high_quality_samples])
    print(f"  ✅ Combined train: {len(combined_train)} samples")
    
    # Create new subset
    new_subset = DatasetDict({
        'train': combined_train,
        'validation': cv_only['validation'],  # Keep CV validation
        'test': cv_only['test']  # Keep CV test
    })
    
    print(f"\n📊 New subset summary:")
    print(f"  Train: {len(new_subset['train'])} samples (CV + High Quality)")
    print(f"  Validation: {len(new_subset['validation'])} samples (CV)")
    print(f"  Test: {len(new_subset['test'])} samples (CV)")
    
    return new_subset

def verify_subset_consistency(new_subset):
    """Verify the new subset consistency"""
    print("\n🔍 Verifying subset consistency...")
    
    # Check features consistency
    train_features = set(new_subset['train'].features.keys())
    val_features = set(new_subset['validation'].features.keys()) 
    test_features = set(new_subset['test'].features.keys())
    
    if train_features == val_features == test_features:
        print("  ✅ All splits have consistent features")
    else:
        print("  ⚠️ Feature mismatch detected!")
        print(f"    Train: {sorted(train_features)}")
        print(f"    Val: {sorted(val_features)}")
        print(f"    Test: {sorted(test_features)}")
    
    # Check audio sampling rates
    try:
        sample_train = new_subset['train'][0]['audio']['sampling_rate']
        sample_val = new_subset['validation'][0]['audio']['sampling_rate']
        sample_test = new_subset['test'][0]['audio']['sampling_rate']
        
        if sample_train == sample_val == sample_test == 16000:
            print(f"  ✅ All audio at 16kHz")
        else:
            print(f"  ⚠️ Sampling rate mismatch: train={sample_train}, val={sample_val}, test={sample_test}")
    except Exception as e:
        print(f"  ⚠️ Could not verify audio sampling rates: {e}")
    
    # Show dataset source distribution in train
    print("\n📊 Dataset source distribution in train:")
    sources = {}
    for item in new_subset['train']:
        source = item.get('dataset_source', 'unknown')
        sources[source] = sources.get(source, 0) + 1
    
    total = len(new_subset['train'])
    for source, count in sources.items():
        percentage = count / total * 100
        print(f"    {source}: {count:,} samples ({percentage:.1f}%)")
    
    # Show similarity score statistics if available
    similarities = []
    for item in new_subset['train']:
        if 'similarity_score' in item and item['similarity_score'] is not None:
            similarities.append(item['similarity_score'])
    
    if similarities:
        print(f"\n📈 Similarity score statistics for high quality samples:")
        print(f"    Count: {len(similarities)}")
        print(f"    Mean: {np.mean(similarities):.4f}")
        print(f"    Min: {np.min(similarities):.4f}")
        print(f"    Max: {np.max(similarities):.4f}")

def push_new_subset_to_hub(new_subset, repo_id="yuriyvnv/synthetic_transcript_pt"):
    """Push the new subset to HuggingFace Hub"""
    print(f"\n📤 Pushing new subset to {repo_id}...")
    
    subset_name = "cv_high_quality"
    
    try:
        # Push with the new configuration name
        new_subset.push_to_hub(
            repo_id=repo_id,
            config_name=subset_name,
            commit_message=f"Add {subset_name} subset (CV + high quality filtered samples with similarity scores)"
        )
        
        print("✅ Successfully pushed new subset!")
        return subset_name
        
    except Exception as e:
        print(f"❌ Error pushing to hub: {e}")
        return None

def main():
    """Main execution"""
    try:
        # Create the new subset
        new_subset = create_cv_high_quality_subset()
        
        if new_subset is None:
            print("❌ Failed to create subset")
            return
        
        # Verify consistency
        verify_subset_consistency(new_subset)
        
        # Save locally first
        print("\n💾 Saving locally for verification...")
        local_path = "cv_high_quality_subset"
        new_subset.save_to_disk(local_path)
        print(f"✅ Saved to ./{local_path}")
        
        # Final sample counts
        print("\n📈 FINAL SAMPLE COUNTS:")
        print("=" * 30)
        print(f"Train split: {len(new_subset['train']):,} samples")
        print(f"Validation split: {len(new_subset['validation']):,} samples")
        print(f"Test split: {len(new_subset['test']):,} samples")
        print(f"Total: {len(new_subset['train']) + len(new_subset['validation']) + len(new_subset['test']):,} samples")
        
        # Ask to push to hub
        response = input("\n❓ Push to HuggingFace Hub? (y/n): ")
        if response.lower() == 'y':
            subset_name = push_new_subset_to_hub(new_subset)
            
            if subset_name:
                print(f"\n📋 To load the new subset:")
                print(f'    dataset = load_dataset("yuriyvnv/synthetic_transcript_pt", "{subset_name}")')
                
                print(f"\n📊 Final uploaded sample counts:")
                print(f"  Train: {len(new_subset['train']):,} samples")
                print(f"  Validation: {len(new_subset['validation']):,} samples") 
                print(f"  Test: {len(new_subset['test']):,} samples")
        else:
            print("📋 Subset saved locally. You can push it later.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()