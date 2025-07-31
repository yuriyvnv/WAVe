#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datasets import load_dataset, concatenate_datasets, DatasetDict

def add_mixed_all_subset():
    """Add the mixed_cv_synthetic_all subset to existing dataset"""
    print("🚀 Adding mixed_cv_synthetic_all subset to existing dataset")
    print("=" * 60)
    
    # Load your existing dataset configurations
    print("📥 Loading existing dataset configurations...")
    
    # Load the fully_synthetic subset (ALL synthetic samples)
    print("  Loading fully_synthetic subset...")
    fully_synthetic = load_dataset("yuriyvnv/synthetic_transcript_pt", "fully_synthetic")
    print(f"    Train: {len(fully_synthetic['train'])} samples")
    
    # Load the cv_only subset (already standardized CV17)
    print("  Loading cv_only subset...")
    cv_only = load_dataset("yuriyvnv/synthetic_transcript_pt", "cv_only")
    print(f"    Train: {len(cv_only['train'])} samples")
    
    # Create the new subset: ALL synthetic + CV train
    print("\n🎯 Creating mixed_cv_synthetic_all subset...")
    print(f"  ALL synthetic train: {len(fully_synthetic['train'])} samples")
    print(f"  CV17 train (from cv_only): {len(cv_only['train'])} samples")
    
    # Concatenate ALL synthetic with CV train
    mixed_all_train = concatenate_datasets([
        fully_synthetic['train'], 
        cv_only['train']
    ])
    
    print(f"  ✅ Combined train: {len(mixed_all_train)} samples")
    
    # The new subset uses CV17 validation and test (from cv_only)
    new_subset = DatasetDict({
        'train': mixed_all_train,
        'validation': cv_only['validation'],  # CV17 validation
        'test': cv_only['test']  # CV17 test
    })
    
    print(f"\n📊 New subset summary:")
    print(f"  Train: {len(new_subset['train'])} (ALL synthetic + CV17)")
    print(f"  Validation: {len(new_subset['validation'])} (CV17)")
    print(f"  Test: {len(new_subset['test'])} (CV17)")
    
    return new_subset

def push_new_subset_to_hub(new_subset, repo_id="yuriyvnv/synthetic_transcript_pt"):
    """Push the new subset to HuggingFace Hub"""
    print(f"\n📤 Pushing new subset to {repo_id}...")
    
    # You'll need to be logged in to HuggingFace
    # Run: huggingface-cli login
    
    # Push with the new configuration name
    new_subset.push_to_hub(
        repo_id=repo_id,
        config_name="mixed_cv_synthetic_all",
        commit_message="Add mixed_cv_synthetic_all subset (ALL synthetic + CV17 train)"
    )
    
    print("✅ Successfully pushed new subset!")

def verify_subset_consistency(new_subset):
    """Quick verification of the new subset"""
    print("\n🔍 Verifying subset consistency...")
    
    # Check features are consistent
    train_features = set(new_subset['train'].features.keys())
    val_features = set(new_subset['validation'].features.keys())
    test_features = set(new_subset['test'].features.keys())
    
    if train_features == val_features == test_features:
        print("  ✅ All splits have consistent features")
    else:
        print("  ⚠️ Feature mismatch detected!")
        print(f"    Train features: {train_features}")
        print(f"    Val features: {val_features}")
        print(f"    Test features: {test_features}")
    
    # Check audio sampling rate
    sample_train = new_subset['train'][0]['audio']['sampling_rate']
    sample_val = new_subset['validation'][0]['audio']['sampling_rate']
    sample_test = new_subset['test'][0]['audio']['sampling_rate']
    
    if sample_train == sample_val == sample_test == 16000:
        print(f"  ✅ All audio at 16kHz")
    else:
        print(f"  ⚠️ Sampling rate mismatch: train={sample_train}, val={sample_val}, test={sample_test}")
    
    # Show sample distribution by dataset_source
    print("\n📊 Dataset source distribution in train:")
    sources = {}
    for item in new_subset['train']:
        source = item.get('dataset_source', 'unknown')
        sources[source] = sources.get(source, 0) + 1
    
    for source, count in sources.items():
        print(f"    {source}: {count:,} samples")

def main():
    """Main execution"""
    try:
        # Create the new subset
        new_subset = add_mixed_all_subset()
        
        # Verify consistency
        verify_subset_consistency(new_subset)
        
        # Save locally first (optional)
        print("\n💾 Saving locally for verification...")
        new_subset.save_to_disk("mixed_cv_synthetic_all_subset")
        print("✅ Saved to ./mixed_cv_synthetic_all_subset")
        
        # Push to hub
        response = input("\n❓ Push to HuggingFace Hub? (y/n): ")
        if response.lower() == 'y':
            push_new_subset_to_hub(new_subset)
            
            # Show how to load it
            print("\n📋 To load the new subset:")
            print('    dataset = load_dataset("yuriyvnv/synthetic_transcript_pt", "mixed_cv_synthetic_all")')
        else:
            print("📋 Subset saved locally. You can push it later.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()