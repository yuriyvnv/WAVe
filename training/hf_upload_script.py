#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datasets import load_from_disk
from pathlib import Path
import time

def upload_subsets():
    """Simple script to upload all subsets to HuggingFace"""
    
    print("🚀 SIMPLE HUGGINGFACE UPLOAD")
    print("=" * 50)
    
    # Configuration
    repo_id = "yuriyvnv/synthetic_transcript_nl"
    subsets_dir = Path("dataset_subsets_nl")
    
    # Check if datasets exist
    if not subsets_dir.exists():
        print("❌ dataset_subsets/ directory not found!")
        print("   Please run dataset_subset_creator.py first")
        return
    
    # List available subsets
    available_subsets = [d.name for d in subsets_dir.iterdir() if d.is_dir()]
    print(f"📂 Found subsets: {available_subsets}")
    
    if not available_subsets:
        print("❌ No subset directories found!")
        return
    
    # Confirm upload
    print(f"\n📤 Will upload to: {repo_id}")
    print(f"🎯 Subsets to upload: {len(available_subsets)}")
    for subset in available_subsets:
        print(f"   - {subset}")
    
    response = input(f"\n❓ Proceed with upload? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Upload cancelled")
        return
    
    # Upload each subset
    successful_uploads = []
    failed_uploads = []
    
    for subset_name in available_subsets:
        print(f"\n📤 Uploading {subset_name}...")
        
        try:
            start_time = time.time()
            
            # Load the dataset from disk
            dataset_path = subsets_dir / subset_name
            print(f"   Loading from {dataset_path}...")
            dataset = load_from_disk(str(dataset_path))
            
            # Show dataset info
            total_samples = sum(len(split) for split in dataset.values())
            print(f"   Dataset loaded: {total_samples:,} total samples")
            for split_name, split_data in dataset.items():
                print(f"     {split_name}: {len(split_data):,} samples")
            
            # Push to HuggingFace Hub
            print(f"   Pushing to hub...")
            dataset.push_to_hub(
                repo_id,
                config_name=subset_name,
                commit_message=f"Add {subset_name} subset with {total_samples:,} samples"
            )
            
            elapsed_time = time.time() - start_time
            print(f"   ✅ {subset_name} uploaded successfully! ({elapsed_time:.1f}s)")
            successful_uploads.append(subset_name)
            
        except Exception as e:
            print(f"   ❌ Failed to upload {subset_name}: {e}")
            failed_uploads.append(subset_name)
            continue
    
    # Summary
    print(f"\n📊 UPLOAD SUMMARY")
    print("=" * 50)
    print(f"✅ Successful: {len(successful_uploads)}")
    for subset in successful_uploads:
        print(f"   - {subset}")
    
    if failed_uploads:
        print(f"\n❌ Failed: {len(failed_uploads)}")
        for subset in failed_uploads:
            print(f"   - {subset}")
    
    if len(successful_uploads) == len(available_subsets):
        print(f"\n🎉 ALL SUBSETS UPLOADED SUCCESSFULLY!")
        print(f"\n📖 Usage:")
        print(f"from datasets import load_dataset")
        for subset in successful_uploads:
            print(f"dataset = load_dataset('{repo_id}', '{subset}')")
    else:
        print(f"\n⚠️ Some uploads failed. Check errors above.")

def test_uploads():
    """Quick test to verify uploads worked"""
    
    print(f"\n🧪 Testing uploaded subsets...")
    
    repo_id = "yuriyvnv/synthetic_transcript_nl"
    subsets_to_test = ["fully_synthetic", "cv_only", "high_quality_cv", "mixed_cv_synthetic_all", "cv_fully_synthetic"]
    
    for subset_name in subsets_to_test:
        try:
            print(f"   Testing {subset_name}...")
            dataset = load_dataset(repo_id, subset_name)
            total = sum(len(split) for split in dataset.values())
            print(f"   ✅ {subset_name}: {total:,} samples loaded")
        except Exception as e:
            print(f"   ❌ {subset_name}: Failed - {e}")

if __name__ == "__main__":
    upload_subsets()
    
    # Ask if user wants to test
    test_response = input(f"\n❓ Test the uploads? (y/N): ").strip().lower()
    if test_response == 'y':
        test_uploads()
    
    print(f"\n🎉 Done!")