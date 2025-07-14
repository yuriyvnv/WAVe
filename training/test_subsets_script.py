#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datasets import load_dataset
import numpy as np
import time

def test_subset_loading():
    """Test loading each subset from HuggingFace"""
    
    print("🧪 TESTING DATASET SUBSETS")
    print("=" * 50)
    
    repo_id = "yuriyvnv/synthetic_transcript_pt"
    subsets = ["fully_synthetic", "mixed_cv_synthetic", "cv_only"]
    
    results = {}
    
    for subset_name in subsets:
        print(f"\n📊 Testing {subset_name}")
        print("-" * 30)
        
        try:
            start_time = time.time()
            
            # Load the subset
            print(f"  Loading...")
            dataset = load_dataset(repo_id, subset_name)
            
            load_time = time.time() - start_time
            print(f"  ✅ Loaded in {load_time:.1f} seconds")
            
            # Check structure
            splits = list(dataset.keys())
            print(f"  Splits: {splits}")
            
            subset_info = {}
            total_samples = 0
            
            # Check each split
            for split_name in splits:
                split_data = dataset[split_name]
                subset_info[split_name] = len(split_data)
                total_samples += len(split_data)
                print(f"    {split_name}: {len(split_data):,} samples")
            
            # Check first sample from train split
            if 'train' in dataset and len(dataset['train']) > 0:
                sample = dataset['train'][0]
                
                print(f"  📝 Sample data:")
                print(f"    Fields: {list(sample.keys())}")
                print(f"    Text: '{sample['text'][:60]}...'")
                print(f"    Audio shape: {np.array(sample['audio']['array']).shape}")
                print(f"    Sampling rate: {sample['audio']['sampling_rate']} Hz")
                print(f"    Voice: {sample.get('voice', 'N/A')}")
                print(f"    Source: {sample.get('dataset_source', 'N/A')}")
            
            results[subset_name] = {
                'status': 'success',
                'load_time': load_time,
                'splits': subset_info,
                'total_samples': total_samples,
                'schema': list(sample.keys()) if 'sample' in locals() else []
            }
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[subset_name] = {
                'status': 'error',
                'error': str(e)
            }
    
    return results

def test_schema_consistency(results):
    """Test that all subsets have consistent schemas"""
    
    print(f"\n🔧 TESTING SCHEMA CONSISTENCY")
    print("-" * 50)
    
    # Get schemas from successful loads
    schemas = {}
    for subset_name, result in results.items():
        if result['status'] == 'success' and 'schema' in result:
            schemas[subset_name] = set(result['schema'])
    
    if len(schemas) < 2:
        print("❌ Need at least 2 successful subsets to compare schemas")
        return
    
    # Compare all schemas
    reference_name = list(schemas.keys())[0]
    reference_schema = schemas[reference_name]
    
    print(f"📋 Reference schema ({reference_name}):")
    print(f"   {sorted(reference_schema)}")
    
    all_consistent = True
    
    for subset_name, schema in schemas.items():
        if subset_name == reference_name:
            continue
            
        if schema == reference_schema:
            print(f"✅ {subset_name}: Schema matches")
        else:
            print(f"❌ {subset_name}: Schema differs")
            missing = reference_schema - schema
            extra = schema - reference_schema
            if missing:
                print(f"   Missing: {missing}")
            if extra:
                print(f"   Extra: {extra}")
            all_consistent = False
    
    if all_consistent:
        print(f"🎉 All schemas are consistent!")
    else:
        print(f"⚠️ Schema inconsistencies found")

def test_audio_consistency():
    """Test audio format consistency across subsets"""
    
    print(f"\n🎵 TESTING AUDIO CONSISTENCY")
    print("-" * 50)
    
    repo_id = "yuriyvnv/synthetic_transcript_pt"
    subsets = ["fully_synthetic", "mixed_cv_synthetic", "cv_only"]
    
    audio_info = {}
    
    for subset_name in subsets:
        try:
            print(f"  Checking {subset_name}...")
            dataset = load_dataset(repo_id, subset_name)
            
            # Check audio from train split
            if 'train' in dataset and len(dataset['train']) > 0:
                # Sample first 3 audio files
                sample_count = min(3, len(dataset['train']))
                sampling_rates = []
                
                for i in range(sample_count):
                    sample = dataset['train'][i]
                    sr = sample['audio']['sampling_rate']
                    sampling_rates.append(sr)
                
                unique_rates = list(set(sampling_rates))
                audio_info[subset_name] = unique_rates
                
                if len(unique_rates) == 1 and unique_rates[0] == 16000:
                    print(f"    ✅ Consistent 16kHz audio")
                else:
                    print(f"    ⚠️ Audio rates: {unique_rates}")
            
        except Exception as e:
            print(f"    ❌ Error checking {subset_name}: {e}")
    
    # Check consistency across subsets
    all_rates = set()
    for rates in audio_info.values():
        all_rates.update(rates)
    
    if all_rates == {16000}:
        print(f"🎉 All subsets have consistent 16kHz audio!")
    else:
        print(f"⚠️ Inconsistent audio rates found: {all_rates}")

def test_data_quality():
    """Quick data quality checks"""
    
    print(f"\n📊 TESTING DATA QUALITY")
    print("-" * 50)
    
    repo_id = "yuriyvnv/synthetic_transcript_pt"
    subsets = ["fully_synthetic", "mixed_cv_synthetic", "cv_only"]
    
    for subset_name in subsets:
        try:
            print(f"  Checking {subset_name}...")
            dataset = load_dataset(repo_id, subset_name)
            
            # Check for empty texts
            if 'train' in dataset:
                train_data = dataset['train']
                sample_size = min(100, len(train_data))
                
                empty_texts = 0
                zero_audio = 0
                
                for i in range(sample_size):
                    sample = train_data[i]
                    
                    # Check text
                    if not sample['text'] or len(sample['text'].strip()) == 0:
                        empty_texts += 1
                    
                    # Check audio
                    audio_array = np.array(sample['audio']['array'])
                    if np.all(audio_array == 0):
                        zero_audio += 1
                
                print(f"    Checked {sample_size} samples:")
                if empty_texts == 0:
                    print(f"    ✅ No empty texts found")
                else:
                    print(f"    ⚠️ {empty_texts} empty texts found")
                
                if zero_audio == 0:
                    print(f"    ✅ No zero audio found")
                else:
                    print(f"    ⚠️ {zero_audio} zero audio samples found")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")

def print_summary(results):
    """Print final summary"""
    
    print(f"\n📋 FINAL SUMMARY")
    print("=" * 50)
    
    successful = [name for name, result in results.items() if result['status'] == 'success']
    failed = [name for name, result in results.items() if result['status'] == 'error']
    
    print(f"✅ Successful subsets: {len(successful)}/3")
    for subset in successful:
        total = results[subset]['total_samples']
        load_time = results[subset]['load_time']
        print(f"   {subset}: {total:,} samples (loaded in {load_time:.1f}s)")
    
    if failed:
        print(f"\n❌ Failed subsets: {len(failed)}")
        for subset in failed:
            print(f"   {subset}: {results[subset]['error']}")
    
    if len(successful) == 3:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"\n📖 Your dataset is ready to use:")
        print(f"from datasets import load_dataset")
        print(f"dataset = load_dataset('yuriyvnv/synthetic_transcript_pt', 'fully_synthetic')")
    else:
        print(f"\n⚠️ Some tests failed - check errors above")

def main():
    """Run all tests"""
    
    print("🚀 COMPREHENSIVE DATASET TESTING")
    print("=" * 70)
    
    try:
        # Test basic loading
        results = test_subset_loading()
        
        # Test schema consistency
        test_schema_consistency(results)
        
        # Test audio consistency  
        test_audio_consistency()
        
        # Test data quality
        test_data_quality()
        
        # Print summary
        print_summary(results)
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()