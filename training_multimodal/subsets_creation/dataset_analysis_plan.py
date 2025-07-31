#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pandas as pd
from datasets import load_dataset
import os
from pathlib import Path

def analyze_current_situation():
    """Analyze current dataset and filtered results to plan subset creation"""
    
    print("🔍 DATASET ANALYSIS & PLANNING")
    print("=" * 60)
    
    # 1. Analyze current HuggingFace dataset
    print("\n1. CURRENT HUGGINGFACE DATASET ANALYSIS")
    print("-" * 40)
    
    try:
        current_dataset = load_dataset("yuriyvnv/synthetic_transcript_nl")
        print(f"✅ Successfully loaded current dataset")
        
        for split_name, split_data in current_dataset.items():
            print(f"  {split_name}: {len(split_data)} samples")
            if len(split_data) > 0:
                print(f"    Sample keys: {list(split_data[0].keys())}")
                print(f"    Audio sampling rate: {split_data[0]['audio']['sampling_rate']} Hz")
                print(f"    Example text: {split_data[0]['text'][:50]}...")
        
    except Exception as e:
        print(f"❌ Error loading current dataset: {e}")
        return False
    
    # 2. Analyze filtered synthetic data
    print("\n2. FILTERED SYNTHETIC DATA ANALYSIS")
    print("-" * 40)
    
    filtered_dir = Path("filtered_datasets_nl")
    if filtered_dir.exists():
        print(f"✅ Found filtered datasets directory")
        
        files = {
            "high_quality": "high_quality_samples.json",
            "medium_quality": "medium_quality_samples.json", 
            "low_quality": "low_quality_samples.json"
        }
        
        quality_counts = {}
        for quality, filename in files.items():
            filepath = filtered_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    quality_counts[quality] = len(data)
                    print(f"  {quality}: {len(data)} samples")
            else:
                print(f"  ❌ {quality}: File not found")
                quality_counts[quality] = 0
        
        total_high_medium = quality_counts['high_quality'] + quality_counts['medium_quality']
        print(f"  📊 High + Medium combined: {total_high_medium} samples")
        
    else:
        print(f"❌ Filtered datasets directory not found")
        return False
    
    # 3. Analyze Common Voice 17 Portuguese
    print("\n3. COMMON VOICE 17 PORTUGUESE ANALYSIS")
    print("-" * 40)
    
    try:
        cv_dataset = load_dataset("mozilla-foundation/common_voice_17_0", "pt")
        print(f"✅ Successfully loaded Common Voice 17 Portuguese")
        
        cv_sizes = {}
        for split_name, split_data in cv_dataset.items():
            cv_sizes[split_name] = len(split_data)
            print(f"  {split_name}: {len(split_data)} samples")
            if len(split_data) > 0:
                print(f"    Audio sampling rate: {split_data[0]['audio']['sampling_rate']} Hz")
                print(f"    Example text: {split_data[0]['sentence'][:50]}...")
        
    except Exception as e:
        print(f"❌ Error loading Common Voice: {e}")
        return False
    
    # 4. Calculate subset sizes
    print("\n4. PLANNED SUBSET SIZES")
    print("-" * 40)
    
    # Current synthetic data size (from HF dataset)
    synthetic_size = len(current_dataset['train'])
    cv_train_size = cv_sizes.get('train', 0)
    cv_val_size = cv_sizes.get('validation', 0) 
    cv_test_size = cv_sizes.get('test', 0)
    
    print(f"Subset 1 - fully_synthetic:")
    print(f"  Train: {synthetic_size} (all synthetic)")
    print(f"  Validation: {cv_val_size} (CV17)")
    print(f"  Test: {cv_test_size} (CV17)")
    print(f"  Total: {synthetic_size + cv_val_size + cv_test_size}")
    
    print(f"\nSubset 2 - mixed_cv_synthetic:")
    print(f"  Train: {total_high_medium} (filtered synthetic) + {cv_train_size} (CV17) = {total_high_medium + cv_train_size}")
    print(f"  Validation: {cv_val_size} (CV17)")
    print(f"  Test: {cv_test_size} (CV17)")
    print(f"  Total: {total_high_medium + cv_train_size + cv_val_size + cv_test_size}")
    
    print(f"\nSubset 3 - cv_only:")
    print(f"  Train: {cv_train_size} (CV17)")
    print(f"  Validation: {cv_val_size} (CV17)")
    print(f"  Test: {cv_test_size} (CV17)")
    print(f"  Total: {cv_train_size + cv_val_size + cv_test_size}")
    
    # 5. Identify potential issues
    print("\n5. POTENTIAL ISSUES TO ADDRESS")
    print("-" * 40)
    
    issues = []
    
    # Check audio format consistency
    current_sr = current_dataset['train'][0]['audio']['sampling_rate']
    cv_sr = cv_dataset['train'][0]['audio']['sampling_rate']
    if current_sr != cv_sr:
        issues.append(f"❌ Sampling rate mismatch: Synthetic={current_sr}Hz, CV17={cv_sr}Hz")
    else:
        print(f"✅ Sampling rates match: {current_sr}Hz")
    
    # Check text field consistency
    current_text_field = 'text'
    cv_text_field = 'sentence'
    if current_text_field != cv_text_field:
        issues.append(f"❌ Text field mismatch: Synthetic='{current_text_field}', CV17='{cv_text_field}'")
    else:
        print(f"✅ Text fields consistent")
    
    # Check if we have enough filtered data
    if total_high_medium < synthetic_size * 0.5:
        issues.append(f"⚠️  Filtered data is only {total_high_medium}/{synthetic_size} ({total_high_medium/synthetic_size*100:.1f}%) of original")
    else:
        print(f"✅ Good amount of filtered data: {total_high_medium}/{synthetic_size} ({total_high_medium/synthetic_size*100:.1f}%)")
    
    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    
    # 6. Recommendations
    print("\n6. RECOMMENDATIONS")
    print("-" * 40)
    
    print("✅ Proceed with subset creation if no critical issues above")
    print("✅ Use incremental upload to avoid timeouts")
    print("✅ Test each subset after upload")
    print("✅ Update dataset card with clear documentation")
    
    if current_sr != cv_sr:
        print("🔧 REQUIRED: Resample audio to consistent sampling rate")
    
    if current_text_field != cv_text_field:
        print("🔧 REQUIRED: Standardize text field names")
    
    return True

def create_implementation_plan():
    """Create detailed implementation steps"""
    
    print("\n" + "=" * 60)
    print("📋 IMPLEMENTATION PLAN")
    print("=" * 60)
    
    steps = [
        "1. Load and prepare Common Voice 17 data",
        "2. Load and filter synthetic data (high + medium quality)",
        "3. Standardize audio formats and field names",
        "4. Create subset 1: fully_synthetic",
        "5. Create subset 2: mixed_cv_synthetic", 
        "6. Create subset 3: cv_only",
        "7. Create dataset loading script with subset support",
        "8. Upload to HuggingFace with subset configurations",
        "9. Test loading each subset",
        "10. Update dataset card and documentation"
    ]
    
    for step in steps:
        print(f"  {step}")
    
    print(f"\n⏱️  Estimated time: 2-4 hours")
    print(f"💾 Estimated upload size: ~3-5GB total")

if __name__ == "__main__":
    success = analyze_current_situation()
    if success:
        create_implementation_plan()
        print(f"\n🚀 Ready to proceed? Run the implementation scripts!")
    else:
        print(f"\n❌ Please resolve issues before proceeding")