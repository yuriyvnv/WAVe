#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np

def test_time_padding():
    """Test the fixed time dimension padding logic"""
    
    print("Testing time dimension padding fix...")
    
    # Simulate audio tensors with different time lengths (like your error)
    audio_tensors = [
        torch.randn(107, 160),  # 107 time frames, 160 features
        torch.randn(148, 160),  # 148 time frames, 160 features  
        torch.randn(26, 160),   # 26 time frames, 160 features
        torch.randn(44, 160),   # 44 time frames, 160 features
    ]
    
    print("Original tensor shapes:")
    for i, tensor in enumerate(audio_tensors):
        print(f"  Tensor {i}: {tensor.shape}")
    
    # Find max time length
    time_lengths = [af.shape[0] for af in audio_tensors]
    max_time_length = max(time_lengths)
    
    print(f"\nTime lengths: {time_lengths}")
    print(f"Max time length: {max_time_length}")
    
    # Apply the fixed padding logic
    padded_tensors = []
    
    for i, af in enumerate(audio_tensors):
        print(f"\nProcessing tensor {i}: {af.shape}")
        
        if af.shape[0] < max_time_length:  # If time frames < max time frames
            print(f"  Needs padding: {af.shape[0]} -> {max_time_length}")
            
            if af.dim() == 2:  # [time_frames, features]
                padding_needed = max_time_length - af.shape[0]
                print(f"  Creating padding: ({padding_needed}, {af.shape[1]})")
                padding = torch.zeros(padding_needed, af.shape[1])
                af_padded = torch.cat([af, padding], dim=0)  # Concatenate on time dimension
                print(f"  After padding: {af_padded.shape}")
            else:
                print(f"  Unexpected dimensions: {af.dim()}")
                af_padded = af
        else:
            print(f"  No padding needed")
            af_padded = af
        
        padded_tensors.append(af_padded)
    
    # Test stacking
    print(f"\nTesting stacking...")
    print("Padded tensor shapes:")
    for i, tensor in enumerate(padded_tensors):
        print(f"  Tensor {i}: {tensor.shape}")
    
    try:
        stacked = torch.stack(padded_tensors)
        print(f"\n✅ SUCCESS! Stacked tensor shape: {stacked.shape}")
        print(f"Expected shape: [batch_size={len(audio_tensors)}, time={max_time_length}, features=160]")
        return True
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return False

def test_mask_padding():
    """Test mask padding with time dimension"""
    
    print("\n" + "="*50)
    print("Testing mask padding...")
    
    # Simulate 1D masks with different time lengths
    masks = [
        torch.ones(107),  # 107 time steps
        torch.ones(148),  # 148 time steps
        torch.ones(26),   # 26 time steps  
        torch.ones(44),   # 44 time steps
    ]
    
    print("Original mask shapes:")
    for i, mask in enumerate(masks):
        print(f"  Mask {i}: {mask.shape}")
    
    max_time_length = max([mask.shape[0] for mask in masks])
    print(f"Max time length: {max_time_length}")
    
    padded_masks = []
    
    for i, mask in enumerate(masks):
        print(f"\nProcessing mask {i}: {mask.shape}")
        
        if mask.shape[0] < max_time_length:
            print(f"  Needs padding: {mask.shape[0]} -> {max_time_length}")
            
            if mask.dim() == 1:
                padding_needed = max_time_length - mask.shape[0]
                mask_padding = torch.zeros(padding_needed)
                mask_padded = torch.cat([mask, mask_padding], dim=0)
                print(f"  After padding: {mask_padded.shape}")
            else:
                print(f"  Unexpected mask dimensions: {mask.dim()}")
                mask_padded = mask
        else:
            print(f"  No padding needed")
            mask_padded = mask
        
        padded_masks.append(mask_padded)
    
    # Test stacking masks
    try:
        stacked_masks = torch.stack(padded_masks)
        print(f"\n✅ Mask stacking SUCCESS! Shape: {stacked_masks.shape}")
        return True
    except Exception as e:
        print(f"\n❌ Mask stacking FAILED: {e}")
        return False

if __name__ == "__main__":
    success1 = test_time_padding()
    success2 = test_mask_padding()
    
    if success1 and success2:
        print(f"\n🎉 ALL TESTS PASSED! The fix should work!")
    else:
        print(f"\n❌ Some tests failed. Need to investigate further.")