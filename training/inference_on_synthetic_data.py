#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import os
import numpy as np
import pandas as pd
import json
from transformers import AutoTokenizer, AutoFeatureExtractor
import librosa
from datasets import load_dataset
from tqdm import tqdm
from trainer_unfreeze import EnhancedAudioTextModel
import warnings
warnings.filterwarnings("ignore")

# 🔧 EDIT THESE SETTINGS
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Change to your GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DatasetSimilarityCalculator:
    def __init__(self, checkpoint_path, batch_size=8):
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.device = device
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        self.model = EnhancedAudioTextModel(
            text_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            audio_model_name="facebook/w2v-bert-2.0",
            projection_dim=checkpoint.get('projection_dim', 768),
            use_cross_modal=checkpoint.get('use_cross_modal', False),
            use_attentive_pooling=checkpoint.get('use_attentive_pooling', False),
            use_word_alignment=checkpoint.get('use_word_alignment', False),
            freeze_encoders="none"
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Load tokenizer and feature extractor
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        
        print("Model loaded successfully!")

    def process_audio_batch(self, audio_arrays):
        """Process a batch of audio arrays - FIXED VERSION"""
        audio_features_batch = []
        audio_masks_batch = []
        
        for i, audio_array in enumerate(audio_arrays):
            try:
                # Ensure audio is numpy array and 1D
                audio_array = np.array(audio_array, dtype=np.float32)
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.mean(axis=0)  # Convert stereo to mono if needed
                
                # Ensure audio is not empty
                if len(audio_array) == 0:
                    print(f"Warning: Empty audio array at index {i}, skipping")
                    continue
                
                audio_features = self.feature_extractor(
                    audio_array, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                )
                
                # Get the correct feature key
                if "input_features" in audio_features:
                    audio_input = audio_features["input_features"]
                elif "input_values" in audio_features:
                    audio_input = audio_features["input_values"]
                else:
                    print(f"Warning: No recognized audio features found for sample {i}")
                    continue
                
                audio_mask = audio_features.get("attention_mask", None)
                
                # Handle tensor dimensions properly
                audio_input_squeezed = audio_input.squeeze(0)
                audio_features_batch.append(audio_input_squeezed)
                
                if audio_mask is not None:
                    audio_mask_squeezed = audio_mask.squeeze(0)
                    audio_masks_batch.append(audio_mask_squeezed)
                
            except Exception as e:
                print(f"Error processing audio sample {i}: {e}")
                continue
        
        if not audio_features_batch:
            print("No valid audio features extracted!")
            return None, None
        
        # Find max length for padding - CHECK TIME DIMENSION (first dim)
        time_lengths = [af.shape[0] for af in audio_features_batch]
        max_time_length = max(time_lengths)
        
        print(f"Audio tensor shapes: {[af.shape for af in audio_features_batch[:3]]}...")  # Show first 3
        print(f"Time lengths: {time_lengths}")
        print(f"Max time length for padding: {max_time_length}")
        
        padded_audio = []
        padded_masks = []
        
        for i, af in enumerate(audio_features_batch):
            try:
                # Pad audio features in TIME dimension (dimension 0)
                if af.shape[0] < max_time_length:  # If time frames < max time frames
                    if af.dim() == 2:  # [time_frames, features]
                        padding = torch.zeros(max_time_length - af.shape[0], af.shape[1])  # Pad time dimension
                        af = torch.cat([af, padding], dim=0)  # Concatenate on time dimension
                    elif af.dim() == 1:  # [time_frames] - shouldn't happen but just in case
                        padding = torch.zeros(max_time_length - af.shape[0])
                        af = torch.cat([af, padding], dim=0)
                        af = af.unsqueeze(1)  # Add feature dimension -> [time, 1]
                
                padded_audio.append(af)
                
                # 🔧 FIXED: Handle masks - also pad in TIME dimension
                if i < len(audio_masks_batch):
                    mask = audio_masks_batch[i]
                    if mask.shape[0] < max_time_length:  # If mask time < max time
                        if mask.dim() == 1:
                            # For 1D masks [time], pad on dimension 0
                            mask_padding = torch.zeros(max_time_length - mask.shape[0])
                            mask = torch.cat([mask, mask_padding], dim=0)
                        elif mask.dim() == 2:
                            # For 2D masks [time, features], pad on dimension 0
                            mask_padding = torch.zeros(max_time_length - mask.shape[0], mask.shape[1])
                            mask = torch.cat([mask, mask_padding], dim=0)
                        else:
                            print(f"Warning: Unexpected mask dimensions: {mask.dim()}")
                    padded_masks.append(mask)
                
            except Exception as e:
                print(f"Error padding audio sample {i}: {e}")
                continue
        
        if not padded_audio:
            return None, None
        
        try:
            audio_batch = torch.stack(padded_audio).to(self.device)
            mask_batch = torch.stack(padded_masks).to(self.device) if padded_masks else None
            
            return audio_batch, mask_batch
            
        except Exception as e:
            print(f"Error stacking audio tensors: {e}")
            print("Individual tensor shapes:")
            for i, af in enumerate(padded_audio):
                print(f"  Tensor {i}: {af.shape}")
            return None, None

    def process_text_batch(self, texts):
        """Process a batch of texts"""
        text_encodings = self.tokenizer(
            texts, 
            max_length=128, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        return {
            "input_ids": text_encodings["input_ids"].to(self.device),
            "attention_mask": text_encodings["attention_mask"].to(self.device)
        }

    def calculate_similarities_batch(self, audio_batch, audio_masks, text_encodings):
        """Calculate similarities for a batch"""
        similarities = []
        alignment_scores = []
        
        batch_size = audio_batch.shape[0]
        
        with torch.no_grad():
            for i in range(batch_size):
                try:
                    # Create batch for single sample
                    single_audio = audio_batch[i:i+1]  # Keep batch dimension
                    single_text_ids = text_encodings["input_ids"][i:i+1]
                    single_text_mask = text_encodings["attention_mask"][i:i+1]
                    single_audio_mask = audio_masks[i:i+1] if audio_masks is not None else None
                    
                    batch = {
                        "input_ids_pos": single_text_ids,
                        "attention_mask_pos": single_text_mask,
                        "input_ids_neg": single_text_ids,  # Same as pos for similarity calc
                        "attention_mask_neg": single_text_mask,
                        "input_values": single_audio,
                        "attention_mask_audio": single_audio_mask
                    }
                    
                    text_emb, _, audio_emb = self.model(batch)
                    
                    # Compute similarity (0-1 range)
                    cosine_sim = (text_emb * audio_emb).sum(dim=1).item()
                    similarity_norm = (cosine_sim + 1) / 2
                    similarities.append(similarity_norm)
                    
                    # Get alignment scores if available
                    alignment_score = None
                    if hasattr(self.model, 'last_pos_alignment_scores') and self.model.last_pos_alignment_scores is not None:
                        alignment_scores_raw = self.model.last_pos_alignment_scores.squeeze(0)
                        alignment_scores_sigmoid = torch.sigmoid(alignment_scores_raw).cpu().numpy()
                        attention_mask = single_text_mask.squeeze(0).cpu().numpy()
                        
                        # Average alignment for valid tokens
                        valid_scores = alignment_scores_sigmoid[attention_mask == 1]
                        alignment_score = valid_scores.mean() if len(valid_scores) > 0 else None
                    
                    alignment_scores.append(alignment_score)
                    
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    similarities.append(0.0)
                    alignment_scores.append(None)
        
        return similarities, alignment_scores

    def process_dataset(self, dataset_name="yuriyvnv/synthetic_transcript_pt", 
                       split="train", output_file="dataset_similarities.json",
                       save_every=100, max_samples=None):
        """Process the entire dataset and calculate similarities"""
        
        print(f"Loading dataset {dataset_name}...")
        dataset = load_dataset(dataset_name, split=split)
        
        # Limit dataset size if in debug mode
        if max_samples is not None:
            print(f"Limiting dataset to first {max_samples} samples for debugging")
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        print(f"Dataset loaded. Total samples: {len(dataset)}")
        
        results = []
        processed_count = 0
        
        # Process in batches
        for i in tqdm(range(0, len(dataset), self.batch_size), desc="Processing batches"):
            batch_end = min(i + self.batch_size, len(dataset))
            batch_samples = dataset[i:batch_end]
            
            try:
                # Extract audio and text from batch
                audio_arrays = []
                texts = []
                
                for j in range(len(batch_samples['audio'])):
                    # Get audio array (assuming it's in the 'array' field)
                    audio_data = batch_samples['audio'][j]['array']
                    sample_rate = batch_samples['audio'][j]['sampling_rate']
                    
                    # Convert to numpy array if needed
                    if not isinstance(audio_data, np.ndarray):
                        audio_data = np.array(audio_data, dtype=np.float32)
                    
                    # Resample to 16kHz if needed
                    if sample_rate != 16000:
                        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                    
                    audio_arrays.append(audio_data)
                    texts.append(batch_samples['text'][j])
                
                # Process audio and text
                audio_batch, audio_masks = self.process_audio_batch(audio_arrays)
                
                # Skip batch if audio processing failed
                if audio_batch is None:
                    print(f"Skipping batch {i}-{batch_end} due to audio processing failure")
                    continue
                
                text_encodings = self.process_text_batch(texts)
                
                # Calculate similarities
                similarities, alignment_scores = self.calculate_similarities_batch(
                    audio_batch, audio_masks, text_encodings
                )
                
                # Store results
                for j, (sim, align) in enumerate(zip(similarities, alignment_scores)):
                    sample_idx = i + j
                    result = {
                        "sample_index": int(sample_idx),  # Convert to Python int
                        "text": texts[j],
                        "similarity": float(sim),  # Convert to Python float
                        "alignment_score": float(align) if align is not None else None,  # Convert to Python float
                        "audio_duration": float(len(audio_arrays[j]) / 16000)  # Convert to Python float
                    }
                    
                    # Add any other fields from the original dataset
                    for key in batch_samples.keys():
                        if key not in ['audio', 'text']:
                            if j < len(batch_samples[key]):  # Safety check
                                value = batch_samples[key][j]
                                # Convert numpy types to Python types for JSON serialization
                                if hasattr(value, 'item'):  # numpy scalar
                                    result[key] = value.item()
                                elif isinstance(value, np.ndarray):  # numpy array
                                    result[key] = value.tolist()
                                else:
                                    result[key] = value
                    
                    results.append(result)
                
                processed_count += len(similarities)
                
                # Save intermediate results
                if processed_count % save_every == 0:
                    self.save_results(results, f"{output_file}.temp")
                    print(f"Saved intermediate results: {processed_count} samples processed")
                
            except Exception as e:
                print(f"Error processing batch {i}-{batch_end}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save final results
        self.save_results(results, output_file)
        
        # Print statistics
        self.print_statistics(results)
        
        return results

    def save_results(self, results, filename):
        """Save results to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def print_statistics(self, results):
        """Print statistics about the similarities"""
        similarities = [r['similarity'] for r in results if r['similarity'] is not None]
        alignment_scores = [r['alignment_score'] for r in results if r['alignment_score'] is not None]
        
        print("\n" + "="*50)
        print("SIMILARITY STATISTICS")
        print("="*50)
        print(f"Total samples processed: {len(results)}")
        print(f"Valid similarities: {len(similarities)}")
        
        if similarities:
            print(f"Mean similarity: {np.mean(similarities):.4f}")
            print(f"Median similarity: {np.median(similarities):.4f}")
            print(f"Std similarity: {np.std(similarities):.4f}")
            print(f"Min similarity: {np.min(similarities):.4f}")
            print(f"Max similarity: {np.max(similarities):.4f}")
            
            # Distribution
            high_sim = sum(1 for s in similarities if s > 0.8)
            med_sim = sum(1 for s in similarities if 0.5 <= s <= 0.8)
            low_sim = sum(1 for s in similarities if s < 0.5)
            
            print(f"\nSimilarity distribution:")
            print(f"  High (>0.8): {high_sim} ({high_sim/len(similarities)*100:.1f}%)")
            print(f"  Medium (0.5-0.8): {med_sim} ({med_sim/len(similarities)*100:.1f}%)")
            print(f"  Low (<0.5): {low_sim} ({low_sim/len(similarities)*100:.1f}%)")
        
        if alignment_scores:
            print(f"\nAlignment scores:")
            print(f"Mean alignment: {np.mean(alignment_scores):.4f}")
            print(f"Median alignment: {np.median(alignment_scores):.4f}")

def main():
    # 🔧 EDIT THESE SETTINGS:
    checkpoint_path = "/home/yperezhohin/speech_transcript_embeddings/training/3_alignment_MHGLU_twoWay_loss/best_model_gap.pt"
    dataset_name = "yuriyvnv/synthetic_transcript_pt"
    output_file = "dataset_similarities.json"
    
    # 🔧 DEBUG MODE - Set to True for testing, False for full run
    debug_mode = False  # Change to False for full dataset
    
    if debug_mode:
        print("🧪 DEBUG MODE: Processing only first 100 samples")
        max_samples = 100
        batch_size = 4
        save_every = 20
    else:
        print("🚀 PRODUCTION MODE: Processing full dataset")
        max_samples = None
        batch_size = 150  # Adjust based on your GPU memory
        save_every = 150
    
    # Initialize calculator
    calculator = DatasetSimilarityCalculator(
        checkpoint_path=checkpoint_path,
        batch_size=batch_size
    )
    
    # Process dataset
    results = calculator.process_dataset(
        dataset_name=dataset_name,
        split="train",  # Change to "test" or "validation" if needed
        output_file=output_file,
        save_every=save_every,
        max_samples=max_samples
    )
    
    print(f"\nProcessing complete! Results saved to {output_file}")
    
    # Optionally, save as CSV for easier analysis
    if results:
        df = pd.DataFrame(results)
        csv_output = output_file.replace('.json', '.csv')
        df.to_csv(csv_output, index=False)
        print(f"Results also saved as CSV: {csv_output}")
        
        # Show preview
        print(f"\nPreview of results:")
        print(df[['sample_index', 'similarity', 'alignment_score']].head())
    else:
        print("No results to save!")

if __name__ == "__main__":
    main()