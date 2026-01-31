#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import os
import numpy as np
from transformers import AutoTokenizer, AutoFeatureExtractor
import librosa
from speech_transcript_embeddings.training_multimodal.trainer_multimodal.trainer_unfreeze import EnhancedAudioTextModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EDIT THESE VALUES
checkpoint_path = "PATH TO MODEL"
audio_path = "PATH TO AUDIO FILE"

# LIST OF TEXTS TO COMPARE
texts = [
    "INPUT COMAPRISONS TEXT HERE"
]

# Load model
print(f"Loading model...")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

model = EnhancedAudioTextModel(
    text_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    audio_model_name="facebook/w2v-bert-2.0",
    projection_dim=checkpoint.get('projection_dim', 768),
    use_cross_modal=checkpoint.get('use_cross_modal', False),
    use_attentive_pooling=checkpoint.get('use_attentive_pooling', False),
    use_word_alignment=checkpoint.get('use_word_alignment', False),
    freeze_encoders="none"
)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Load tokenizer and feature extractor
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

# Process audio once
print(f"Processing audio: {audio_path}")
audio_array, _ = librosa.load(audio_path, sr=16000)
audio_features = feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")
audio_input = audio_features["input_features" if "input_features" in audio_features else "input_values"].to(device)
audio_mask = audio_features.get("attention_mask", None)
if audio_mask is not None:
    audio_mask = audio_mask.to(device)

# Process each text
results = []
print(f"\nComparing {len(texts)} texts:")
print("-" * 80)

for i, text in enumerate(texts):
    # Process text
    text_encoding = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    
    # Create batch
    batch = {
        "input_ids_pos": text_encoding["input_ids"].to(device),
        "attention_mask_pos": text_encoding["attention_mask"].to(device),
        "input_ids_neg": text_encoding["input_ids"].to(device),
        "attention_mask_neg": text_encoding["attention_mask"].to(device),
        "input_values": audio_input,
        "attention_mask_audio": audio_mask
    }
    
    # Run model
    with torch.no_grad():
        text_emb, _, audio_emb = model(batch)
        
        # Compute similarity (0-1 range)
        cosine_sim = (text_emb * audio_emb).sum(dim=1).item()
        similarity_norm = (cosine_sim + 1) / 2
    
    # Get alignment scores
    alignment_data = {}
    if hasattr(model, 'last_pos_alignment_scores') and model.last_pos_alignment_scores is not None:
        alignment_scores_raw = model.last_pos_alignment_scores.squeeze(0)
        # Convert to 0-1 range using sigmoid
        alignment_scores = torch.sigmoid(alignment_scores_raw).cpu().numpy()
        attention_mask = text_encoding["attention_mask"].squeeze(0).cpu().numpy()
        
        # Average alignment
        valid_scores = alignment_scores[attention_mask == 1]
        avg_alignment = valid_scores.mean()
        
        # Get tokens and scores
        tokens = tokenizer.convert_ids_to_tokens(text_encoding["input_ids"].squeeze(0).tolist())
        token_scores = []
        for token, score, mask in zip(tokens, alignment_scores, attention_mask):
            if mask == 1 and token not in ['<s>', '</s>', '<pad>']:
                token_scores.append((token, score))
        
        alignment_data = {
            "avg": avg_alignment,
            "tokens": token_scores
        }
    
    # Store result
    results.append({
        "text": text,
        "similarity": similarity_norm,
        "alignment": alignment_data
    })
    
    # Print result
    print(f"\n[{i+1}] Text: {text}")
    print(f"    Similarity: {similarity_norm:.4f}")
    if alignment_data:
        print(f"    Avg alignment: {alignment_data['avg']:.4f}")
        print(f"    Token alignments:")
        for token, score in alignment_data['tokens']:
            print(f"      {token:20s} {score:.4f}")

# Print summary
print("\n" + "=" * 80)
print("SUMMARY (sorted by similarity):")
print("-" * 80)
results_sorted = sorted(results, key=lambda x: x['similarity'], reverse=True)
for i, r in enumerate(results_sorted):
    align_str = f", align={r['alignment']['avg']:.4f}" if r['alignment'] else ""
    print(f"{i+1}. {r['similarity']:.4f}{align_str} - {r['text']}")