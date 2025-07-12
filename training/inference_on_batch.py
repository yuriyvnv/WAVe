#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import os
import numpy as np
from transformers import AutoTokenizer, AutoFeatureExtractor
import librosa
from trainer_unfreeze import EnhancedAudioTextModel
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
checkpoint_path = "/home/yperezhohin/speech_transcript_embeddings/training/3_alignment_MHGLU_twoWay_loss/best_model_gap.pt"
audio_folder = "/home/yperezhohin/speech_transcript_embeddings/synthetic_audio/audios"

texts = [
    "A manhã amanheceu com brisa suave, anunciando um dia cheio de possibilidades.",
    "Lembre‑se de beber água e alongar o corpo antes de começar seu trabalho.",
    "O café na mesa ainda solta vapor, perfumando toda a cozinha.",
    "Escreva seus objetivos do dia e revise‑os quando o sol se pôr.",
    "A luz dourada do pôr do sol reflete nos prédios da cidade.",
    "A música suave ao fundo cria o ambiente perfeito para concentração.",
    "Cada pequeno passo aproxima você de um grande resultado.",
    "Respire fundo, conte até cinco e deixe a ansiedade escoar.",
    "O som das ondas lembra que tudo na vida é movimento.",
    "Compartilhe um sorriso hoje; ele pode mudar o dia de alguém.",
    "Abra a janela e deixe o vento trazer novas ideias.",
    "Organize sua mesa; um espaço limpo clareia a mente.",
    "Aprender algo novo expande horizontes e renova a criatividade.",
    "A noite estrelada convida a sonhar acordado e planejar o amanhã.",
    "Valorize cada encontro; toda pessoa traz uma história única.",
    "Uma pausa de cinco minutos pode render horas de produtividade.",
    "Ler um bom livro é viajar sem sair do lugar.",
    "A gratidão diária transforma desafios em oportunidades.",
    "Movimentar o corpo libera energia e melhora o humor.",
    "Faça hoje algo que seu futuro vai agradecer.",
    "O silêncio também é música quando a mente está em paz.",
    "Plantar uma semente de bondade gera florestas de empatia.",
    "Pequenas vitórias merecem grande celebração interior.",
    "Confie no processo; a jornada molda o destino.",
    "Cada amanhecer traz a chance de recomeçar melhor.",
]

# Load model
print("Loading model...")
checkpoint = torch.load(checkpoint_path, map_location=device)

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

# Process all audio-text pairs
results = []
print(f"\nProcessing {len(texts)} audio-text pairs...")
print("-" * 80)

for i in range(len(texts)):
    audio_filename = f"audio_{i+1:02d}.mp3"
    audio_path = os.path.join(audio_folder, audio_filename)
    text = texts[i]
    
    print(f"\n[{i+1}/25] Processing: {audio_filename}")
    print(f"Text: {text[:60]}...")
    
    try:
        # Load and process audio
        audio_array, _ = librosa.load(audio_path, sr=16000)
        audio_features = feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")
        audio_input = audio_features["input_features" if "input_features" in audio_features else "input_values"].to(device)
        audio_mask = audio_features.get("attention_mask", None)
        if audio_mask is not None:
            audio_mask = audio_mask.to(device)
        
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
            cosine_sim = (text_emb * audio_emb).sum(dim=1).item()
            similarity_norm = (cosine_sim + 1) / 2
        
        # Get alignment scores
        avg_alignment = None
        if hasattr(model, 'last_pos_alignment_scores') and model.last_pos_alignment_scores is not None:
            alignment_scores_raw = model.last_pos_alignment_scores.squeeze(0)
            alignment_scores = torch.sigmoid(alignment_scores_raw).cpu().numpy()
            attention_mask = text_encoding["attention_mask"].squeeze(0).cpu().numpy()
            valid_scores = alignment_scores[attention_mask == 1]
            avg_alignment = valid_scores.mean()
        
        # Store result
        result = {
            "audio": audio_filename,
            "text": text,
            "similarity": float(similarity_norm),
            "alignment": float(avg_alignment) if avg_alignment is not None else None
        }
        results.append(result)
        
        print(f"Similarity: {similarity_norm:.4f}")
        if avg_alignment is not None:
            print(f"Alignment: {avg_alignment:.4f}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        results.append({
            "audio": audio_filename,
            "text": text,
            "error": str(e)
        })

# Save results
output_file = "audio_text_similarities.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nResults saved to: {output_file}")

# Print summary
successful = [r for r in results if "error" not in r]
if successful:
    avg_sim = sum(r["similarity"] for r in successful) / len(successful)
    avg_align = sum(r["alignment"] for r in successful if r["alignment"]) / len([r for r in successful if r["alignment"]])
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"Processed: {len(results)} pairs")
    print(f"Successful: {len(successful)}")
    print(f"Average similarity: {avg_sim:.4f}")
    print(f"Average alignment: {avg_align:.4f}")
    
    # Show best and worst matches
    sorted_results = sorted(successful, key=lambda x: x["similarity"], reverse=True)
    print("\nTop 5 matches:")
    for r in sorted_results[:5]:
        print(f"  {r['similarity']:.4f} - {r['audio']} - {r['text'][:50]}...")
    
    print("\nBottom 5 matches:")
    for r in sorted_results[-5:]:
        print(f"  {r['similarity']:.4f} - {r['audio']} - {r['text'][:50]}...")