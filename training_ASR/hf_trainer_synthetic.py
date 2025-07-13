import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(torch.cuda.is_available())
torch.cuda.empty_cache()
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

import json
import pandas as pd
import numpy as np
from datasets import load_dataset, Audio, DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, Seq2SeqTrainer
from transformers import WhisperForConditionalGeneration, AutoConfig
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import Seq2SeqTrainingArguments
import evaluate
from dotenv import load_dotenv
import wandb

load_dotenv()
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
MODEL_NAME = "whisper-tiny-mixed-pt"  # ✅ UPDATED: Reflect mixed training
os.environ["WANDB_PROJECT"] = MODEL_NAME
HF_TOKEN= os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN

# ✅ UPDATED: Load your mixed dataset (synthetic train + CV val/test)
print("Loading mixed synthetic-CommonVoice Portuguese dataset...")
dataset = load_dataset("yuriyvnv/synthetic_transcript_pt", token=HF_TOKEN)

# Cast audio column to 16kHz sampling rate
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# ✅ FIXED: Use the actual splits instead of creating artificial ones
train_dataset = dataset["train"]        # Synthetic training data
val_dataset = dataset["validation"]     # Real Common Voice validation
test_dataset = dataset["test"]          # Real Common Voice test

print(f"✅ Mixed dataset loaded:")
print(f"   🤖 Train (Synthetic): {len(train_dataset):,} samples")
print(f"   🎤 Validation (Real CV): {len(val_dataset):,} samples") 
print(f"   🎤 Test (Real CV): {len(test_dataset):,} samples")

# ✅ CHECK: Verify dataset sources
train_sources = set(train_dataset["dataset_source"])
val_sources = set(val_dataset["dataset_source"])
test_sources = set(test_dataset["dataset_source"])

print(f"📊 Dataset composition:")
print(f"   Train sources: {train_sources}")
print(f"   Validation sources: {val_sources}")
print(f"   Test sources: {test_sources}")

model_pretrained = f"openai/whisper-tiny"  # Keep base model name for loading

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_pretrained, token=HF_TOKEN)
tokenizer = WhisperTokenizer.from_pretrained(model_pretrained, language="pt", task="transcribe", token=HF_TOKEN)
processor = WhisperProcessor.from_pretrained(model_pretrained, language="pt", task="transcribe", token=HF_TOKEN)

metric = evaluate.load("wer")

def prepare_dataset(batch):
    # Load and process audio
    audio = batch["audio"]
    
    # Compute input features from audio array
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # ✅ WORKS: Both synthetic and CV data now use "text" column
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids

    return batch

print("Preprocessing datasets...")
# Apply preprocessing
train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(prepare_dataset, remove_columns=val_dataset.column_names)
test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        # Split inputs and labels since they have to be of different lengths and need different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = processor.feature_extractor.pad(input_features, return_tensors="pt")

        labels_batch = processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer}

# ✅ UPDATED: Update checkpoint folder name to reflect mixed training
checkpoint_folder = f"/root/speech_transcript_embeddings/training_ASR/trained_models/{MODEL_NAME}"

# Load model
print("Loading Whisper model...")
model = WhisperForConditionalGeneration.from_pretrained(model_pretrained)
model.generation_config.language = "pt"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.use_cache = False

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)   

# ✅ ADJUSTED: Training arguments for mixed synthetic→real evaluation
training_args = Seq2SeqTrainingArguments(
    output_dir=checkpoint_folder,
    per_device_train_batch_size=128,  # Keep your H100-optimized batch size
    per_device_eval_batch_size=64,
    learning_rate=1e-5,              # Good LR for synthetic data
    warmup_steps=500,                
    num_train_epochs=5,              
    fp16=True,                      
    eval_strategy="steps",
    predict_with_generate=False,      # ✅ CHANGED: Enable for better WER calculation
    save_steps=1000,                 
    eval_steps=250,                  # Frequent evaluation on real data
    logging_steps=25,
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    optim="adamw_torch_fused",
    push_to_hub=True,
    run_name=f"{MODEL_NAME}-synthetic-to-real",  # ✅ UPDATED: Better description
)

print("Setting up trainer...")
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=processor,
    compute_metrics=compute_metrics,
)

# ✅ UPDATED: Log mixed dataset info to wandb
if wandb.run is not None:
    wandb.log({
        "dataset_name": "yuriyvnv/synthetic_transcript_pt",
        "dataset_type": "mixed_synthetic_real",
        "train_samples": len(train_dataset),
        "train_type": "synthetic_openai_tts",
        "val_samples": len(val_dataset),
        "val_type": "real_common_voice",
        "test_samples": len(test_dataset),
        "test_type": "real_common_voice",
        "training_paradigm": "synthetic_train_real_eval"
    })

print("🚀 Starting training: Synthetic data → Real speech evaluation...")
print("🤖 Training on synthetic Portuguese TTS data")
print("🎤 Evaluating on real Common Voice Portuguese speech")

# Train the model
trainer.train()
trainer.save_model(checkpoint_folder + "/trainer_save")

# Evaluate on test set
print("Evaluating on real Common Voice test set...")
test_results = trainer.evaluate(eval_dataset=test_dataset)
print(f"🎯 Test WER (Synthetic→Real): {test_results['eval_wer']:.2f}%")

# ✅ ENHANCED: Save comprehensive results with mixed dataset info
results_summary = {
    "model_name": MODEL_NAME,
    "dataset": "yuriyvnv/synthetic_transcript_pt",
    "dataset_type": "mixed_synthetic_real",
    "training_paradigm": "synthetic_train_real_eval",
    "train_samples": len(train_dataset),
    "train_data_type": "synthetic_openai_tts",
    "val_samples": len(val_dataset),
    "val_data_type": "real_common_voice",
    "test_samples": len(test_dataset),
    "test_data_type": "real_common_voice",
    "test_wer_synthetic_to_real": test_results['eval_wer'],
    "training_args": {
        "epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
    }
}

# Save test results
with open(checkpoint_folder + "/test_results.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print(f"✅ Mixed training completed!")
print(f"🤖 Trained on: {len(train_dataset):,} synthetic samples")
print(f"🎤 Evaluated on: {len(test_dataset):,} real speech samples") 
print(f"📊 Final WER (Synthetic→Real): {test_results['eval_wer']:.2f}%")
print(f"💾 Model saved to: {checkpoint_folder}")
print(f"📋 Results saved to: {checkpoint_folder}/test_results.json")

print(f"\n🔬 RESEARCH INSIGHT:")
print(f"   This WER shows how well synthetic TTS training generalizes to real speech!")
print(f"   Compare this to models trained on real Common Voice data.")