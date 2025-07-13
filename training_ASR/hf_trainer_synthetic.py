import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
MODEL_NAME = "whisper-tiny-synthetic-pt"  # ✅ CHANGED: Reflect synthetic training
os.environ["WANDB_PROJECT"] = MODEL_NAME
HF_TOKEN= os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN

# ✅ CHANGED: Load your synthetic Portuguese dataset
print("Loading synthetic Portuguese dataset...")
dataset = load_dataset("yuriyvnv/synthetic_transcript_pt", token=HF_TOKEN)

# Cast audio column to 16kHz sampling rate
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# ✅ CHANGED: Create train/validation/test splits since synthetic dataset only has "train"
print(f"Total samples: {len(dataset['train'])}")

# Create splits: 80% train, 10% validation, 10% test
train_size = int(0.8 * len(dataset["train"]))
val_size = int(0.1 * len(dataset["train"]))
test_size = len(dataset["train"]) - train_size - val_size

print(f"Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")

# Split the dataset
train_dataset = dataset["train"].select(range(0, train_size))
val_dataset = dataset["train"].select(range(train_size, train_size + val_size))
test_dataset = dataset["train"].select(range(train_size + val_size, train_size + val_size + test_size))

print(f"✅ Datasets created:")
print(f"   Train: {len(train_dataset)} samples")
print(f"   Validation: {len(val_dataset)} samples") 
print(f"   Test: {len(test_dataset)} samples")

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
    
    # ✅ CHANGED: Use "text" column instead of "sentence" for synthetic dataset
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

# ✅ CHANGED: Update checkpoint folder name to reflect synthetic training
checkpoint_folder = f"/home/yperezhohin/research/src/trained_models/{MODEL_NAME}"

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

# ✅ OPTIMIZED: Adjusted training arguments for synthetic data
training_args = Seq2SeqTrainingArguments(
    output_dir=checkpoint_folder,
    per_device_train_batch_size=128,  # Reduced batch size for synthetic data
    per_device_eval_batch_size=64,
    learning_rate=1e-5,              # Lower learning rate for synthetic data
    warmup_steps=500,                # Reduced warmup steps
    num_train_epochs=5,              # Fewer epochs might be sufficient for synthetic data
    fp16=False,                       # Enable fp16 for faster training
    eval_strategy="steps",
    predict_with_generate=False,
    save_steps=1000,                  # Save more frequently
    eval_steps=250,                  # Evaluate more frequently
    logging_steps=25,
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    optim="adamw_torch_fused",
    push_to_hub=True,
    run_name=f"{MODEL_NAME}-synthetic-audio",  # ✅ CHANGED: Custom run name
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

# ✅ ADDED: Log dataset info to wandb
if wandb.run is not None:
    wandb.log({
        "dataset_name": "yuriyvnv/synthetic_transcript_pt",
        "total_samples": train_size + val_size + test_size,
        "train_samples": train_size,
        "val_samples": val_size,
        "test_samples": test_size,
        "dataset_type": "synthetic_openai_tts"
    })

print("🚀 Starting training on synthetic Portuguese dataset...")
# Train the model
trainer.train()
trainer.save_model(checkpoint_folder + "/trainer_save")

# Evaluate on test set
print("Evaluating on test set...")
test_results = trainer.evaluate(eval_dataset=test_dataset)
print(f"Test WER: {test_results['eval_wer']:.2f}%")

# ✅ ENHANCED: Save comprehensive results with dataset info
results_summary = {
    "model_name": MODEL_NAME,
    "dataset": "yuriyvnv/synthetic_transcript_pt",
    "dataset_type": "synthetic_openai_tts",
    "total_samples": train_size + val_size + test_size,
    "train_samples": train_size,
    "val_samples": val_size,
    "test_samples": test_size,
    "test_wer": test_results['eval_wer'],
    "training_args": {
        "epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps
    }
}

# Save test results
with open(checkpoint_folder + "/test_results.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print(f"✅ Training completed!")
print(f"📊 Final Test WER: {test_results['eval_wer']:.2f}%")
print(f"💾 Model saved to: {checkpoint_folder}")
print(f"📋 Results saved to: {checkpoint_folder}/test_results.json")