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
from datasets import load_dataset, Audio
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
MODEL_NAME = "whisper-tiny"
os.environ["WANDB_PROJECT"] = MODEL_NAME
HF_TOKEN= os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN

# Load CommonVoice 17.0 Portuguese dataset directly
dataset = load_dataset("mozilla-foundation/common_voice_17_0", "pt", token=HF_TOKEN)

# Cast audio column to 16kHz sampling rate
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Prepare datasets
train_dataset = dataset["train"]
val_dataset = dataset["validation"] 
test_dataset = dataset["test"]

model_pretrained = f"openai/{MODEL_NAME}"

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
    
    # Encode target text to label ids
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids

    
    return batch

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

checkpoint_folder = f"/home/yperezhohin/research/src/trained_models/{MODEL_NAME}"

# Load model
model = WhisperForConditionalGeneration.from_pretrained(model_pretrained)
model.generation_config.language = "pt"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.use_cache = False

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)   

training_args = Seq2SeqTrainingArguments(
    dataloader_num_workers=3,
    output_dir=checkpoint_folder,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=10,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=5,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    predict_with_generate=True,
    save_steps=0,
    eval_steps=250,
    logging_steps=10,
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=processor,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
trainer.save_model(checkpoint_folder + "/trainer_save")

# Evaluate on test set
print("Evaluating on test set...")
test_results = trainer.evaluate(eval_dataset=test_dataset)
print(f"Test WER: {test_results['eval_wer']:.2f}%")

# Save test results
with open(checkpoint_folder + "/test_results.json", "w") as f:
    json.dump(test_results, f, indent=2)