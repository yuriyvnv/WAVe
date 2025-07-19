import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(torch.cuda.is_available())
torch.cuda.empty_cache()
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from transformers.utils import is_torch_sdpa_available
print(is_torch_sdpa_available())
MODEL_NAME = "whisper-large-v3-cv-fully-synthetic-pt"

import json
from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, Seq2SeqTrainer
from transformers import WhisperForConditionalGeneration
from dataclasses import dataclass
from typing import Any
from transformers import Seq2SeqTrainingArguments
from dotenv import load_dotenv
import wandb
import jiwer
from tqdm import tqdm
import logging
from datetime import datetime

# Quick logging setup
log_file = f"training_log_{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
def log_print(message):
    print(message)  # Still shows in terminal
    logging.info(message) 
load_dotenv()
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
PROJECT_NAME = "whisper-large-v3-training"
os.environ["WANDB_PROJECT"] = PROJECT_NAME
HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN



dataset = load_dataset("yuriyvnv/synthetic_transcript_pt", "mixed_cv_synthetic_all",token=HF_TOKEN)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]


log_print(f"✅ Mixed dataset loaded:")
log_print(f"   🤖 Train (Synthetic): {len(train_dataset):,} samples")
log_print(f"   🎤 Validation (Real CV): {len(val_dataset):,} samples") 
log_print(f"   🎤 Test (Real CV): {len(test_dataset):,} samples")

model_pretrained = "openai/whisper-large-v3"
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_pretrained, token=HF_TOKEN)
tokenizer = WhisperTokenizer.from_pretrained(model_pretrained, language="pt", task="transcribe", token=HF_TOKEN)
processor = WhisperProcessor.from_pretrained(model_pretrained, language="pt", task="transcribe", token=HF_TOKEN)
log_print("🔧 PRE-PROCESSING DATASETS...")

def prepare_dataset(batch):
    audio = batch["audio"]
    transcription = batch["text"]
    if transcription.startswith('"') and transcription.endswith('"'):
        # we can remove trailing quotation marks as they do not affect the transcription
        transcription = transcription[1:-1]
    if transcription[-1] not in [".", "?", "!"]:
        # append a full-stop to sentences that do not end in punctuation
        transcription = transcription + "."
    batch["text"] = transcription
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch
# Check if cached data exists


log_print("🔧 PRE-PROCESSING DATASETS...")
# Your original preprocessing code
train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names,   desc="Processing train dataset")
val_dataset = val_dataset.map(prepare_dataset, remove_columns=val_dataset.column_names,  desc="Processing val dataset")
test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names,  desc="Processing test dataset")
train_dataset = train_dataset.shuffle(seed=42)
val_dataset = val_dataset.shuffle(seed=42)
test_dataset = test_dataset.shuffle(seed=42)
    
log_print("✅ Preprocessing complete!")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = processor.tokenizer.pad(label_features, return_tensors="pt")
        
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

checkpoint_folder = f"/root/speech_transcript_embeddings/training_ASR/trained_models/{MODEL_NAME}"

# Load model
log_print("Loading Whisper model...")
model = WhisperForConditionalGeneration.from_pretrained(model_pretrained,low_cpu_mem_usage=True,attn_implementation="sdpa")
model.generation_config.language = "pt"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.use_cache = False
model.to("cuda")

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Training arguments (your exact settings)
training_args = Seq2SeqTrainingArguments(
    output_dir=checkpoint_folder,
    gradient_checkpointing=True,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=8,
    learning_rate=1e-5,
    num_train_epochs=10,
    warmup_ratio=0.1,
    bf16=True,
    dataloader_num_workers=32,

    
    # Evaluation settings
    eval_strategy="steps",
    eval_steps=50,
    predict_with_generate=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # Save settings
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    load_best_model_at_end=True,
    
    logging_steps=25,
    report_to=["wandb"],
    optim="adamw_torch_fused",
    push_to_hub=True,
    run_name=f"{MODEL_NAME}",
)

log_print("Setting up trainer...")
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=processor,
)

# Train the model
trainer.train()

# Save final model
trainer.save_model(checkpoint_folder)
trainer.push_to_hub(checkpoint_folder)
log_print("✅ Training completed! Now evaluating all checkpoints...")

# ==========================================
# POST-TRAINING CHECKPOINT EVALUATION TEST
# ==========================================
# def evaluate_final_model_on_validation(best_checkpoint_path, validation_dataset, processor, data_collator):
#     """Evaluate the best model on validation set - FULL DATASET VERSION"""
#     log_print(f"🎯 Final evaluation on validation set using: {best_checkpoint_path}")
    
#     # Load best model
#     model = WhisperForConditionalGeneration.from_pretrained(best_checkpoint_path,low_cpu_mem_usage=True, attn_implementation="sdpa")
#     model.eval()
#     model.cuda()
    
#     # DIRECT LOSS CALCULATION (no trainer)
#     log_print("   Computing validation loss...")
#     total_loss = 0.0
#     num_batches = 0
    
#     with torch.no_grad():
#         for i in tqdm(range(0, len(validation_dataset), 128), desc="Computing validation loss", unit="batch"):
#             batch_samples = [validation_dataset[j] for j in range(i, min(i+128, len(validation_dataset)))]
            
#             # Prepare batch using data collator
#             batch = data_collator(batch_samples)
            
#             # Move to GPU
#             input_features = batch["input_features"].cuda()
#             labels = batch["labels"].cuda()
            
#             # Forward pass
#             outputs = model(input_features=input_features, labels=labels)
#             loss = outputs.loss
            
#             total_loss += loss.item()
#             num_batches += 1
            
#             # Clear memory
#             del batch, input_features, labels, outputs
#             torch.cuda.empty_cache()
    
#     validation_loss = total_loss / num_batches
#     print(validation_loss)
#     # Calculate WER on FULL validation set
#     validation_subset = validation_dataset
    
#     log_print(f"   Computing WER on {len(validation_subset)} validation samples...")
#     predictions = []
#     references = []
    
#     with torch.no_grad():
#         for i in tqdm(range(0, len(validation_subset), 128), desc="Computing validation WER", unit="batch"):
#             batch = [validation_subset[j] for j in range(i, min(i+128, len(validation_subset)))]
            
#             input_features = [sample["input_features"] for sample in batch]
#             input_batch = processor.feature_extractor.pad(
#                 [{"input_features": f} for f in input_features], 
#                 return_tensors="pt"
#             ).to(model.device)
            
#             generated_ids = model.generate(
#                 input_batch["input_features"],
#                 max_length=448,
#             )
            
#             pred_texts = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
#             ref_texts = []
#             for sample in batch:
#                 labels = [l for l in sample["labels"] if l != -100]
#                 ref_text = processor.tokenizer.decode(labels, skip_special_tokens=True)
#                 ref_texts.append(ref_text)
            
#             predictions.extend(pred_texts)
#             references.extend(ref_texts)
            
#             del input_batch, generated_ids
#             torch.cuda.empty_cache()
    
#     validation_wer = jiwer.wer(references, predictions) * 100
#     print(validation_wer)
#     # Clear model from memory
#     del model
#     torch.cuda.empty_cache()
    
#     return {
#         'validation_loss': validation_loss,
#         'validation_wer': validation_wer,
#         'validation_samples_for_wer': len(validation_subset),
#         'validation_samples_total': len(validation_dataset)
#     }



# def evaluate_final_model_on_test(best_checkpoint_path, test_dataset, processor, data_collator):
#     """Evaluate the best model on test set - FIXED VERSION"""
#     log_print(f"🎯 Final evaluation on test set using: {best_checkpoint_path}")
    
#     # Load best model
#     model = WhisperForConditionalGeneration.from_pretrained(best_checkpoint_path,low_cpu_mem_usage=True, attn_implementation="sdpa",)
#     model.eval()
#     model.cuda()
    
#     # DIRECT LOSS CALCULATION (no trainer)
#     log_print("   Computing test loss...")
#     total_loss = 0.0
#     num_batches = 0
    
#     with torch.no_grad():
#         for i in tqdm(range(0, len(test_dataset), 128), desc="Computing test loss", unit="batch"):  # Process 75 samples at a time
#             batch_samples = [test_dataset[j] for j in range(i, min(i+128, len(test_dataset)))]
            
#             # Prepare batch using data collator
#             batch = data_collator(batch_samples)
            
#             # Move to GPU
#             input_features = batch["input_features"].cuda()
#             labels = batch["labels"].cuda()
            
#             # Forward pass
#             outputs = model(input_features=input_features, labels=labels)
#             loss = outputs.loss
            
#             total_loss += loss.item()
#             num_batches += 1
            
#             # Clear memory
#             del batch, input_features, labels, outputs
#             torch.cuda.empty_cache()
    
#     test_loss = total_loss / num_batches
#     print(test_loss)
#     # Calculate WER on subset of test set (to avoid memory issues)
#     test_subset = test_dataset
    
#     log_print(f"   Computing WER on {len(test_subset)} test samples...")
#     predictions = []
#     references = []
    
#     with torch.no_grad():
#         for i in tqdm(range(0, len(test_subset), 128), desc="Computing test WER", unit="batch"):
#             batch = [test_subset[j] for j in range(i, min(i+128, len(test_subset)))]
            
#             input_features = [sample["input_features"] for sample in batch]
#             input_batch = processor.feature_extractor.pad(
#                 [{"input_features": f} for f in input_features], 
#                 return_tensors="pt"
#             ).to(model.device)
            
#             generated_ids = model.generate(
#                 input_batch["input_features"],
#                 max_length=448,

#             )
            
#             pred_texts = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
#             ref_texts = []
#             for sample in batch:
#                 labels = [l for l in sample["labels"] if l != -100]
#                 ref_text = processor.tokenizer.decode(labels, skip_special_tokens=True)
#                 ref_texts.append(ref_text)
            
#             predictions.extend(pred_texts)
#             references.extend(ref_texts)
            
#             del input_batch, generated_ids
#             torch.cuda.empty_cache()
    
#     test_wer = jiwer.wer(references, predictions) * 100
#     print(test_wer)
#     # Clear model from memory
#     del model
#     torch.cuda.empty_cache()
    
#     return {
#         'test_loss': test_loss,
#         'test_wer': test_wer,
#         'test_samples_for_wer': len(test_subset),
#         'test_samples_total': len(test_dataset)
#     }

# ==========================================
# EXECUTE CHECKPOINT EVALUATION
# ==========================================


# if __name__ == "__main__":
#     try:
#         # Use the final model from training
#         final_model_path = checkpoint_folder 
        
#         log_print(f"🎯 Using final trained model from: {final_model_path}")
        
#         # Evaluate final model on validation set
#         validation_results = evaluate_final_model_on_validation(final_model_path, val_dataset, processor, data_collator)
        
#         # Evaluate final model on test set
#         test_results = evaluate_final_model_on_test(final_model_path, test_dataset, processor, data_collator)
        
#         # Combine results
#         complete_results = {
#             "model_name": MODEL_NAME,
#             "dataset": "yuriyvnv/synthetic_transcript_pt",
#             "training_paradigm": "synthetic_train_real_eval",
#             "final_model_path": final_model_path,
#             "final_validation_results": validation_results,
#             "final_test_results": test_results
#         }
        
#         # Save comprehensive results
#         with open(os.path.join(checkpoint_folder, "final_evaluation_results.json"), "w") as f:
#             json.dump(complete_results, f, indent=2)
        
#         log_print(f"\n🎉 FINAL RESULTS:")
#         log_print(f"   Final model path: {final_model_path}")
#         log_print(f"   Validation Loss: {validation_results['validation_loss']:.4f}")
#         log_print(f"   Validation WER: {validation_results['validation_wer']:.2f}%")
#         log_print(f"   Test Loss: {test_results['test_loss']:.4f}")
#         log_print(f"   Test WER: {test_results['test_wer']:.2f}%")
#         log_print(f"   Results saved to: {checkpoint_folder}/final_evaluation_results.json")
        
#         # Log to wandb
#         if wandb.run is not None:
#             wandb.log({
#                 "final_validation_loss": validation_results['validation_loss'],
#                 "final_validation_wer": validation_results['validation_wer'],
#                 "final_test_loss": test_results['test_loss'],
#                 "final_test_wer": test_results['test_wer']
#             })
        
#         log_print(f"\n🔬 EVALUATION COMPLETE:")
#         log_print(f"   Final model evaluated on both validation and test sets!")
        
#     except Exception as e:
#         log_print(f"\n❌ FATAL ERROR: {e}")
#         import traceback
#         traceback.print_exc()
#         exit(1)