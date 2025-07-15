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
import datetime
log_filename = f"training_log_.txt"

import logging
import sys
from datetime import datetime
# Create logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
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
import jiwer
from pathlib import Path
import glob
import re
from tqdm import tqdm

load_dotenv()
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
MODEL_NAME = "whisper-large-v3-mixed-pt"
os.environ["WANDB_PROJECT"] = MODEL_NAME
HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN

# Load and preprocess dataset
logger.info("Loading mixed synthetic-CommonVoice Portuguese dataset...")
dataset = load_dataset("yuriyvnv/synthetic_transcript_pt", "mixed_cv_synthetic",token=HF_TOKEN,)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]

logger.info(f"✅ Mixed dataset loaded:")
logger.info(f"   🤖 Train (Synthetic): {len(train_dataset):,} samples")
logger.info(f"   🎤 Validation (Real CV): {len(val_dataset):,} samples") 
logger.info(f"   🎤 Test (Real CV): {len(test_dataset):,} samples")

model_pretrained = "openai/whisper-large-v3"
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_pretrained, token=HF_TOKEN)
tokenizer = WhisperTokenizer.from_pretrained(model_pretrained, language="pt", task="transcribe", token=HF_TOKEN)
processor = WhisperProcessor.from_pretrained(model_pretrained, language="pt", task="transcribe", token=HF_TOKEN)

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

logger.info("🔧 PRE-PROCESSING DATASETS...")
train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, desc="Processing train dataset")
val_dataset = val_dataset.map(prepare_dataset, remove_columns=val_dataset.column_names,  desc="Processing val dataset")
test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names, desc="Processing test dataset")
train_dataset = train_dataset.shuffle(seed=42)
val_dataset = val_dataset.shuffle(seed=42)
test_dataset = test_dataset.shuffle(seed=42)
logger.info("✅ Preprocessing complete!")

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
logger.info("Loading Whisper model...")
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
    warmup_steps=80,
    max_steps=810,
    bf16=True,
    dataloader_num_workers=16,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    
    # Evaluation settings
    eval_strategy="steps",
    eval_steps=50,
    predict_with_generate=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # Save settings
    save_strategy="steps",
    save_steps=50,
    save_total_limit=None,
    load_best_model_at_end=True,
    
    logging_steps=25,
    report_to=["wandb"],
    optim="adamw_torch_fused",
    push_to_hub=True,
    run_name=f"{MODEL_NAME}-synthetic-to-real",
)

logger.info("Setting up trainer...")
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
trainer.save_model(checkpoint_folder + "/final_model")
logger.info("✅ Training completed! Now evaluating all checkpoints...")

# ==========================================
# POST-TRAINING CHECKPOINT EVALUATION
# ==========================================

def get_all_checkpoints(checkpoint_folder):
    """Get all checkpoint folders sorted by step number"""
    checkpoint_pattern = os.path.join(checkpoint_folder, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    # Extract step numbers and sort
    checkpoints = []
    for checkpoint_dir in checkpoint_dirs:
        match = re.search(r'checkpoint-(\d+)', checkpoint_dir)
        if match:
            step_num = int(match.group(1))
            checkpoints.append((step_num, checkpoint_dir))
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def evaluate_checkpoint_on_validation(checkpoint_path, val_dataset, processor, data_collator, max_samples=500):
    """Evaluate a single checkpoint on validation set - FIXED VERSION"""
    logger.info(f"📊 Evaluating checkpoint: {checkpoint_path}")
    
    try:
        # Load model from checkpoint
        model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path,low_cpu_mem_usage=True, attn_implementation="sdpa")
        model.eval()
        model.cuda()
        
        # Use subset of validation data to avoid memory issues
        val_subset = val_dataset.select(range(min(max_samples, len(val_dataset))))
        
        # DIRECT LOSS CALCULATION (no trainer needed)
        logger.info("Computing evaluation loss...")
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i in tqdm(range(0, len(val_subset), 512), desc="Computing loss", unit="batch"): 
                batch_samples = [val_subset[j] for j in range(i, min(i+512, len(val_subset)))]
                
                # Prepare batch using data collator
                batch = data_collator(batch_samples)
                
                # Move to GPU
                input_features = batch["input_features"].cuda()
                labels = batch["labels"].cuda()
                
                # Forward pass
                outputs = model(input_features=input_features, labels=labels)
                loss = outputs.loss
                
                total_loss += loss.item()
                num_batches += 1
                
                # Clear memory
                del batch, input_features, labels, outputs
                torch.cuda.empty_cache()
        
        eval_loss = total_loss / num_batches
        
        # DIRECT WER CALCULATION
        logger.info("   Computing WER with text generation...")
        predictions = []
        references = []
        
        # Generate text for WER calculation (small batches to avoid OOM)
        with torch.no_grad():
            for i in tqdm(range(0, len(val_subset), 512), desc="Computing WER", unit="batch"):  # Process 10 samples at a time
                batch = [val_subset[j] for j in range(i, min(i+512, len(val_subset)))]
                
                # Prepare inputs
                input_features = [sample["input_features"] for sample in batch]
                input_batch = processor.feature_extractor.pad(
                    [{"input_features": f} for f in input_features], 
                    return_tensors="pt"
                ).to(model.device)
                
                # Generate with strict memory limits
                generated_ids = model.generate(
                    input_batch["input_features"],
                    max_length=448,          
                )
                
                # Decode predictions
                pred_texts = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Get references
                ref_texts = []
                for sample in batch:
                    # Decode reference labels
                    labels = sample["labels"]
                    # Remove -100 tokens
                    labels = [l for l in labels if l != -100]
                    ref_text = processor.tokenizer.decode(labels, skip_special_tokens=True)
                    ref_texts.append(ref_text)
                
                predictions.extend(pred_texts)
                references.extend(ref_texts)
                
                # Clear memory
                del input_batch, generated_ids
                torch.cuda.empty_cache()
        
        # Calculate WER using jiwer
        wer = jiwer.wer(references, predictions) * 100
        
        # Clear model from memory
        del model
        torch.cuda.empty_cache()
        
        return {
            'eval_loss': eval_loss,
            'wer': wer,
            'num_samples': len(val_subset)
        }
        
    except Exception as e:
        logger.error(f"   ❌ Detailed error: {str(e)}")
        # Clear any remaining GPU memory
        torch.cuda.empty_cache()
        raise e

def find_best_checkpoint(checkpoint_folder, val_dataset, processor, data_collator):
    """Evaluate all checkpoints and find the best one - FIXED VERSION"""
    logger.info("🔍 Finding best checkpoint based on validation metrics...")
    
    checkpoints = get_all_checkpoints(checkpoint_folder)
    logger.info(f"Found {len(checkpoints)} checkpoints to evaluate")
    
    results = []
    
    for step_num, checkpoint_path in tqdm(checkpoints, desc="Evaluating checkpoints", unit="checkpoint"):
        try:
            metrics = evaluate_checkpoint_on_validation(checkpoint_path, val_dataset, processor, data_collator)
            
            result = {
                'step': step_num,
                'checkpoint_path': checkpoint_path,
                'eval_loss': metrics['eval_loss'],
                'wer': metrics['wer'],
                'num_samples': metrics['num_samples']
            }
            results.append(result)
            
            logger.info(f"   ✅ Step {step_num}: Loss={metrics['eval_loss']:.4f}, WER={metrics['wer']:.2f}%")
            
        except Exception as e:
            logger.error(f"   ❌ Error evaluating step {step_num}: {e}")
            continue
    
    # CHECK IF ANY RESULTS EXIST
    if not results:
        logger.error("❌ No checkpoints could be evaluated!")
        return None, []
    
    # Save all results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(checkpoint_folder, "checkpoint_evaluation_results.csv"), index=False)
    
    # Find best checkpoint
    best_by_loss = min(results, key=lambda x: x['eval_loss'])
    best_by_wer = min(results, key=lambda x: x['wer'])
    
    logger.info(f"\n📊 CHECKPOINT EVALUATION RESULTS:")
    logger.info(f"   Best by Loss: Step {best_by_loss['step']} (Loss={best_by_loss['eval_loss']:.4f}, WER={best_by_loss['wer']:.2f}%)")
    logger.info(f"   Best by WER:  Step {best_by_wer['step']} (Loss={best_by_wer['eval_loss']:.4f}, WER={best_by_wer['wer']:.2f}%)")
    
    return best_by_loss, results

def evaluate_final_model_on_test(best_checkpoint_path, test_dataset, processor, data_collator):
    """Evaluate the best model on test set - FIXED VERSION"""
    logger.info(f"🎯 Final evaluation on test set using: {best_checkpoint_path}")
    
    # Load best model
    model = WhisperForConditionalGeneration.from_pretrained(best_checkpoint_path,low_cpu_mem_usage=True, attn_implementation="sdpa")
    model.eval()
    model.cuda()
    
    # DIRECT LOSS CALCULATION (no trainer)
    logger.info("   Computing test loss...")
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset), 512), desc="Computing test loss", unit="batch"):  # Process 75 samples at a time
            batch_samples = [test_dataset[j] for j in range(i, min(i+512, len(test_dataset)))]
            
            # Prepare batch using data collator
            batch = data_collator(batch_samples)
            
            # Move to GPU
            input_features = batch["input_features"].cuda()
            labels = batch["labels"].cuda()
            
            # Forward pass
            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            num_batches += 1
            
            # Clear memory
            del batch, input_features, labels, outputs
            torch.cuda.empty_cache()
    
    test_loss = total_loss / num_batches
    
    # Calculate WER on subset of test set (to avoid memory issues)
    test_subset = test_dataset.select(range(min(1000, len(test_dataset))))
    
    logger.info(f"   Computing WER on {len(test_subset)} test samples...")
    predictions = []
    references = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_subset), 512), desc="Computing test WER", unit="batch"):
            batch = [test_subset[j] for j in range(i, min(i+512, len(test_subset)))]
            
            input_features = [sample["input_features"] for sample in batch]
            input_batch = processor.feature_extractor.pad(
                [{"input_features": f} for f in input_features], 
                return_tensors="pt"
            ).to(model.device)
            
            generated_ids = model.generate(
                input_batch["input_features"],
                max_length=448,

            )
            
            pred_texts = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            ref_texts = []
            for sample in batch:
                labels = [l for l in sample["labels"] if l != -100]
                ref_text = processor.tokenizer.decode(labels, skip_special_tokens=True)
                ref_texts.append(ref_text)
            
            predictions.extend(pred_texts)
            references.extend(ref_texts)
            
            del input_batch, generated_ids
            torch.cuda.empty_cache()
    
    test_wer = jiwer.wer(references, predictions) * 100
    
    # Clear model from memory
    del model
    torch.cuda.empty_cache()
    
    return {
        'test_loss': test_loss,
        'test_wer': test_wer,
        'test_samples_for_wer': len(test_subset),
        'test_samples_total': len(test_dataset)
    }

# ==========================================
# EXECUTE CHECKPOINT EVALUATION
# ==========================================

if __name__ == "__main__":
    try:
        # Find best checkpoint
        best_checkpoint, all_results = find_best_checkpoint(checkpoint_folder, val_dataset, processor, data_collator)
        
        if best_checkpoint is None:
            logger.error("❌ No valid checkpoints found. Exiting.")
            exit(1)
        
        # Evaluate best model on test set
        final_results = evaluate_final_model_on_test(best_checkpoint['checkpoint_path'], test_dataset, processor, data_collator)
        
        # Combine results
        complete_results = {
            "model_name": MODEL_NAME,
            "dataset": "yuriyvnv/synthetic_transcript_pt",
            "training_paradigm": "synthetic_train_real_eval",
            "best_checkpoint": {
                "step": best_checkpoint['step'],
                "path": best_checkpoint['checkpoint_path'],
                "validation_loss": best_checkpoint['eval_loss'],
                "validation_wer": best_checkpoint['wer']
            },
            "final_test_results": final_results,
            "all_checkpoint_results": all_results
        }
        
        # Save comprehensive results
        with open(os.path.join(checkpoint_folder, "final_evaluation_results.json"), "w") as f:
            json.dump(complete_results, f, indent=2)
        
        logger.info(f"\n🎉 FINAL RESULTS:")
        logger.info(f"   Best checkpoint: Step {best_checkpoint['step']}")
        logger.info(f"   Validation Loss: {best_checkpoint['eval_loss']:.4f}")
        logger.info(f"   Validation WER: {best_checkpoint['wer']:.2f}%")
        logger.info(f"   Test Loss: {final_results['test_loss']:.4f}")
        logger.info(f"   Results saved to: {checkpoint_folder}/final_evaluation_results.json")
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                "best_step": best_checkpoint['step'],
                "best_validation_loss": best_checkpoint['eval_loss'],
                "best_validation_wer": best_checkpoint['wer'],
                "final_test_loss": final_results['test_loss'],
                "final_test_wer": final_results['test_wer']
            })
        
        logger.info(f"\n🔬 RESEARCH INSIGHT:")
        logger.info(f"   This methodology shows the optimal stopping point for synthetic→real transfer!")
        logger.info(f"   Best checkpoint was at step {best_checkpoint['step']} with {best_checkpoint['wer']:.2f}% validation WER")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)