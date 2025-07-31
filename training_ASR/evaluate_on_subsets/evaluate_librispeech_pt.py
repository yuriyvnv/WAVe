from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset, Audio
from tqdm import tqdm
import jiwer
import os
import json
import logging
import pathlib
import sys
from datetime import datetime
import torch

# Set up logging
run_name = f"whisper_tiny_multi_eval_pt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
log_dir = pathlib.Path("logs")
log_dir.mkdir(exist_ok=True)

log_path = log_dir / f"{run_name}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)7s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)
logger.info(f"🔖 Logging to {log_path}")

class _StreamToLogger:
    def __init__(self, level):
        self.level = level
        self.buf = ""

    def write(self, msg):
        if msg.rstrip():
            logging.log(self.level, msg.rstrip())

    def flush(self):
        pass

sys.stdout = _StreamToLogger(logging.INFO)
sys.stderr = _StreamToLogger(logging.ERROR)

from transformers import logging as hf_logging
hf_logging.set_verbosity_info()

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device.type}")

# Models to evaluate
MODELS = [
    "ANONYMOUS_USER/whisper-small-high-mixed-pt",
    "ANONYMOUS_USER/whisper-small-cv-only-pt", 
    "ANONYMOUS_USER/whisper-small-mixed-pt",
    "ANONYMOUS_USER/whisper-small-cv-full-synthetic-pt"
]

def prepare_dataset(batch, text_column="sentence"):
    """Apply the same preprocessing as in the original code"""
    txt = batch[text_column]

    # remove outer quotation marks
    if txt.startswith('"') and txt.endswith('"'):
        txt = txt[1:-1]

    # add a trailing full-stop if missing
    if txt[-1] not in [".", "?", "!"]:
        txt += "."

    batch[text_column] = txt
    return batch

def load_datasets():
    """Load multilingual_librispeech datasets for Portuguese"""

    print("Loading Multilingual LibriSpeech dataset for Portuguese... 📚")
    try:
        mls_dataset = load_dataset("facebook/multilingual_librispeech", "portuguese", trust_remote_code=True)
        mls_dataset = mls_dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        # Apply preprocessing to MLS (uses "transcript" column)
        mls_dataset["test"] = mls_dataset["test"].map(
            lambda x: prepare_dataset(x, "transcript"),
            desc="apply preprocessing MLS test",
        )
        print(f"✅ MLS loaded: {len(mls_dataset['test'])} test samples")
    except Exception as e:
        print(f"❌ Error loading MLS: {e}")
        mls_dataset = None

    return  mls_dataset

def evaluate_model_on_dataset(model, processor, dataset, dataset_name, text_column):
    """Evaluate a model on a specific dataset"""
    print(f"\n🔄 Evaluating on {dataset_name} ({len(dataset)} samples)...")
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for sample in tqdm(dataset, desc=f"Processing {dataset_name}"):
            # Process audio
            inputs = processor(
                sample["audio"]["array"],
                sampling_rate=sample["audio"]["sampling_rate"],
                return_tensors="pt",
                padding="longest",
                return_attention_mask=True
            ).to(device)
            
            # Get reference text
            reference = sample[text_column]
            references.append(reference)
            
            # Generate prediction
            generated_ids = model.generate(
                inputs.input_features,
                attention_mask=inputs.attention_mask,
                max_new_tokens=444,
                temperature=0.0,
                num_beams=3,
                early_stopping=True,
            )
            prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            predictions.append(prediction)

    # Calculate WER
    wer = jiwer.wer(references, predictions)
    
    return {
        'predictions': predictions,
        'references': references,
        'wer': wer,
        'num_samples': len(predictions)
    }

def main():
    # Load datasets once
    mls_dataset = load_datasets()
    
    # Results storage
    all_results = {}
    
    # Iterate through each model
    for model_idx, model_name in enumerate(MODELS):
        print(f"\n{'='*80}")
        print(f"🤖 EVALUATING MODEL {model_idx + 1}/{len(MODELS)}: {model_name}")
        print(f"{'='*80}")
        
        # Load model and processor
        print("Loading model... 🎯")
        try:
            model = WhisperForConditionalGeneration.from_pretrained(model_name)
            processor = WhisperProcessor.from_pretrained(model_name, language="pt", task="transcribe")
            
            # Set up model configuration
            model.config.use_cache = True
            model.to(device)
            model.eval()
            print(f"✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model {model_name}: {e}")
            continue
        
        model_results = {}
        
        
        # Evaluate on MLS if available
        if mls_dataset is not None:
            try:
                mls_results = evaluate_model_on_dataset(
                    model, processor, 
                    mls_dataset["test"], 
                    "MLS", 
                    "transcript",
                )
                model_results["mls"] = mls_results
                print(f"📊 MLS WER: {mls_results['wer']:.4f} ({mls_results['wer']:.2%})")
            except Exception as e:
                print(f"❌ Error evaluating on MLS: {e}")
        
        all_results[model_name] = model_results
        
        # Clean up GPU memory
        del model, processor
        torch.cuda.empty_cache()
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"📈 FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    
    for model_name, results in all_results.items():
        print(f"\n🤖 {model_name}:")
        if "fleurs" in results:
            print(f"   FLEURS WER: {results['fleurs']['wer']:.4f} ({results['fleurs']['wer']:.2%})")
        if "mls" in results:
            print(f"   MLS WER: {results['mls']['wer']:.4f} ({results['mls']['wer']:.2%})")
    
    # Save results to JSON
    print(f"\n💾 Saving results to JSON...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"whisper_small_multi_eval_pt_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Results saved to: {filename}")
    print(f"📁 File contains WER and predictions for all models on both datasets")
    print(f"{'='*80}")
    print("✅ Multi-model evaluation complete!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()