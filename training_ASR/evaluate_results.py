from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset, Audio
from tqdm import tqdm
import jiwer
import os
import json
import logging
import pathlib
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
run_name = f"whisper_small_cv_only_nl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
log_dir  = pathlib.Path("logs")
log_dir.mkdir(exist_ok=True)

log_path = log_dir / f"{run_name}.log"
logging.basicConfig(
    level=logging.INFO,                       # DEBUG for very detailed
    format="%(asctime)s | %(levelname)7s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf‑8"),
        logging.StreamHandler(sys.stdout),    # still see output in terminal
    ],
)

logger = logging.getLogger(__name__)
logger.info(f"🔖  Logging to {log_path}")

class _StreamToLogger:
    def __init__(self, level):
        self.level = level
        self.buf = ""

    def write(self, msg):
        if msg.rstrip():                      # skip empty lines
            logging.log(self.level, msg.rstrip())

    def flush(self):                          # dummy; logger handles it
        pass

sys.stdout = _StreamToLogger(logging.INFO)
sys.stderr = _StreamToLogger(logging.ERROR)
from transformers import logging as hf_logging
hf_logging.set_verbosity_info()  

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device.type}")

MODEL_NAME = "yuriyvnv/whisper-small-cv-only-nl"
print(f"Using model: {MODEL_NAME}")
# Load model and processor 🚀
print("Loading model... 🎯")
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME, 
)
processor = WhisperProcessor.from_pretrained(
    MODEL_NAME,
    language="nl", 
    task="transcribe",
)
def prepare_dataset(batch):
    txt = batch["sentence"]

    # remove outer quotation marks
    if txt.startswith('"') and txt.endswith('"'):
        txt = txt[1:-1]

    # add a trailing full‑stop if missing
    if txt[-1] not in [".", "?", "!"]:
        txt += "."

    batch["sentence"] = txt
    return batch

# Then set your language and task


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.config.use_cache = True 
model.to(device)
model.eval()
print(f"Using device: {device} 💻")

# Load Common Voice 17.0 dataset for Portuguese 🇵🇹
print("Loading Common Voice 17.0 dataset... 📊")
dataset = load_dataset("mozilla-foundation/common_voice_17_0", "nl", trust_remote_code=True)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

dataset["validation"] = dataset["validation"].map(
    prepare_dataset,
    desc="apply CV‑17 preprocessing validation",
)
dataset["test"] = dataset["test"].map(
    prepare_dataset,
    desc="apply CV‑17 preprocessing test",
)
def evaluate_split(split_name, split_data):
    """Evaluate on a specific dataset split"""
    print(f"\n🔄 Evaluating {split_name} split ({len(split_data)} samples)...")
    
    predictions = []
    references = []
    num_samples = len(split_data)
    
    with torch.no_grad():
        for sample in tqdm(split_data, desc=f"Processing {split_name}"):
            # Process audio
            inputs = processor(
                sample["audio"]["array"],
                sampling_rate=sample["audio"]["sampling_rate"],
                return_tensors="pt",
                padding="longest",
                return_attention_mask=True
            ).to(device)
            
            # Get reference text (no normalization)
            reference = sample["sentence"]
            references.append(reference)
            
            generated_ids = model.generate(
                inputs.input_features,
                attention_mask=inputs.attention_mask,
                max_new_tokens=444,
                temperature=0.0,
                num_beams=3,
                early_stopping=True,     # stop when all beams emit <|endoftext|>                                                                                                            
            )
            prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(prediction)
            print(reference)
            # Store prediction (no normalization)
            predictions.append(prediction)

            

    
    # Calculate metrics
    wer = jiwer.wer(references, predictions)
    
    return {
        'predictions': predictions,
        'references': references,
        'wer': wer,
        'num_samples': num_samples
    }

# Evaluate on validation set
print("\n" + "="*60)
print("🔍 EVALUATING ON VALIDATION SET")
print("="*60)
val_results = evaluate_split("validation", dataset["validation"])
# Evaluate on test set  
print("\n" + "="*60)
print("🔍 EVALUATING ON TEST SET")
print("="*60)
test_results = evaluate_split("test", dataset["test"])

# Print final results 🎉
print(f"\n{'='*60}")
print(f"📈 FINAL RESULTS")
print(f"{'='*60}")

print(f"\n📊 VALIDATION SET:")
print(f"   Samples: {val_results['num_samples']}")
print(f"   Word Error Rate (WER): {val_results['wer']:.2%}")

print(f"\n📊 TEST SET:")
print(f"   Samples: {test_results['num_samples']}")
print(f"   Word Error Rate (WER): {test_results['wer']:.2%}")

print(f"\n{'='*60}")

# Show some examples from test set
print("\n🔍 Sample predictions from TEST set:")
for i in range(min(3, len(test_results['predictions']))):
    print(f"\n#{i+1}")
    print(f"📝 Reference: {test_results['references'][i]}")
    print(f"🤖 Predicted: {test_results['predictions'][i]}")

print(f"\n{'='*60}")
print("✅ Evaluation complete!")
print(f"{'='*60}")

# Save results to JSON file
print("\n💾 Saving results to JSON...")
results_data = {
    "model_name": MODEL_NAME,
    "validation": {
        "predictions": val_results['predictions'],
        "references": val_results['references'],
        "wer": val_results['wer'],
        "num_samples": val_results['num_samples']
    },
    "test": {
        "predictions": test_results['predictions'],
        "references": test_results['references'],
        "wer": test_results['wer'],
        "num_samples": test_results['num_samples']
    }
}

# Create filename with timestamp
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"whisper_evaluation_results_{timestamp}.json"

with open(filename, 'w', encoding='utf-8') as f:
    json.dump(results_data, f, ensure_ascii=False, indent=2)

print(f"✅ Results saved to: {filename}")
print(f"📁 File contains predictions, references, and WER for both validation and test sets")