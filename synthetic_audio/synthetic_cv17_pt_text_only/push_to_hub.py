# push_to_hub.py
"""
Simple script to push existing synthetic dataset to Hugging Face Hub
Combines all splits into train only
"""

from datasets import DatasetDict, concatenate_datasets
from pathlib import Path
import os 
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN

# Configuration
DATASET_PATH = "/home/yperezhohin/speech_transcript_embeddings/synthetic_audio/synthetic_cv17_pt_text_only"  # Your existing dataset folder
HUB_REPO = "yuriyvnv/synthetic_transcript_pt"

def main():
    print("🚀 Loading dataset from disk...")
    
    # Load the existing dataset
    dataset_path = Path(DATASET_PATH)
    if not dataset_path.exists():
        print(f"❌ Dataset folder not found: {dataset_path}")
        return
    
    try:

        dsd = DatasetDict.load_from_disk(str(dataset_path))
        print(f"✅ Loaded dataset with {len(dsd)} splits:")
        for split, ds in dsd.items():
            print(f"   - {split}: {len(ds):,} sentences")
        
        # Combine all splits into train only
        print("\n🔄 Combining all splits into 'train' split...")
        all_datasets = [ds for ds in dsd.values()]
        combined_dataset = concatenate_datasets(all_datasets)
        
        # Create new DatasetDict with only train split
        train_only_dsd = DatasetDict({
            "train": combined_dataset
        })
        
        print(f"✅ Combined dataset: train split with {len(combined_dataset):,} sentences")
        
        print(f"\n🚀 Pushing to Hub: {HUB_REPO}")
        
        # Push to Hub
        train_only_dsd.push_to_hub(
            HUB_REPO,
            commit_message=f"Add synthetic Portuguese transcripts ({len(combined_dataset):,} sentences, train split only)",
            private=False  # Change to True for private repo
        )
        
        print(f"✅ Successfully pushed to: https://huggingface.co/datasets/{HUB_REPO}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you're logged in:")
        print("   huggingface-cli login")

if __name__ == "__main__":
    main()