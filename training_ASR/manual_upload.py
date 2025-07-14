# manual_upload_mixed_dataset.py
"""
Manual upload script for the mixed dataset that was created locally.
Use this if the automatic upload failed.
"""

import os
from datasets import load_dataset, DatasetDict, Audio
from dotenv import load_dotenv
import logging
import multiprocessing
max_workers = min(multiprocessing.cpu_count() - 2, 8)
print(f"Using {max_workers} workers for data processing")
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ───────────────────────── CONFIG ─────────────────────────
# UPDATE THESE PATHS TO MATCH YOUR LOCAL DATASET
LOCAL_DATASET_PATH = "./mixed_dataset"  # Adjust if needed
NEW_DATASET_NAME = "yuriyvnv/synthetic_transcript_pt"

def load_local_mixed_dataset():
    """Load the locally created mixed dataset"""
    
    logger.info("📂 Loading locally created mixed dataset...")
    
    # If you saved it as a DatasetDict
    try:
        mixed_dataset = DatasetDict.load_from_disk(LOCAL_DATASET_PATH)
        logger.info("✅ Loaded from disk successfully")
        return mixed_dataset
    except:
        logger.info("❌ Could not load from disk, recreating...")
        return None

def recreate_mixed_dataset():
    """Recreate the mixed dataset with aligned column schemas"""
    
    logger.info("🔄 Recreating mixed dataset with aligned schemas...")
    
    # Load synthetic dataset
    synthetic_dataset = load_dataset("yuriyvnv/synthetic_transcript_pt", token=HF_TOKEN)
    synthetic_train = synthetic_dataset["train"]
    
    # Load Common Voice dataset  
    cv_dataset = load_dataset("mozilla-foundation/common_voice_17_0", "pt", token=HF_TOKEN)
    cv_validation = cv_dataset["validation"]
    cv_test = cv_dataset["test"]
    
    # Ensure audio is at 16kHz
    synthetic_train = synthetic_train.cast_column("audio", Audio(sampling_rate=16000))
    cv_validation = cv_validation.cast_column("audio", Audio(sampling_rate=16000))
    cv_test = cv_test.cast_column("audio", Audio(sampling_rate=16000))
    
    # ✅ STANDARDIZE SCHEMAS: Keep only essential + useful columns
    
    def standardize_synthetic(batch):
        """Standardize synthetic dataset columns"""
        return {
            'text': batch['text'],
            'audio': batch['audio'],
            'dataset_source': ['synthetic'] * len(batch['text']),
            # Add placeholder columns for Common Voice metadata
            'age': [''] * len(batch['text']),
            'gender': [''] * len(batch['text']),
            'accent': [''] * len(batch['text']),
            'locale': ['pt'] * len(batch['text']),
            # Keep some synthetic-specific info
            'voice': batch['voice'],
            'model': batch['model'],
            # Add placeholders for CV columns
            'client_id': ['synthetic'] * len(batch['text']),
            'up_votes': [0] * len(batch['text']),
            'down_votes': [0] * len(batch['text']),
        }
    
    def standardize_cv(batch):
        """Standardize Common Voice dataset columns"""
        return {
            'text': batch['sentence'],
            'audio': batch['audio'],
            'dataset_source': ['common_voice'] * len(batch['sentence']),
            # Keep CV metadata
            'age': batch.get('age', [''] * len(batch['sentence'])),
            'gender': batch.get('gender', [''] * len(batch['sentence'])),
            'accent': batch.get('accent', [''] * len(batch['sentence'])),
            'locale': batch.get('locale', ['pt'] * len(batch['sentence'])),
            'client_id': batch.get('client_id', [''] * len(batch['sentence'])),
            'up_votes': batch.get('up_votes', [0] * len(batch['sentence'])),
            'down_votes': batch.get('down_votes', [0] * len(batch['sentence'])),
            # Add placeholders for synthetic columns
            'voice': ['human'] * len(batch['sentence']),
            'model': ['human'] * len(batch['sentence']),
        }
    
    # Apply standardization and remove original columns
    logger.info("🔄 Standardizing synthetic training data...")
    synthetic_train = synthetic_train.map(
        standardize_synthetic, 
        batched=True, 
        remove_columns=synthetic_train.column_names,
        num_proc= 64
    )
    
    logger.info("🔄 Standardizing Common Voice validation data...")
    cv_validation = cv_validation.map(
        standardize_cv, 
        batched=True, 
        remove_columns=cv_validation.column_names,
        num_proc= 64
    )
    
    logger.info("🔄 Standardizing Common Voice test data...")
    cv_test = cv_test.map(
        standardize_cv, 
        batched=True, 
        remove_columns=cv_test.column_names,
        num_proc= 64
    )
    
    # Verify schemas match
    logger.info("🔍 Verifying column schemas...")
    train_columns = set(synthetic_train.column_names)
    val_columns = set(cv_validation.column_names)
    test_columns = set(cv_test.column_names)
    
    if train_columns == val_columns == test_columns:
        logger.info("✅ All schemas match!")
        logger.info(f"📋 Columns: {sorted(train_columns)}")
    else:
        logger.error("❌ Schema mismatch!")
        logger.error(f"Train: {train_columns}")
        logger.error(f"Val: {val_columns}")
        logger.error(f"Test: {test_columns}")
        raise ValueError("Column schemas don't match")
    
    # Create mixed dataset
    mixed_dataset = DatasetDict({
        "train": synthetic_train,
        "validation": cv_validation,
        "test": cv_test
    })
    
    return mixed_dataset

def manual_upload():
    """Manually upload the mixed dataset"""
    
    logger.info("🚀 Starting manual upload process...")
    
    # Try to load local dataset first
    mixed_dataset = load_local_mixed_dataset()
    
    # If loading failed, recreate it
    if mixed_dataset is None:
        mixed_dataset = recreate_mixed_dataset()
    
    # Show dataset info
    logger.info("📊 Dataset summary:")
    total_samples = 0
    for split, dataset in mixed_dataset.items():
        logger.info(f"   {split}: {len(dataset):,} samples")
        total_samples += len(dataset)
    logger.info(f"   Total: {total_samples:,} samples")
    
    # Manual confirmation
    print("\n" + "="*60)
    print("📊 READY TO UPLOAD:")
    print(f"   Dataset: {NEW_DATASET_NAME}")
    print(f"   Train: {len(mixed_dataset['train']):,} samples (synthetic)")
    print(f"   Validation: {len(mixed_dataset['validation']):,} samples (real CV)")
    print(f"   Test: {len(mixed_dataset['test']):,} samples (real CV)")
    print("="*60)
    
    # Get user confirmation
    while True:
        confirm = input("\n🚀 Proceed with upload? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            break
        elif confirm in ['n', 'no']:
            logger.info("❌ Upload cancelled by user")
            return
        else:
            print("Please enter 'y' or 'n'")
    
    # Upload to Hub
    try:
        logger.info(f"🚀 Uploading to {NEW_DATASET_NAME}...")
        
        mixed_dataset.push_to_hub(
            NEW_DATASET_NAME,
            commit_message="Mixed dataset: synthetic training + Common Voice validation/test",
            private=False,
            token=HF_TOKEN
        )
        
        logger.info("✅ Upload completed successfully!")
        logger.info(f"🔗 Dataset URL: https://huggingface.co/datasets/{NEW_DATASET_NAME}")
        
        # Save local copy for future use
        mixed_dataset.save_to_disk("./mixed_dataset_backup")
        logger.info("💾 Local backup saved to: ./mixed_dataset_backup")
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        logger.info("🔧 Troubleshooting tips:")
        logger.info("   1. Check HF_TOKEN is set correctly")
        logger.info("   2. Ensure you have write access to the repo")
        logger.info("   3. Try: huggingface-cli login")
        raise

def check_prerequisites():
    """Check if everything is set up correctly"""
    
    logger.info("🔍 Checking prerequisites...")
    
    # Check HF token
    if not HF_TOKEN:
        logger.error("❌ HF_TOKEN not found in environment")
        logger.info("💡 Set it with: export HF_TOKEN=your_token_here")
        return False
    
    # Check HF CLI login
    try:
        from huggingface_hub import whoami
        user_info = whoami(token=HF_TOKEN)
        logger.info(f"✅ Logged in as: {user_info['name']}")
    except Exception as e:
        logger.warning(f"⚠️ Could not verify HF login: {e}")
        logger.info("💡 Try: huggingface-cli login")
    
    return True

if __name__ == "__main__":
    logger.info("🚀 Manual Dataset Upload Tool")
    
    if not check_prerequisites():
        logger.error("❌ Prerequisites not met")
        exit(1)
    
    try:
        manual_upload()
    except KeyboardInterrupt:
        logger.info("\n⏹️ Upload interrupted by user")
    except Exception as e:
        logger.error(f"🚨 Upload failed: {e}")
        exit(1)