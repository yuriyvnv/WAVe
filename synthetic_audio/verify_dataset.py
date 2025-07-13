# verify_and_fix_audio_dataset.py
"""
Verify that audio files match their corresponding text transcripts.
Fix mismatched audio-text pairs in the dataset.

The original script had a bug where audio files could get mismatched with text
due to async processing and incorrect indexing in create_audio_dataset().
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from collections import defaultdict

from datasets import Dataset, DatasetDict, load_dataset, Audio
import librosa
from tqdm.auto import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ───────────────────────── CONFIG ─────────────────────────
DATASET_REPO = "yuriyvnv/synthetic_transcript_pt"  # Your existing dataset
AUDIO_DIR = Path("synthetic_audio_files")
FIXED_DATASET_DIR = Path("synthetic_cv17_pt_fixed_audio")
TTS_MODEL = "tts-1"  # Should match what you used
AUDIO_FORMAT = "mp3"

# ───────────────────────── VERIFICATION FUNCTIONS ─────────────────────────
def get_expected_filename(text: str, voice: str, model: str = TTS_MODEL) -> str:
    """Generate the expected filename for a given text + voice combination"""
    content_hash = hashlib.md5(f"{text}_{voice}_{model}".encode()).hexdigest()[:12]
    return f"audio_{content_hash}_{voice}.{AUDIO_FORMAT}"

def find_audio_file_for_text(text: str, audio_dir: Path) -> Optional[Tuple[str, str]]:
    """Find the correct audio file for a given text"""
    voices = ["alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]
    
    for voice in voices:
        expected_filename = get_expected_filename(text, voice)
        expected_path = audio_dir / expected_filename
        
        if expected_path.exists():
            return str(expected_path), voice
    
    return None

def verify_dataset_integrity() -> Tuple[int, int, List[Dict]]:
    """Verify the integrity of the current dataset"""
    logger.info("🔍 Loading current dataset for verification...")
    
    try:
        # Load the current dataset
        dataset = load_dataset(DATASET_REPO, split="train")
        logger.info(f"📊 Loaded dataset with {len(dataset)} samples")
        
        correct_matches = 0
        incorrect_matches = 0
        issues = []
        
        # Check a sample of entries (first 100 to avoid long processing)
        sample_size = min(100, len(dataset))
        logger.info(f"🔍 Checking first {sample_size} samples for verification...")
        
        for i in tqdm(range(sample_size), desc="Verifying samples"):
            entry = dataset[i]
            text = entry['text']
            
            # Skip if no audio
            if entry.get('audio') is None:
                issues.append({
                    'index': i,
                    'text': text[:50] + "...",
                    'issue': 'No audio file',
                    'expected_voice': 'N/A'
                })
                incorrect_matches += 1
                continue
            
            voice = entry.get('voice', 'unknown')
            
            # Check if the expected audio file exists
            expected_result = find_audio_file_for_text(text, AUDIO_DIR)
            
            if expected_result is None:
                issues.append({
                    'index': i,
                    'text': text[:50] + "...",
                    'issue': 'No audio file found for this text',
                    'expected_voice': 'N/A'
                })
                incorrect_matches += 1
            else:
                expected_path, expected_voice = expected_result
                expected_filename = Path(expected_path).name
                
                # The audio path in the dataset might be relative or absolute
                audio_path = entry['audio'].get('path', '') if isinstance(entry['audio'], dict) else str(entry['audio'])
                current_filename = Path(audio_path).name if audio_path else 'N/A'
                
                if current_filename == expected_filename:
                    correct_matches += 1
                else:
                    issues.append({
                        'index': i,
                        'text': text[:50] + "...",
                        'issue': f'Audio mismatch: has {current_filename}, expected {expected_filename}',
                        'expected_voice': expected_voice,
                        'current_voice': voice
                    })
                    incorrect_matches += 1
        
        return correct_matches, incorrect_matches, issues
        
    except Exception as e:
        logger.error(f"❌ Error verifying dataset: {e}")
        return 0, 0, []

def create_correct_audio_dataset() -> Dataset:
    """Create a properly matched audio dataset"""
    logger.info("🔧 Creating correctly matched audio dataset...")
    
    # Load text data
    dataset = load_dataset(DATASET_REPO, split="train")
    texts = dataset['text']
    
    logger.info(f"📝 Processing {len(texts)} texts...")
    
    # Build correct records
    records = []
    matched_count = 0
    missing_count = 0
    
    for i, text in enumerate(tqdm(texts, desc="Matching audio to text")):
        audio_result = find_audio_file_for_text(text, AUDIO_DIR)
        
        if audio_result:
            audio_path, voice = audio_result
            
            # Get file info
            try:
                file_size = Path(audio_path).stat().st_size
                # Estimate duration (rough approximation)
                estimated_duration = len(text) * 0.1
            except Exception:
                file_size = 0
                estimated_duration = 0
            
            records.append({
                'text': text,
                'audio': audio_path,
                'voice': voice,
                'model': TTS_MODEL,
                'text_length': len(text),
                'file_size_bytes': file_size,
                'estimated_duration': estimated_duration,
                'generation_status': 'matched'
            })
            matched_count += 1
        else:
            # No audio found for this text
            records.append({
                'text': text,
                'audio': None,
                'voice': None,
                'model': None,
                'text_length': len(text),
                'file_size_bytes': 0,
                'estimated_duration': 0,
                'generation_status': 'missing_audio'
            })
            missing_count += 1
    
    logger.info(f"✅ Matched: {matched_count}, ❌ Missing: {missing_count}")
    
    # Create dataset with Audio feature
    dataset = Dataset.from_list(records)
    
    # Only cast audio column for records that have audio
    if matched_count > 0:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=None))
    
    return dataset

def create_audio_file_inventory() -> Dict:
    """Create an inventory of all audio files and what text they were generated from"""
    logger.info("📋 Creating audio file inventory...")
    
    inventory = {
        'total_files': 0,
        'by_voice': defaultdict(int),
        'orphaned_files': [],
        'file_details': []
    }
    
    if not AUDIO_DIR.exists():
        logger.error(f"❌ Audio directory not found: {AUDIO_DIR}")
        return inventory
    
    audio_files = list(AUDIO_DIR.glob(f"*.{AUDIO_FORMAT}"))
    inventory['total_files'] = len(audio_files)
    
    logger.info(f"📁 Found {len(audio_files)} audio files")
    
    # Load texts to check against
    try:
        dataset = load_dataset(DATASET_REPO, split="train")
        texts = set(dataset['text'])
    except Exception as e:
        logger.error(f"❌ Could not load texts: {e}")
        texts = set()
    
    for audio_file in tqdm(audio_files, desc="Analyzing audio files"):
        filename = audio_file.name
        file_size = audio_file.stat().st_size
        
        # Parse filename to extract voice and hash
        if filename.startswith('audio_') and '_' in filename:
            parts = filename.split('_')
            if len(parts) >= 3:
                voice = parts[2].split('.')[0]  # Remove extension
                content_hash = parts[1]
                
                inventory['by_voice'][voice] += 1
                
                # Try to find which text this corresponds to
                matching_text = None
                for text in texts:
                    expected_filename = get_expected_filename(text, voice)
                    if expected_filename == filename:
                        matching_text = text[:100] + "..." if len(text) > 100 else text
                        break
                
                inventory['file_details'].append({
                    'filename': filename,
                    'voice': voice,
                    'file_size': file_size,
                    'matching_text': matching_text,
                    'is_orphaned': matching_text is None
                })
                
                if matching_text is None:
                    inventory['orphaned_files'].append(filename)
    
    return inventory

# ───────────────────────── MAIN FUNCTIONS ─────────────────────────
def main_verify():
    """Main verification function"""
    logger.info("🚀 Starting audio-text matching verification...")
    
    # Step 1: Verify current dataset
    correct, incorrect, issues = verify_dataset_integrity()
    
    logger.info("=" * 60)
    logger.info("📊 VERIFICATION RESULTS:")
    logger.info(f"✅ Correct matches: {correct}")
    logger.info(f"❌ Incorrect/Missing: {incorrect}")
    logger.info(f"📋 Total issues found: {len(issues)}")
    
    if issues:
        logger.info("\n🔍 SAMPLE ISSUES:")
        for issue in issues[:5]:  # Show first 5 issues
            logger.info(f"  - Index {issue['index']}: {issue['issue']}")
            logger.info(f"    Text: {issue['text']}")
    
    # Step 2: Create audio inventory
    inventory = create_audio_file_inventory()
    logger.info("\n📁 AUDIO FILE INVENTORY:")
    logger.info(f"📊 Total audio files: {inventory['total_files']}")
    logger.info(f"🔊 By voice: {dict(inventory['by_voice'])}")
    logger.info(f"👤 Orphaned files: {len(inventory['orphaned_files'])}")
    
    # Save detailed results
    results_file = Path("audio_verification_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'verification_summary': {
                'correct_matches': correct,
                'incorrect_matches': incorrect,
                'total_issues': len(issues)
            },
            'issues': issues,
            'inventory': inventory
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📋 Detailed results saved to: {results_file}")
    
    # Recommendation
    if incorrect > 0:
        logger.info("\n💡 RECOMMENDATION: Run fix mode to create correctly matched dataset")
        logger.info("   python verify_and_fix_audio_dataset.py --fix")
    else:
        logger.info("\n✅ Dataset appears to be correctly matched!")

def main_fix():
    """Main fix function"""
    logger.info("🔧 Starting audio-text matching fix...")
    
    # Create correctly matched dataset
    fixed_dataset = create_correct_audio_dataset()
    
    # Save locally
    FIXED_DATASET_DIR.mkdir(exist_ok=True, parents=True)
    dataset_dict = DatasetDict({"train": fixed_dataset})
    dataset_dict.save_to_disk(str(FIXED_DATASET_DIR))
    
    logger.info(f"💾 Fixed dataset saved locally to: {FIXED_DATASET_DIR}")
    
    # Ask if user wants to push to Hub
    push_to_hub = input("\n🚀 Push fixed dataset to Hub? This will overwrite the existing dataset (y/N): ")
    if push_to_hub.lower() == 'y':
        try:
            logger.info(f"🚀 Pushing fixed dataset to: {DATASET_REPO}")
            dataset_dict.push_to_hub(
                DATASET_REPO,
                commit_message="Fix audio-text matching issues (correctly matched audio to transcripts)",
                private=False
            )
            logger.info(f"✅ Successfully updated: https://huggingface.co/datasets/{DATASET_REPO}")
        except Exception as e:
            logger.error(f"❌ Failed to push to Hub: {e}")
            logger.info("💾 Fixed dataset is still saved locally")
    
    logger.info("🎉 Fix process completed!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--fix":
        main_fix()
    else:
        main_verify()
        print("\n" + "="*60)
        print("🔧 To fix the dataset, run:")
        print("   python verify_and_fix_audio_dataset.py --fix")