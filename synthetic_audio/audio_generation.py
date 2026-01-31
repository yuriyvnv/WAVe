# generate_tts_audio.py
"""

Usage:
  export OPENAI_API_KEY=...
  python generate_tts_audio.py
"""

import asyncio
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import datetime
import hashlib

from datasets import Dataset, DatasetDict, load_dataset, Audio
from openai import AsyncOpenAI
import aiofiles
from tqdm.auto import tqdm
import logging
from dotenv import load_dotenv

load_dotenv()

# ───────────────────────── CONFIG ─────────────────────────
DATASET_REPO = "ANONYMOUS_USER/synthetic_transcript_nl"
TTS_MODEL = "tts-1"  # Options: "tts-1", "tts-1-hd", or "gpt-4o-mini-tts"
AUDIO_DIR = Path("synthetic_audio_files_nl")
OUTPUT_DATASET_DIR = Path("synthetic_cv17_nl_with_audio")
HUB_REPO_AUDIO = "ANONYMOUS_USER/synthetic_transcript_nl"  # Same repo, will update with audio

# OpenAI TTS available voices (9 voices actually supported by API)
TTS_VOICES = ["alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]

# Audio generation settings
AUDIO_FORMAT = "mp3"  # Options: "mp3", "opus", "aac", "flac", "wav", "pcm"
MAX_CONCURRENT = 50  
RETRY_ATTEMPTS = 3
CHUNK_SIZE = 1024 * 8  # For file writing

# Audio pricing (TTS-1: $15/1M characters, TTS-1-HD: $30/1M characters)
TTS_COST_PER_CHAR = 0.000015  # For TTS-1 model

# ───────────────────────── LOGGING SETUP ─────────────────────────
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = Path("tts_logs") / f"tts_generation_{timestamp}.log"
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ───────────────────────── OPENAI CLIENT ─────────────────────────
api_key = os.getenv("OPENAI_API_KEY")
assert api_key, "Set OPENAI_API_KEY environment variable"
openai_client = AsyncOpenAI(api_key=api_key)

# ───────────────────────── TTS TRACKER ─────────────────────────
class TTSTracker:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_characters = 0
        self.total_audio_duration = 0  # Will estimate
        self.start_time = time.time()
        self.voice_usage = {voice: 0 for voice in TTS_VOICES}
        
    def log_request_start(self, text: str, voice: str):
        self.total_requests += 1
        self.total_characters += len(text)
        self.voice_usage[voice] += 1
        
    def log_request_success(self, estimated_duration: float = 0):
        self.successful_requests += 1
        self.total_audio_duration += estimated_duration
        
    def log_request_failure(self):
        self.failed_requests += 1
        
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
        
    @property
    def estimated_cost(self) -> float:
        # Use correct pricing based on model
        if TTS_MODEL == "tts-1-hd":
            return self.total_characters * 0.00003  # $30/1M characters
        else:  # tts-1 or gpt-4o-mini-tts  
            return self.total_characters * 0.000015  # $15/1M characters
        
    @property
    def requests_per_minute(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return (self.total_requests / elapsed) * 60
        
    def get_stats(self) -> Dict:
        elapsed = time.time() - self.start_time
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.success_rate,
            'total_characters': self.total_characters,
            'estimated_cost_usd': self.estimated_cost,
            'requests_per_minute': self.requests_per_minute,
            'elapsed_time_seconds': elapsed,
            'voice_usage': self.voice_usage.copy(),
            'estimated_audio_hours': self.total_audio_duration / 3600
        }

# Global tracker
tts_tracker = TTSTracker()

# ───────────────────────── AUDIO GENERATION ─────────────────────────
def get_audio_filename(text: str, voice: str, model: str) -> str:
    """Generate deterministic filename based on content"""
    # ✅ FIX: Ensure consistent encoding for hash generation
    content_str = f"{text}_{voice}_{model}"
    content_hash = hashlib.md5(content_str.encode('utf-8')).hexdigest()[:12]
    return f"audio_{content_hash}_{voice}.{AUDIO_FORMAT}"

async def generate_audio(text: str, voice: str, audio_dir: Path, semaphore: asyncio.Semaphore) -> Optional[Dict]:
    """Generate audio for a single text using OpenAI TTS"""
    async with semaphore:
        filename = get_audio_filename(text, voice, TTS_MODEL)
        audio_path = audio_dir / filename
        
        # Skip if already exists
        if audio_path.exists():
            file_size = audio_path.stat().st_size
            estimated_duration = len(text) * 0.1  # Rough estimate: 0.1s per character
            return {
                'text': text,  # ✅ STORE TEXT WITH RESULT
                'audio_path': str(audio_path),
                'voice': voice,
                'model': TTS_MODEL,
                'text_length': len(text),
                'file_size_bytes': file_size,
                'estimated_duration': estimated_duration,
                'status': 'cached'
            }
        
        tts_tracker.log_request_start(text, voice)
        
        for attempt in range(RETRY_ATTEMPTS):
            try:
                logger.debug(f"🎤 Generating audio: {voice} | {len(text)} chars | Attempt {attempt+1}")
                
                # Make TTS request
                response = await openai_client.audio.speech.create(
                    model=TTS_MODEL,
                    voice=voice,
                    input=text,
                    response_format=AUDIO_FORMAT
                )
                
                # Save audio file
                audio_content = response.content
                async with aiofiles.open(audio_path, 'wb') as f:
                    await f.write(audio_content)
                
                # Estimate duration (rough approximation)
                estimated_duration = len(text) * 0.1  # Rough estimate
                file_size = len(audio_content)
                
                tts_tracker.log_request_success(estimated_duration)
                
                logger.debug(f"✅ Audio saved: {filename} | {file_size/1024:.1f}KB")
                
                return {
                    'text': text,  # ✅ STORE TEXT WITH RESULT
                    'audio_path': str(audio_path),
                    'voice': voice,
                    'model': TTS_MODEL,
                    'text_length': len(text),
                    'file_size_bytes': file_size,
                    'estimated_duration': estimated_duration,
                    'status': 'generated'
                }
                
            except Exception as e:
                logger.warning(f"❌ TTS attempt {attempt+1} failed for {voice}: {e}")
                if attempt == RETRY_ATTEMPTS - 1:
                    tts_tracker.log_request_failure()
                    logger.error(f"🚨 All attempts failed for text with {voice}: {text[:50]}...")
                    return None
                await asyncio.sleep(2 ** attempt)
        
        return None

async def process_batch(texts: List[str], voices: List[str], audio_dir: Path, 
                       semaphore: asyncio.Semaphore, pbar: tqdm) -> List[Dict]:
    """Process a batch of texts with different voices"""
    tasks = []
    
    for i, text in enumerate(texts):
        voice = voices[i % len(voices)]  # Cycle through voices
        task = generate_audio(text, voice, audio_dir, semaphore)
        tasks.append(task)
    
    results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:  # ✅ Only add non-None results
            results.append(result)
        pbar.update(1)
        
        # Log progress every 50 completions
        if len(results) % 50 == 0 and len(results) > 0:
            stats = tts_tracker.get_stats()
            logger.info(f"🎵 Progress: {stats['successful_requests']}/{stats['total_requests']} "
                       f"({stats['success_rate']:.1f}% success) | "
                       f"${stats['estimated_cost_usd']:.3f} cost | "
                       f"{stats['requests_per_minute']:.1f} req/min")
    
    return results

# ───────────────────────── DATASET CREATION ─────────────────────────
def create_audio_dataset(texts: List[str], audio_results: List[Dict]) -> Dataset:
    """Create HuggingFace dataset with audio column - FIXED VERSION"""
    
    # ✅ FIX: Create mapping from text to audio result (instead of assuming order)
    audio_map = {}
    for result in audio_results:
        if result and 'text' in result:
            audio_map[result['text']] = result
    
    logger.info(f"📊 Audio mapping: {len(audio_map)} audio files mapped to texts")
    
    # Build dataset records with correct matching
    records = []
    matched_count = 0
    missing_count = 0
    
    for text in texts:
        if text in audio_map:
            # ✅ CORRECTLY MATCHED: Use audio that was generated for this specific text
            result = audio_map[text]
            records.append({
                'text': text,
                'audio': result['audio_path'],
                'voice': result['voice'],
                'model': result['model'],
                'text_length': result['text_length'],
                'file_size_bytes': result['file_size_bytes'],
                'estimated_duration': result['estimated_duration'],
                'generation_status': result['status']
            })
            matched_count += 1
        else:
            # No audio generated for this text (failed generation)
            records.append({
                'text': text,
                'audio': None,
                'voice': None,
                'model': None,
                'text_length': len(text),
                'file_size_bytes': 0,
                'estimated_duration': 0,
                'generation_status': 'failed'
            })
            missing_count += 1
    
    logger.info(f"✅ Dataset created: {matched_count} with audio, {missing_count} without audio")
    
    # Create dataset with Audio feature
    dataset = Dataset.from_list(records)
    if matched_count > 0:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=None))  # Auto-detect sampling rate
    
    return dataset

# ───────────────────────── MAIN FUNCTION ─────────────────────────
async def main():
    AUDIO_DIR.mkdir(exist_ok=True, parents=True)
    OUTPUT_DATASET_DIR.mkdir(exist_ok=True, parents=True)
    
    logger.info("🚀 Starting TTS audio generation for synthetic dataset")
    logger.info(f"📁 Audio output: {AUDIO_DIR}")
    logger.info(f"💾 Dataset output: {OUTPUT_DATASET_DIR}")
    logger.info(f"🎤 TTS Model: {TTS_MODEL}")
    logger.info(f"🎵 Audio Format: {AUDIO_FORMAT}")
    logger.info(f"🔊 Voices: {', '.join(TTS_VOICES)}")
    
    # Load synthetic dataset
    logger.info(f"📊 Loading dataset from: {DATASET_REPO}")
    try:
        dataset = load_dataset(DATASET_REPO, split="train")
        texts = dataset['text']
        logger.info(f"✅ Loaded {len(texts):,} transcripts")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return
    
    # Calculate estimated cost based on model
    total_chars = sum(len(text) for text in texts)
    if TTS_MODEL == "tts-1-hd":
        estimated_cost = total_chars * 0.00003  # $30/1M characters
    else:  # tts-1 or gpt-4o-mini-tts
        estimated_cost = total_chars * 0.000015  # $15/1M characters
        
    estimated_hours = (total_chars * 0.1) / 3600  # Rough audio duration estimate
    
    logger.info(f"💰 Estimated cost: ${estimated_cost:.2f} ({total_chars:,} characters)")
    logger.info(f"⏱️  Estimated audio: {estimated_hours:.1f} hours")
    logger.info(f"🎯 Target: {len(texts)} audio files across {len(TTS_VOICES)} voices")
    
    # Confirm before proceeding (skip if audio already exists)
    if AUDIO_DIR.exists() and any(AUDIO_DIR.glob(f"*.{AUDIO_FORMAT}")):
        existing_files = len(list(AUDIO_DIR.glob(f"*.{AUDIO_FORMAT}")))
        logger.info(f"📁 Found {existing_files} existing audio files - will use cached files and rebuild dataset")
        confirm = input(f"\n💡 Rebuild dataset with existing audio files? (y/N): ")
    else:
        confirm = input(f"\n💡 Proceed with generation? Estimated cost: ${estimated_cost:.2f} (y/N): ")
    
    if confirm.lower() != 'y':
        logger.info("❌ Generation cancelled by user")
        return
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    # Process all texts
    logger.info(f"🎵 Starting audio generation with {MAX_CONCURRENT} concurrent requests...")
    
    with tqdm(total=len(texts), desc="Generating audio", unit="files") as pbar:
        # Process in batches to manage memory
        batch_size = 100
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            logger.info(f"📦 Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            batch_results = await process_batch(batch_texts, TTS_VOICES, AUDIO_DIR, semaphore, pbar)
            # ✅ FIX: Filter out None results before adding to all_results
            valid_results = [result for result in batch_results if result is not None]
            all_results.extend(valid_results)
            
            # Small delay between batches to avoid overwhelming API
            await asyncio.sleep(1)
    
    # Final statistics
    final_stats = tts_tracker.get_stats()
    logger.info("=" * 60)
    logger.info("🎉 AUDIO GENERATION COMPLETED!")
    logger.info(f"🎵 Generated: {final_stats['successful_requests']}/{final_stats['total_requests']} files")
    logger.info(f"💰 Total cost: ${final_stats['estimated_cost_usd']:.3f}")
    logger.info(f"⏱️  Total time: {datetime.timedelta(seconds=int(final_stats['elapsed_time_seconds']))}")
    logger.info(f"🔊 Voice distribution: {final_stats['voice_usage']}")
    
    # Create audio dataset
    logger.info("💾 Creating HuggingFace dataset with audio...")
    try:
        audio_dataset = create_audio_dataset(texts, all_results)
        
        # Save locally
        dataset_dict = DatasetDict({"train": audio_dataset})
        dataset_dict.save_to_disk(str(OUTPUT_DATASET_DIR))
        logger.info(f"✅ Audio dataset saved to: {OUTPUT_DATASET_DIR}")
        
        # Push to Hub (update existing repo with audio)
        try:
            logger.info(f"🚀 Updating repository with audio: {HUB_REPO_AUDIO}")
            dataset_dict.push_to_hub(
                HUB_REPO_AUDIO,
                commit_message=f"Add audio files to synthetic Portuguese dataset ({len(all_results)} audio files with {len(TTS_VOICES)} voices)",
                private=False
            )
            logger.info(f"✅ Successfully updated repository: https://huggingface.co/datasets/{HUB_REPO_AUDIO}")
        except Exception as e:
            logger.error(f"❌ Failed to push to Hub: {e}")
            logger.info("💾 Dataset is still saved locally")
            
    except Exception as e:
        logger.error(f"❌ Failed to create audio dataset: {e}")
    
    # Save generation statistics
    stats_file = Path("tts_logs") / f"generation_stats_{timestamp}.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📊 Generation statistics saved to: {stats_file}")
    logger.info("=" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⏹️ Generation interrupted by user")
    except Exception as e:
        logger.error(f"🚨 Fatal error: {e}", exc_info=True)
        raise