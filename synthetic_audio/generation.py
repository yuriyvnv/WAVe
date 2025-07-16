# synthetic_cv17_pt_generator_enhanced_logging.py
"""
Generate a **synthetic Portuguese transcription corpus** whose word-count
distribution mirrors the *train* split of Mozilla Common Voice 17.

ENHANCED WITH COMPREHENSIVE LOGGING:
- Detailed API call tracking (total, successful, failed, retries)
- Real-time progress monitoring with ETAs
- Token usage and cost estimation
- Generation statistics per bucket
- Rate limiting monitoring
- Resource usage tracking
- Dashboard-style progress display

Usage:
  export OPENAI_API_KEY=...
  python synthetic_cv17_pt_generator_enhanced_logging.py
"""

from __future__ import annotations
import asyncio
import json
import math
import os
import re
import datetime
import time
from collections import Counter, deque
from pathlib import Path
from typing import List, Dict, Optional
import psutil

import openai
from openai import AsyncOpenAI
from datasets import Dataset, DatasetDict, load_dataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import logging
from dotenv import load_dotenv

load_dotenv()   


# ───────────────────────── CONFIG ─────────────────────────
LANG_CFG      = "pt"
GPT_MODEL     = "gpt-4o-mini"
BATCH_SIZE    = 25               # sentences per request
ROUND_SIZE    = 10               # concurrent GPT calls before updating ban-list
BAN_TOP_K     = 10              # most frequent tokens to ban
SCALE_FACTOR  = 1.0              # 1.0=replicate CV hours; >1 to augment
DEV_RATIO     = TEST_RATIO = 0.10
OUT_DIR       = Path("synthetic_cv17_pt_text_only")
LOG_DIR       = Path("histogram_logs")
MAX_RETRIES   = 3
LOG_INTERVAL  = 50               # log comparison every N sentences
DASHBOARD_INTERVAL = 10          # dashboard update every N seconds

# Token pricing (estimated for GPT-4o-mini)
INPUT_TOKEN_COST = 0.00015 / 1000   # per token
OUTPUT_TOKEN_COST = 0.0006 / 1000   # per token

# initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
assert api_key, "Set OPENAI_API_KEY env var before running."
openai_client = AsyncOpenAI(api_key=api_key)

# ───────────────────────── ENHANCED LOGGING SETUP ─────────────────────────
LOG_DIR.mkdir(exist_ok=True, parents=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = LOG_DIR / f"generation_{timestamp}.log"
stats_file = LOG_DIR / f"stats_{timestamp}.json"

# Create custom formatter for better readability
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

# Setup logging with both file and colored console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(logfile, encoding='utf-8'),
    ]
)

# Add colored console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter('%(asctime)s [%(levelname)s] %(message)s'))
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger(__name__)

# ───────────────────────── API CALL TRACKER ─────────────────────────
class APITracker:
    def __init__(self):
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.retry_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.start_time = time.time()
        self.call_times = deque(maxlen=100)  # Track last 100 call times
        self.rate_limit_hits = 0
        
    def log_call_start(self):
        self.total_calls += 1
        
    def log_call_success(self, input_tokens: int, output_tokens: int):
        self.successful_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.call_times.append(time.time())
        
    def log_call_failure(self, is_retry: bool = False):
        self.failed_calls += 1
        if is_retry:
            self.retry_count += 1
            
    def log_rate_limit(self):
        self.rate_limit_hits += 1
        
    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_calls) * 100
        
    @property
    def estimated_cost(self) -> float:
        input_cost = self.total_input_tokens * INPUT_TOKEN_COST
        output_cost = self.total_output_tokens * OUTPUT_TOKEN_COST
        return input_cost + output_cost
        
    @property
    def calls_per_minute(self) -> float:
        if len(self.call_times) < 2:
            return 0.0
        time_span = self.call_times[-1] - self.call_times[0]
        if time_span == 0:
            return 0.0
        return (len(self.call_times) / time_span) * 60
        
    def get_stats(self) -> Dict:
        elapsed = time.time() - self.start_time
        return {
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'retry_count': self.retry_count,
            'success_rate': self.success_rate,
            'rate_limit_hits': self.rate_limit_hits,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'estimated_cost_usd': self.estimated_cost,
            'calls_per_minute': self.calls_per_minute,
            'elapsed_time_seconds': elapsed
        }

# Global API tracker
api_tracker = APITracker()

# ───────────────────────── ENHANCED HISTOGRAM TRACKER ─────────────────────────
class EnhancedHistogramTracker:
    def __init__(self, target: Counter[int], scale: float = 1.0):
        self.target = target
        self.scaled = {wc: math.ceil(cnt*scale) for wc,cnt in target.items()}
        self.generated = Counter()
        self.logbook: List[Dict] = []
        self.start_time = time.time()
        self.bucket_stats = {}  # Track per-bucket statistics
        self.total_target = sum(self.scaled.values())
        
    def add(self, sentence: str, bucket: int):
        wc = len(sentence.split())
        self.generated[wc] += 1
        
        # Update bucket stats
        if bucket not in self.bucket_stats:
            self.bucket_stats[bucket] = {
                'target': self.scaled.get(bucket, 0),
                'generated': 0,
                'start_time': time.time()
            }
        self.bucket_stats[bucket]['generated'] += 1

    def get_bucket_progress(self, bucket: int) -> Dict:
        if bucket not in self.bucket_stats:
            return {'progress': 0.0, 'eta_seconds': None}
            
        stats = self.bucket_stats[bucket]
        target = stats['target']
        generated = stats['generated']
        progress = (generated / target * 100) if target > 0 else 100
        
        # Calculate ETA for this bucket
        elapsed = time.time() - stats['start_time']
        if generated > 0 and progress < 100:
            rate = generated / elapsed
            remaining = target - generated
            eta_seconds = remaining / rate if rate > 0 else None
        else:
            eta_seconds = None
            
        return {
            'progress': progress,
            'generated': generated,
            'target': target,
            'eta_seconds': eta_seconds
        }

    def get_overall_progress(self) -> Dict:
        total_generated = sum(self.generated.values())
        progress = (total_generated / self.total_target * 100) if self.total_target > 0 else 0
        
        # Calculate overall ETA
        elapsed = time.time() - self.start_time
        if total_generated > 0 and progress < 100:
            rate = total_generated / elapsed
            remaining = self.total_target - total_generated
            eta_seconds = remaining / rate if rate > 0 else None
        else:
            eta_seconds = None
            
        return {
            'progress': progress,
            'generated': total_generated,
            'target': self.total_target,
            'eta_seconds': eta_seconds,
            'elapsed_seconds': elapsed
        }

    def stats(self) -> Dict[int,Dict[str,float]]:
        s = {}
        for wc, tgt in self.scaled.items():
            gen = self.generated.get(wc,0)
            s[wc] = {
                'target': tgt,
                'generated': gen,
                'pct': gen/tgt*100 if tgt>0 else 0,
                'diff': gen-tgt
            }
        return s

    def bhattacharyya(self) -> float:
        all_wc = set(self.scaled)|set(self.generated)
        total_t = sum(self.scaled.values())
        total_g = sum(self.generated.values())
        if total_g==0: return 0.0
        bc = 0.0
        for wc in sorted(all_wc):
            p = self.scaled.get(wc,0)/total_t
            q = self.generated.get(wc,0)/total_g
            bc += math.sqrt(p*q)
        return bc

    def log_progress(self, total_sentences: int, current_bucket: Optional[int] = None):
        overall = self.get_overall_progress()
        stats = self.stats()
        sim = self.bhattacharyya()
        
        # Format ETA
        eta_str = "N/A"
        if overall['eta_seconds']:
            eta_delta = datetime.timedelta(seconds=int(overall['eta_seconds']))
            eta_str = str(eta_delta)
            
        logger.info(f"🎯 PROGRESS: {overall['progress']:.1f}% complete ({overall['generated']}/{overall['target']}) | ETA: {eta_str} | Similarity: {sim:.3f}")
        
        # Log current bucket progress if provided
        if current_bucket is not None:
            bucket_progress = self.get_bucket_progress(current_bucket)
            bucket_eta_str = "N/A"
            if bucket_progress['eta_seconds']:
                bucket_eta_delta = datetime.timedelta(seconds=int(bucket_progress['eta_seconds']))
                bucket_eta_str = str(bucket_eta_delta)
            logger.info(f"📊 BUCKET {current_bucket}: {bucket_progress['progress']:.1f}% ({bucket_progress['generated']}/{bucket_progress['target']}) | ETA: {bucket_eta_str}")
        
        # Log API stats
        api_stats = api_tracker.get_stats()
        logger.info(f"🔗 API: {api_stats['successful_calls']}/{api_stats['total_calls']} calls ({api_stats['success_rate']:.1f}% success) | ${api_stats['estimated_cost_usd']:.4f} cost | {api_stats['calls_per_minute']:.1f} calls/min")
        
        self.logbook.append({
            'timestamp': time.time(),
            'sentences': total_sentences,
            'overall_progress': overall['progress'],
            'similarity': sim,
            'api_stats': api_stats
        })

    def plot(self, path: Path):
        wc = sorted(self.scaled.keys())
        tgt = [self.scaled[w] for w in wc]
        gen = [self.generated.get(w,0) for w in wc]
        x = np.arange(len(wc))
        
        plt.figure(figsize=(14,8))
        
        # Main histogram
        plt.subplot(2, 1, 1)
        plt.bar(x-0.2,tgt,0.4,label='CV target', alpha=0.7)
        plt.bar(x+0.2,gen,0.4,label='Generated', alpha=0.7)
        plt.xticks(x,wc)
        plt.xlabel('Word Count')
        plt.ylabel('Count')
        plt.title('Word Count Distribution Comparison')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Progress over time
        plt.subplot(2, 1, 2)
        if len(self.logbook) > 1:
            times = [(entry['timestamp'] - self.start_time) / 3600 for entry in self.logbook]  # Convert to hours
            progress = [entry['overall_progress'] for entry in self.logbook]
            plt.plot(times, progress, 'b-', linewidth=2, label='Progress %')
            plt.xlabel('Time (hours)')
            plt.ylabel('Progress (%)')
            plt.title('Generation Progress Over Time')
            plt.grid(alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', dpi=200)
        plt.close()
        logger.info(f"📈 Saved enhanced histogram plot to {path}")

# ───────────────────────── DASHBOARD LOGGER ─────────────────────────
class DashboardLogger:
    def __init__(self, tracker: EnhancedHistogramTracker):
        self.tracker = tracker
        self.last_update = 0
        
    def update(self, current_bucket: Optional[int] = None, force: bool = False):
        now = time.time()
        if not force and now - self.last_update < DASHBOARD_INTERVAL:
            return
            
        self.last_update = now
        
        # Clear screen and show dashboard
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 80)
        print(f"🚀 SYNTHETIC CV17 PORTUGUESE GENERATOR - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Overall progress
        overall = self.tracker.get_overall_progress()
        progress_bar = self._create_progress_bar(overall['progress'])
        eta_str = self._format_eta(overall['eta_seconds'])
        elapsed_str = str(datetime.timedelta(seconds=int(overall['elapsed_seconds'])))
        
        print(f"📊 OVERALL PROGRESS: {progress_bar} {overall['progress']:.1f}%")
        print(f"📈 Generated: {overall['generated']:,} / {overall['target']:,} sentences")
        print(f"⏱️  Elapsed: {elapsed_str} | ETA: {eta_str}")
        print()
        
        # Current bucket progress
        if current_bucket is not None:
            bucket_progress = self.tracker.get_bucket_progress(current_bucket)
            bucket_bar = self._create_progress_bar(bucket_progress['progress'])
            bucket_eta = self._format_eta(bucket_progress['eta_seconds'])
            print(f"🎯 CURRENT BUCKET ({current_bucket} words): {bucket_bar} {bucket_progress['progress']:.1f}%")
            print(f"   Generated: {bucket_progress['generated']} / {bucket_progress['target']} | ETA: {bucket_eta}")
            print()
        
        # API Statistics
        api_stats = api_tracker.get_stats()
        print("🔗 API STATISTICS:")
        print(f"   Total calls: {api_stats['total_calls']} | Success: {api_stats['successful_calls']} ({api_stats['success_rate']:.1f}%)")
        print(f"   Failed: {api_stats['failed_calls']} | Retries: {api_stats['retry_count']} | Rate limits: {api_stats['rate_limit_hits']}")
        print(f"   Tokens: {api_stats['total_input_tokens']:,} in + {api_stats['total_output_tokens']:,} out")
        print(f"   Cost: ${api_stats['estimated_cost_usd']:.4f} | Rate: {api_stats['calls_per_minute']:.1f} calls/min")
        print()
        
        # System resources
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        print(f"💻 SYSTEM: CPU {cpu_percent:.1f}% | RAM {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
        print()
        
        # Bucket breakdown (top 10 by target size)
        print("📋 BUCKET STATUS (Top 10):")
        bucket_items = sorted(self.tracker.scaled.items(), key=lambda x: x[1], reverse=True)[:10]
        for wc, target in bucket_items:
            generated = self.tracker.generated.get(wc, 0)
            pct = (generated / target * 100) if target > 0 else 0
            status = "✅" if pct >= 100 else "🔄" if pct > 0 else "⏳"
            print(f"   {status} {wc:2d} words: {generated:4d}/{target:4d} ({pct:5.1f}%)")
        
        print("=" * 80)
        
    def _create_progress_bar(self, percent: float, width: int = 30) -> str:
        filled = int(width * percent / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"
        
    def _format_eta(self, seconds: Optional[float]) -> str:
        if seconds is None:
            return "N/A"
        return str(datetime.timedelta(seconds=int(seconds)))

# ───────────────────────── ENHANCED GPT CALL ─────────────────────────
async def enhanced_gpt_call(messages: List[Dict]) -> List[str]:
    api_tracker.log_call_start()
    
    for attempt in range(MAX_RETRIES):
        try:
            resp = await openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                temperature=0.9,
                top_p=0.9,
            )
            
            # Extract token usage if available
            usage = resp.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            
            api_tracker.log_call_success(input_tokens, output_tokens)
            
            result = json.loads(resp.choices[0].message.content)
            logger.debug(f"✅ GPT call successful (attempt {attempt+1}): {len(result)} sentences generated")
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"❌ JSON decode error (attempt {attempt+1}): {e}")
            api_tracker.log_call_failure(is_retry=attempt > 0)
        except Exception as e:
            error_msg = str(e).lower()
            if 'rate limit' in error_msg or 'rate_limit' in error_msg:
                api_tracker.log_rate_limit()
                logger.warning(f"⏳ Rate limit hit (attempt {attempt+1}), waiting...")
                await asyncio.sleep(60)  # Wait longer for rate limits
            else:
                logger.warning(f"❌ GPT call failed (attempt {attempt+1}): {e}")
                
            api_tracker.log_call_failure(is_retry=attempt > 0)
            
            if attempt == MAX_RETRIES - 1:
                logger.error(f"🚨 All {MAX_RETRIES} attempts failed for GPT call")
                raise
                
            await asyncio.sleep(2 ** attempt)
    
    return []

# ───────────────────────── PROMPTS ─────────────────────────
SYSTEM_MSG = (
    "Você é um redator profissional de português brasileiro."
    " Escreva frases independentes, naturais, adequadas para leitura em voz alta."
)

PROMPT_BASE = """
TAREFA: Gere exatamente {N} frases independentes, cada uma com EXATAMENTE {W} palavras.

REGRAS OBRIGATÓRIAS:
✅ Cada frase deve ter exatamente {W} palavras (conte cuidadosamente)
✅ Frases completas com pontuação final (. ? !)
✅ Conteúdo natural para leitura em voz alta
✅ Vocabulário cotidiano (evite marcas/nomes próprios)
✅ Sem aspas ou formatação especial
✅ Resposta: array JSON puro de strings

EXEMPLO de formato correto para {W} palavras:
{example_format}

Responda APENAS com o array JSON:
""".strip()

def create_correction_msg(word_count: int, batch_size: int, error_details: Dict) -> str:
    """Create specific feedback based on actual errors encountered"""
    msg = f"⚠️ CORREÇÃO NECESSÁRIA - Gere novamente {batch_size} frases de exatamente {word_count} palavras:\n"
    
    if error_details.get('wrong_word_count', 0) > 0:
        msg += f"❌ {error_details['wrong_word_count']} frases tinham contagem incorreta de palavras\n"
        if error_details.get('word_count_examples'):
            examples = error_details['word_count_examples'][:2]  # Show 2 examples
            for actual_wc, example in examples:
                msg += f"   Exemplo: '{example[:50]}...' = {actual_wc} palavras (deve ser {word_count})\n"
    
    if error_details.get('duplicates', 0) > 0:
        msg += f"❌ {error_details['duplicates']} frases eram duplicatas\n"
    
    if error_details.get('format_errors', 0) > 0:
        msg += f"❌ {error_details['format_errors']} frases tinham formato incorreto\n"
    
    msg += f"✅ REQUISITOS: Exatamente {word_count} palavras cada | JSON array | Sem duplicatas | Pontuação final"
    return msg

# ───────────────────────── STATS SAVER ─────────────────────────
def save_final_stats(tracker: EnhancedHistogramTracker, synthetic: List[str]):
    """Save comprehensive statistics to JSON file"""
    final_stats = {
        'generation_info': {
            'timestamp': datetime.datetime.now().isoformat(),
            'language': LANG_CFG,
            'model': GPT_MODEL,
            'scale_factor': SCALE_FACTOR,
            'total_sentences': len(synthetic)
        },
        'progress': tracker.get_overall_progress(),
        'api_stats': api_tracker.get_stats(),
        'bucket_stats': {
            str(wc): {
                'target': stats['target'],
                'generated': stats['generated'],
                'progress': stats['progress']
            }
            for wc, stats in {
                wc: tracker.get_bucket_progress(wc) 
                for wc in tracker.scaled.keys()
            }.items()
        },
        'histogram_similarity': tracker.bhattacharyya(),
        'logbook': tracker.logbook
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📊 Final statistics saved to {stats_file}")

# ───────────────────────── MAIN ─────────────────────────
async def main():
    OUT_DIR.mkdir(exist_ok=True)
    
    logger.info("🚀 Starting enhanced synthetic CV17 Portuguese generator")
    logger.info(f"📁 Output directory: {OUT_DIR}")
    logger.info(f"📋 Logs directory: {LOG_DIR}")
    logger.info(f"🤖 Model: {GPT_MODEL}")
    logger.info(f"📏 Scale factor: {SCALE_FACTOR}")
    
    # Load base histogram with progress bar
    logger.info("📊 Loading Common Voice 17 Portuguese training data...")
    with tqdm(desc="Loading CV17 dataset", unit="MB", colour="blue") as pbar:
        base_hist_data = load_dataset("mozilla-foundation/common_voice_17_0", LANG_CFG, split="train")
        pbar.update(1)
        pbar.set_description("Processing word counts")
        base_hist = Counter(len(r['sentence'].split()) for r in tqdm(base_hist_data, desc="Counting words", leave=False))
        pbar.set_description("✅ Dataset loaded")
    
    tracker = EnhancedHistogramTracker(base_hist, SCALE_FACTOR)
    dashboard = DashboardLogger(tracker)
    
    logger.info(f"🎯 Target: {tracker.total_target:,} sentences across {len(tracker.scaled)} word-count buckets")
    
    seen = set()
    token_counts = Counter()
    total_generated = 0
    synthetic: List[str] = []

    def ban_block():
        if not token_counts: return ""
        common = [w for w,_ in token_counts.most_common(BAN_TOP_K)]
        return "🚫 Não use: "+json.dumps(common,ensure_ascii=False)+"\n"

    dashboard.update(force=True)
    
    # Main generation loop
    for wc, quota in tqdm(tracker.scaled.items(), desc="Processing buckets"):
        logger.info(f"🎯 Starting bucket {wc} words: target {quota} sentences")
        accepted = 0
        correction = None
        pending = []
        
        while accepted < quota:
            # Schedule new tasks
            while len(pending) < ROUND_SIZE and accepted + len(pending)*BATCH_SIZE < quota:
                prompt = ban_block()
                if correction:
                    prompt += correction + "\n"
                
                # Create example format for current word count
                example_words = ["O", "tempo", "está", "muito", "bom", "hoje", "aqui", "em", "casa", "nossa"][:wc]
                if len(example_words) < wc:
                    example_words.extend([f"palavra{i}" for i in range(len(example_words), wc)])
                example_sentence = " ".join(example_words) + "."
                example_format = f'["{example_sentence}"]'
                
                prompt += PROMPT_BASE.format(N=BATCH_SIZE, W=wc, example_format=example_format)
                msgs = [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user",   "content": prompt}
                ]
                pending.append(asyncio.create_task(enhanced_gpt_call(msgs)))
                correction = None

            # Wait for completions
            if pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                pending = list(pending)  # Convert set back to list
                for task in done:
                    try:
                        batch = task.result()
                    except Exception as e:
                        logger.error(f"❌ Task failed: {e}")
                        continue
                        
                    batch_accepted = 0
                    error_details = {
                        'wrong_word_count': 0,
                        'duplicates': 0,
                        'format_errors': 0,
                        'word_count_examples': []
                    }
                    
                    for sent in batch:
                        if not isinstance(sent, str) or not sent.strip():
                            error_details['format_errors'] += 1
                            continue
                            
                        wc_actual = len(sent.split())
                        norm = sent.lower().strip()
                        
                        if wc_actual != wc:
                            error_details['wrong_word_count'] += 1
                            if len(error_details['word_count_examples']) < 3:
                                error_details['word_count_examples'].append((wc_actual, sent))
                            logger.debug(f"❌ Wrong word count: expected {wc}, got {wc_actual}")
                            continue
                            
                        if norm in seen:
                            error_details['duplicates'] += 1
                            logger.debug(f"❌ Duplicate sentence detected")
                            continue
                        
                        # Accept sentence
                        seen.add(norm)
                        synthetic.append(sent)
                        tracker.add(sent, wc)
                        token_counts.update(sent.split())
                        accepted += 1
                        batch_accepted += 1
                        total_generated += 1
                        
                        if accepted >= quota:
                            break
                    
                    logger.debug(f"✅ Batch processed: {batch_accepted}/{len(batch)} sentences accepted")
                    
                    # Create specific correction if needed
                    total_errors = error_details['wrong_word_count'] + error_details['duplicates'] + error_details['format_errors']
                    if total_errors > 0:
                        correction = create_correction_msg(wc, BATCH_SIZE, error_details)
                        logger.debug(f"⚠️ Setting specific correction message: {total_errors} errors detected")
                    
                    # Update dashboard and logs
                    if total_generated % LOG_INTERVAL == 0:
                        tracker.log_progress(total_generated, wc)
                        
                    dashboard.update(wc)
                    
                    if accepted >= quota:
                        # Cancel remaining tasks
                        for remaining_task in pending:
                            remaining_task.cancel()
                        pending.clear()
                        break
        
        logger.info(f"✅ Completed bucket {wc} words: {accepted}/{quota} sentences")

    # Final processing
    logger.info("🏁 Generation complete! Processing final outputs...")
    dashboard.update(force=True)
    tracker.log_progress(total_generated)
    
    # Save histogram comparison
    tracker.plot(LOG_DIR / "final_histogram.png")
    
    # Save CSV comparison
    syn_hist = Counter(len(s.split()) for s in synthetic)
    csv_path = OUT_DIR / "histogram_comparison.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("wc,cv_train,synthetic\n")
        for wc in sorted(set(tracker.scaled) | set(syn_hist)):
            f.write(f"{wc},{tracker.scaled.get(wc,0)},{syn_hist.get(wc,0)}\n")
    logger.info(f"📊 Histogram comparison saved to {csv_path}")
    
    # Build and save HuggingFace dataset
    logger.info("💾 Building HuggingFace dataset...")
    ds = Dataset.from_dict({"text": synthetic}).shuffle(seed=42)
    n = len(ds)
    n_test = int(n * TEST_RATIO)
    n_dev = int(n * DEV_RATIO)
    
    dsd = DatasetDict({
        "train": ds.select(range(0, n - n_dev - n_test)),
        "validation": ds.select(range(n - n_dev - n_test, n - n_test)),
        "test": ds.select(range(n - n_test, n))
    })
    dsd.save_to_disk(str(OUT_DIR))
    
    # Save final statistics
    save_final_stats(tracker, synthetic)
    
    # Final summary
    final_stats = api_tracker.get_stats()
    elapsed_time = str(datetime.timedelta(seconds=int(final_stats['elapsed_time_seconds'])))
    
    logger.info("=" * 60)
    logger.info("🎉 GENERATION COMPLETED SUCCESSFULLY!")
    logger.info(f"📊 Generated: {len(synthetic):,} sentences")
    logger.info(f"⏱️  Total time: {elapsed_time}")
    logger.info(f"🔗 API calls: {final_stats['total_calls']} ({final_stats['success_rate']:.1f}% success)")
    logger.info(f"💰 Estimated cost: ${final_stats['estimated_cost_usd']:.4f}")
    logger.info(f"📁 Dataset saved to: {OUT_DIR}")
    logger.info(f"📋 Logs saved to: {LOG_DIR}")
    logger.info("=" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⏹️  Generation interrupted by user")
    except Exception as e:
        logger.error(f"🚨 Fatal error: {e}", exc_info=True)
        raise