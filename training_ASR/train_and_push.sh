#!/bin/bash

# train_and_stop.sh
# Runs Whisper training and automatically stops VastAI instance to save costs
# (Training script handles HF Hub push automatically via push_to_hub=True)

# 💡 SETUP: Set your VastAI API key to enable auto-stop:
#    export VASTAI_API_KEY="your_api_key_here"

set -e  # Exit on any error

# ─────────────────────── CONFIG ───────────────────────
TRAINING_SCRIPT="hf_trainer_complete.py"
MODEL_NAME="whisper-small-cv-only-pt"
MODEL_DIR="/root/speech_transcript_embeddings/training_ASR/trained_models/${MODEL_NAME}"
LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"

# VastAI configuration
VASTAI_INSTANCE_ID="23177966"
VASTAI_API_KEY="8ae7bf55c0e0d706ec35e022dfcc991547b70da038b169380ef994c802f32b43"  # Set this environment variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ─────────────────────── FUNCTIONS ───────────────────────
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

check_prerequisites() {
    log_info "🔍 Checking prerequisites..."
    
    # Check if training script exists
    if [ ! -f "$TRAINING_SCRIPT" ]; then
        log_error "Training script not found: $TRAINING_SCRIPT"
        exit 1
    fi
    
    # Check GPU
    if ! nvidia-smi &>/dev/null; then
        log_warning "nvidia-smi not found or no GPU detected"
    else
        log_info "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    fi
    
    log_success "Prerequisites check passed ✅"
}

run_training() {
    log_info "🚀 Starting Whisper training..."
    log_info "Script: $TRAINING_SCRIPT"
    log_info "Model: $MODEL_NAME"
    log_info "Log file: $LOG_FILE"
    log_info "📤 HF Hub push: Automatic (via push_to_hub=True)"
    
    echo "─────────────────────────────────────────" | tee -a "$LOG_FILE"
    
    # Run training with output to both console and log
    if python3 "$TRAINING_SCRIPT" 2>&1 | tee -a "$LOG_FILE"; then
        log_success "Training completed successfully! 🎉"
        log_info "📤 Model automatically pushed to Hub during training"
        git add .
        git commit -m "Training complete"
        git push
        return 0
    else
        log_error "Training failed! ❌"
        return 1
    fi
}

stop_vastai_instance() {
    log_info "🛑 Stopping VastAI instance..."
    
    if [ -z "$VASTAI_API_KEY" ]; then
        log_warning "VASTAI_API_KEY not set, skipping instance stop"
        log_info "💡 To auto-stop instance, set: export VASTAI_API_KEY=your_api_key"
        log_warning "⚠️ Please manually stop instance $VASTAI_INSTANCE_ID to avoid charges!"
        return 0
    fi
    
    if [ -z "$VASTAI_INSTANCE_ID" ]; then
        log_warning "VASTAI_INSTANCE_ID not set, skipping instance stop"
        return 0
    fi
    
    log_info "Instance ID: $VASTAI_INSTANCE_ID"
    
    # Stop the VastAI instance
    if vastai stop instance "$VASTAI_INSTANCE_ID" --api-key "$VASTAI_API_KEY" ; then
        log_success "✅ VastAI instance stopped successfully!"
        log_info "💰 Instance $VASTAI_INSTANCE_ID has been stopped to save costs"
    else
        log_error "❌ Failed to stop VastAI instance"
        log_warning "⚠️ Please manually stop instance $VASTAI_INSTANCE_ID to avoid charges!"
        return 1
    fi
}

show_summary() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    log_success "🎉 TRAINING COMPLETED!"
    echo "═══════════════════════════════════════════════════════════════"
    log_info "📊 Model: $MODEL_NAME"
    log_info "📁 Local path: $MODEL_DIR"
    log_info "🔗 Hub URL: https://huggingface.co/yuriyvnv/$MODEL_NAME"
    log_info "📋 Log file: $LOG_FILE"
    echo "═══════════════════════════════════════════════════════════════"
    
    # Show final model info
    if [ -f "$MODEL_DIR/test_results.json" ]; then
        log_info "📈 Final results:"
        python -c "
import json
try:
    with open('$MODEL_DIR/test_results.json', 'r') as f:
        results = json.load(f)
    print(f\"   🎯 Test WER: {results.get('test_wer', 'N/A'):.2f}%\")
    print(f\"   📊 Train samples: {results.get('train_samples', 'N/A'):,}\")
    print(f\"   🎤 Test samples: {results.get('test_samples', 'N/A'):,}\")
except:
    print('   Results file not found')
"
    fi
}

cleanup() {
    log_info "🧹 Cleaning up temporary files..."
}

# ─────────────────────── MAIN ───────────────────────
main() {
    echo "🚀 Whisper Training and Auto-Stop Script"
    echo "Started at: $(date)"
    echo ""
    
    # Trap cleanup on exit
    trap cleanup EXIT
    
    # Run the pipeline
    check_prerequisites
    
    if run_training; then
        show_summary
        
        # Stop VastAI instance after training
        log_info "🛑 Stopping VastAI instance to save costs..."
        stop_vastai_instance
        
    else
        log_error "Training failed, stopping pipeline"
        
        # Stop instance even on failure to avoid charges
        log_info "🛑 Stopping VastAI instance due to training failure..."
        stop_vastai_instance
        exit 1
    fi
    
    log_success "🎉 All done! Your model is trained and instance stopped!"
}

# ─────────────────────── USAGE ───────────────────────
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h     Show this help message"
    echo "  --no-stop      Run training but don't stop VastAI instance"
    echo ""
    echo "Environment variables:"
    echo "  VASTAI_API_KEY     VastAI API key for auto-stopping instance"
    echo ""
    echo "This script will:"
    echo "  1. Run $TRAINING_SCRIPT (auto-pushes to HF Hub)"
    echo "  2. Stop VastAI instance $VASTAI_INSTANCE_ID (saves money!)"
    echo "  3. Log everything to a timestamped log file"
    echo ""
    echo "💡 Pro tip: Set VASTAI_API_KEY to auto-stop your instance and save costs!"
    exit 0
fi

if [ "$1" = "--no-stop" ]; then
    log_info "🚫 NO-STOP MODE: Will skip stopping VastAI instance"
    check_prerequisites
    
    if run_training; then
        show_summary
        log_info "🚫 Skipping VastAI instance stop (--no-stop mode)"
    else
        log_error "Training failed"
        exit 1
    fi
    
    log_success "🎉 Training completed! Instance still running."
    exit 0
fi

# Run the main pipeline
main