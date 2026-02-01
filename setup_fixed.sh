#!/bin/bash
# Fixed setup script for speech_transcript_embeddings project

set -e  # Exit on error

echo "======================================"
echo "Speech Transcript Embeddings Setup"
echo "======================================"

# 1. Check Python version
echo -e "\n[1/7] Checking Python version..."
python3.11 --version

# 2. Install UV Package Manager (if not already installed)
echo -e "\n[2/7] Installing UV package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
else
    echo "UV already installed"
fi

# 3. Install Python Packages using pip directly
echo -e "\n[3/7] Installing Python packages..."
python3.11 -m pip install --user \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    transformers==4.50.2 \
    datasets>=3.6.0 \
    librosa==0.10.1 \
    matplotlib \
    numpy \
    python-dotenv \
    setuptools==77.0.1 \
    soundfile==0.12.1 \
    sox>=1.5.0 \
    tqdm>=4.67.1 \
    huggingface-hub \
    accelerate \
    sentence-transformers

# 4. Install Screen (if not already installed)
echo -e "\n[4/7] Checking screen..."
if ! command -v screen &> /dev/null; then
    sudo apt update
    sudo apt install -y screen
else
    echo "Screen already installed"
fi

# 5. Git Configuration (optional - skip if already configured)
echo -e "\n[5/7] Git configuration..."
if [ -z "$(git config --global user.name)" ]; then
    read -p "Enter your Git username: " git_username
    read -p "Enter your Git email: " git_email
    git config --global user.name "$git_username"
    git config --global user.email "$git_email"
    git config --global credential.helper store
else
    echo "Git already configured for: $(git config --global user.name)"
fi

# 6. HuggingFace Configuration
echo -e "\n[6/7] Setting up HuggingFace..."
if [ -z "$HF_TOKEN" ]; then
    echo "Please login to HuggingFace:"
    huggingface-cli login
else
    echo "HuggingFace token already configured"
fi

# 7. Create .env file if needed
echo -e "\n[7/7] Creating .env file..."
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "# HuggingFace Token" > .env
    echo "HF_TOKEN=$HF_TOKEN" >> .env
else
    echo ".env file already exists"
fi

echo -e "\n======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Navigate to training directory:"
echo "   cd /workspace/WAVe/training_multimodal/trainer_multimodal"
echo ""
echo "2. Fix the language bug in trainer_unfreeze.py (line 2200):"
echo "   Change 'nl' to 'pt' for Portuguese dataset"
echo ""
echo "3. Run the training script:"
echo "   bash run_embedding_trainer_unfreeze.sh"
echo ""