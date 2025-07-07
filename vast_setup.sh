#!/bin/bash
# Setup script for speech_transcript_embeddings project

set -e  # Exit on error

echo "======================================"
echo "Speech Transcript Embeddings Setup"
echo "======================================"

# 1. Clone Repository
echo -e "\n[1/9] Cloning repository..."
git clone https://github.com/yuriyvnv/speech_transcript_embeddings.git
cd speech_transcript_embeddings

# 2. System Update and Python Installation
echo -e "\n[2/9] Installing Python 3.11..."
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-dev python3.11-distutils python3.11-venv

# 3. Install pip for Python 3.11
echo -e "\n[3/9] Installing pip for Python 3.11..."
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.11 get-pip.py
rm get-pip.py

# 4. Install UV Package Manager
echo -e "\n[4/9] Installing UV package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"

# 5. Install Python Packages
echo -e "\n[5/9] Installing Python packages..."
uv pip install --system \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    transformers==4.50.2 \
    datasets>=3.6.0 \
    librosa==0.10.1 \
    matplotlib>=3.10.3 \
    numpy==2.0 \
    python-dotenv \
    setuptools==77.0.1 \
    soundfile==0.12.1 \
    sox>=1.5.0 \
    tqdm>=4.67.1

# 6. Install Screen
echo -e "\n[6/9] Installing screen..."
sudo apt update
sudo apt install -y screen

# 7. Git Configuration
echo -e "\n[7/9] Configuring Git..."
read -p "Enter your Git username: " git_username
read -p "Enter your Git email: " git_email
git config --global user.name "$git_username"
git config --global user.email "$git_email"
git config --global credential.helper store

# 8. HuggingFace Configuration
echo -e "\n[8/9] Setting up HuggingFace..."
echo "Please login to HuggingFace:"
huggingface-cli login

# 9. Vast.ai Configuration (optional)
echo -e "\n[9/9] Vast.ai setup (optional)..."
read -p "Do you want to configure Vast.ai? (y/n): " configure_vast
if [[ $configure_vast == "y" ]]; then
    read -p "Enter your Vast.ai API key: " vast_key
    vast auth --api-key "$vast_key"
fi

echo -e "\n======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Create a .env file with your HF_TOKEN:"
echo "   echo 'HF_TOKEN=your_token_here' > .env"
echo ""
echo "2. Run the training script:"
echo "   ./run_embedding_trainer_unfreeze.sh"
echo ""