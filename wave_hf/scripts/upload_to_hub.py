#!/usr/bin/env python3
"""
Upload WAVe model to HuggingFace Hub.

This script will:
1. Read HF token from environment
2. Copy the professional README to the model directory
3. Upload the model to HuggingFace Hub
4. Provide the public URL

Usage:
    export HF_TOKEN="your_token_here"
    python upload_to_hub.py
"""

import os
import sys
import shutil
from pathlib import Path

# Add to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from huggingface_hub import HfApi, login
from transformers import AutoModel, AutoConfig
from configuration_wave import WAVeConfig
from modeling_wave import WAVeForQualityVerification

def main():
    print("=" * 80)
    print("WAVe Model Upload to HuggingFace Hub")
    print("=" * 80)

    # Check for HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("\n❌ ERROR: HF_TOKEN not found in environment")
        print("\nPlease set your HuggingFace token:")
        print("  export HF_TOKEN='your_token_here'")
        print("\nGet your token from: https://huggingface.co/settings/tokens")
        return 1

    print("\n✓ HF_TOKEN found")

    # Configuration
    model_dir = "./wave-portuguese"
    hub_model_id = "yuriyvnv/WAVe-1B-Multimodal-PT"

    print(f"\nConfiguration:")
    print(f"  Model directory: {model_dir}")
    print(f"  Hub model ID: {hub_model_id}")

    # Verify model directory exists
    if not Path(model_dir).exists():
        print(f"\n❌ ERROR: Model directory not found: {model_dir}")
        print("Please run convert_checkpoint.py first")
        return 1

    print(f"  ✓ Model directory exists")

    # Copy professional README
    print("\n1. Preparing model card...")
    model_card_source = Path("MODEL_CARD.md")
    model_card_dest = Path(model_dir) / "README.md"

    if model_card_source.exists():
        shutil.copy(model_card_source, model_card_dest)
        print(f"   ✓ Copied professional README to {model_card_dest}")
    else:
        print(f"   ⚠ Warning: MODEL_CARD.md not found, using existing README")

    # Login to HuggingFace
    print("\n2. Logging in to HuggingFace...")
    try:
        login(token=hf_token)
        print("   ✓ Successfully logged in")
    except Exception as e:
        print(f"   ❌ ERROR: Failed to login: {e}")
        return 1

    # Register model (for loading)
    print("\n3. Registering model architecture...")
    AutoConfig.register("wave", WAVeConfig)
    AutoModel.register(WAVeConfig, WAVeForQualityVerification)
    print("   ✓ Model architecture registered")

    # Load model
    print("\n4. Loading model for verification...")
    try:
        model = AutoModel.from_pretrained(model_dir)
        print("   ✓ Model loaded successfully")

        # Check model size
        model_size = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"   Model size: {model_size:.1f}M parameters")
    except Exception as e:
        print(f"   ❌ ERROR: Failed to load model: {e}")
        return 1

    # Upload to Hub
    print(f"\n5. Uploading to HuggingFace Hub: {hub_model_id}")
    print("   This may take several minutes (model is ~3.3GB)...")

    try:
        model.push_to_hub(
            hub_model_id,
            use_auth_token=hf_token,
            commit_message="Upload WAVe Portuguese model"
        )
        print("   ✓ Model uploaded successfully!")
    except Exception as e:
        print(f"   ❌ ERROR: Failed to upload model: {e}")
        return 1

    # Upload processor
    print("\n6. Uploading processor...")
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_dir)
        processor.push_to_hub(
            hub_model_id,
            use_auth_token=hf_token
        )
        print("   ✓ Processor uploaded successfully!")
    except Exception as e:
        print(f"   ⚠ Warning: Could not upload processor: {e}")
        print("   You may need to upload it separately")

    # Upload processing_wave.py as a file
    print("\n7. Uploading processing_wave.py...")
    try:
        api = HfApi()
        processing_file = Path(model_dir) / "processing_wave.py"
        if processing_file.exists():
            api.upload_file(
                path_or_fileobj=str(processing_file),
                path_in_repo="processing_wave.py",
                repo_id=hub_model_id,
                token=hf_token
            )
            print("   ✓ processing_wave.py uploaded")
        else:
            print("   ⚠ Warning: processing_wave.py not found")
    except Exception as e:
        print(f"   ⚠ Warning: Could not upload processing_wave.py: {e}")

    # Success!
    print("\n" + "=" * 80)
    print("✅ SUCCESS! Model uploaded to HuggingFace Hub")
    print("=" * 80)

    print(f"\n🔗 Model URL: https://huggingface.co/{hub_model_id}")
    print("\n📝 Next steps:")
    print("  1. Visit the model page to verify it looks correct")
    print(f"  2. Test loading: AutoModel.from_pretrained('{hub_model_id}')")
    print("  3. Update your paper/README with the model link")
    print("  4. Share with the community!")

    print("\n💡 Tip: The model is now public and anyone can use it with:")
    print("  >>> from transformers import AutoModel")
    print(f"  >>> model = AutoModel.from_pretrained('{hub_model_id}')")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
