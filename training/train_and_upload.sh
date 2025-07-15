#!/bin/bash
set -e  # Exit on any error

echo "=== Starting training ==="

uv run ./run_embedding_trainer_unfreeze.sh


echo "=== Pushing logs to GitHub ==="
git add .
git commit -m "Training completed: $(date)"
git push

echo "=== Uploading to Hugging Face Hub ==="
huggingface-cli upload yuriyvnv/3_alignment_MHGLU_twoWay_loss 3_alignment_MHGLU_twoWay_loss .

echo "=== All done! Shutting down instance ==="
vastai stop instance 22824684 --api-key 8ae7bf55c0e0d706ec35e022dfcc991547b70da038b169380ef994c802f32b43

