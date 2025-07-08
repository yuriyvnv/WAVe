#!/bin/bash
set -e  # Exit on any error

echo "=== Starting training ==="
uv run ./run_embedding_trainer_unfreeze.sh
echo "=== Pushing logs to GitHub ==="
git add .
git commit -m "Training completed: $(date)"
git push

echo "=== Uploading to Hugging Face Hub ==="
huggingface-cli upload yuriyvnv/3layers_wt_alignment_pooling_6 3layers_wt_alignment_pooling_6 .
 .



echo "=== All done! Shutting down instance ==="
vastai stop instance 22745726 --api-key 8ae7bf55c0e0d706ec35e022dfcc991547b70da038b169380ef994c802f32b43