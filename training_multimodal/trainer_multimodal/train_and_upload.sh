#!/bin/bash
set -e  # Exit on any error

echo "=== Starting training ==="

uv run ./run_embedding_trainer_unfreeze.sh


echo "=== Pushing logs to GitHub ==="
git add .
git commit -m "Training completed  with word alignment for portuguese : $(date)"
git push

echo "=== Uploading to Hugging Face Hub ==="
huggingface-cli upload yuriyvnv/3layers_wt_alignment_PT_100_epochs ./3layers_wt_alignment_PT_100_epochs

echo "=== All done! Shutting down instance ==="
vastai stop instance 30840095  --api-key 4300ecc932fe3c152b84b2210d16cf69f5d87b9b26b0b07cf7eb5cfe03f125ea

