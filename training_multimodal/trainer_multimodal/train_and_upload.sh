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
vastai stop instance 30840095  --api-key ac9eb3f9342873e5bdbc3fdff92bb93c254c3f86d79306110877d1933b452ae5

