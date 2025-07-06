#!/bin/bash
set -e  # Exit on any error

echo "=== Starting training ==="
./run_embedding_trainer_unfreeze.sh

echo "=== Pushing logs to GitHub ==="

git add .
git commit -m "Training completed: $(date)"
git push

echo "=== Uploading to Hugging Face Hub ==="
huggingface-cli upload yuriyvnv/model_3_wt_alignment_correct_loss  model_3_wt_alignment_correct_loss .

echo "=== All done! Ready to shutdown ==="
echo "Run 'sudo shutdown -h now' to terminate the instance"