#!/bin/bash
set -e  # Exit on any error

echo "=== Starting training ==="
./run_embedding_trainer_unfreeze.sh

echo "=== Pushing logs to GitHub ==="
git add .
git commit -m "Training completed: $(date)"
git push

echo "=== Uploading to Hugging Face Hub ==="
huggingface-cli upload yuriyvnv/model_3_wo_alignment audio_text_model_optimized_unfreeze_3_layers_wo_alignment_correct_encoder .

echo "=== All done! Ready to shutdown ==="
echo "Run 'vast stop instance <INSTANCE_ID>' to terminate the instance"