#!/bin/bash
set -e  # Exit on any error

echo "=== Starting training ==="
./run_embedding_trainer_unfreeze.sh

echo "=== Pushing logs to GitHub ==="
git add .
git commit -m "Training completed: $(date)"
git push

echo "=== Uploading to Hugging Face Hub ==="
huggingface-cli upload yuriyvnv/3_layers_wt_alignment_correct_loss 3_layers_wt_alignment_correct_loss .

# Source environment variables from the speech_transcript_embeddings folder
if [ -f "speech_transcript_embeddings/.env" ]; then
    source speech_transcript_embeddings/.env
    echo "Loaded environment variables"
else
    echo "Warning: .env file not found in speech_transcript_embeddings folder"
fi



echo "=== All done! Ready to shutdown ==="
echo "Run 'vastai stop instance <INSTANCE_ID> --api-key <API_KEY>' to terminate the instance"