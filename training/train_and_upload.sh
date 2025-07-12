#!/bin/bash
set -e  # Exit on any error

echo "=== Starting training ==="
<<<<<<< HEAD

uv run ./run_embedding_trainer_unfreeze.sh

echo "=== Pushing logs to GitHub ==="

=======

echo "=== Pushing logs to GitHub ==="
>>>>>>> main
git add .
git commit -m "Training completed: $(date)"
git push

echo "=== Uploading to Hugging Face Hub ==="
<<<<<<< HEAD
huggingface-cli upload yuriyvnv/3_alignment_MHGLU_twoWay_loss 3_alignment_MHGLU_twoWay_loss .

echo "=== All done! Shutting down instance ==="
vastai stop instance 22824684 --api-key 8ae7bf55c0e0d706ec35e022dfcc991547b70da038b169380ef994c802f32b43
=======
<<<<<<< HEAD
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
=======
huggingface-cli upload yuriyvnv/3layers_wt_alignment_correct_loss 3layers_wt_alignment_correct_loss .






echo "=== All done! Shutting down instance ==="
vastai stop instance 22745726 --api-key 8ae7bf55c0e0d706ec35e022dfcc991547b70da038b169380ef994c802f32b43

>>>>>>> 13dc2ffc0ae6f7b9351b566570ae276eb2d06626
>>>>>>> main
