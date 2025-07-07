#!/bin/bash
set -e  # Exit on any error

echo "=== Starting training ==="

echo "=== Pushing logs to GitHub ==="
git add .
git commit -m "Training completed: $(date)"
git push

echo "=== Uploading to Hugging Face Hub ==="
huggingface-cli upload yuriyvnv/3layers_wt_alignment_correct_loss 3layers_wt_alignment_correct_loss .


echo "=== Getting instance details for shutdown ==="
# Source environment variables from the speech_transcript_embeddings folder
if [ -f "speech_transcript_embeddings/.env" ]; then
    source speech_transcript_embeddings/.env
    echo "Loaded environment variables"
else
    echo "Warning: .env file not found in speech_transcript_embeddings folder"
fi

# Get instance ID from .vast_containerlabel file
if [ -f "/.vast_containerlabel" ]; then
    # Extract only the numeric part from the container label (e.g., "C.22745726" -> "22745726")
    INSTANCE_ID=$(cat /.vast_containerlabel | grep -o '[0-9]\+')
    echo "Found instance ID in .vast_containerlabel: $INSTANCE_ID"
else
    echo "Error: .vast_containerlabel file not found in root directory"
    exit 1
fi

# Check if we have the required variables
if [ -z "$VASTAI_API_KEY" ]; then
    echo "Error: VASTAI_API_KEY not found in environment"
    echo "Please set it in speech_transcript_embeddings/.env file"
    exit 1
fi

if [ -z "$INSTANCE_ID" ]; then
    echo "Error: Could not determine INSTANCE_ID"
    echo "Please create a file with the instance ID or set it manually"
    exit 1
fi

echo "Instance ID: $INSTANCE_ID"
echo "API Key: ${VASTAI_API_KEY:0:8}..." # Show only first 8 chars for security

echo "=== All done! Shutting down instance ==="
vastai stop instance "$INSTANCE_ID" --api-key "$VASTAI_API_KEY"

echo "Instance shutdown initiated successfully!"