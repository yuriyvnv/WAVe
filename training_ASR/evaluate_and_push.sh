#!/bin/bash
set -e  # Exit on any error

echo "=== Starting evaluation    ==="

uv run ./evaluate_results.py


echo "=== Pushing logs to GitHub ==="
git pull
git add .
git commit -m "Evaluation done for model yuriyvnv/whisper-small-high-mixed-nl: $(date)"
git push


echo "==== All done! Shutting down instance ==="
vastai stop instance 23906825 --api-key 8ae7bf55c0e0d706ec35e022dfcc991547b70da038b169380ef994c802f32b43

