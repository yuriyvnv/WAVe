#!/bin/bash
set -e  # Exit on any error

echo "=== Starting evaluation    ==="

uv run ./evaluate_librispeech_nl.py


echo "=== Pushing logs to GitHub ==="
git pull
git add .
git commit -m "Evaluation done for multiple models: $(date)"
git push


echo "==== All done! Shutting down instance ==="
vastai stop instance 24223293 --api-key 8ae7bf55c0e0d706ec35e022dfcc991547b70da038b169380ef994c802f32b43

