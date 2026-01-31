#!/bin/bash
set -e  # Exit on any error

echo "=== Starting training ==="

uv run ./hf_trainer_complete.py


echo "=== Pushing logs to GitHub ==="
git pull
git add .
git commit -m "Training Completed Tiny mixed quality cv: $(date)"
git push


echo "=== All done! Shutting down instance ==="
vastai stop instance (ANONYMOUS)

