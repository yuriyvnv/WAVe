#!/bin/bash
set -e  # Exit on any error

echo "=== Starting training ==="

uv run ./hf_trainer_complete.py


echo "=== Pushing logs to GitHub ==="
git add .
git commit -m "Training completed for tiny learning rate 1e-5: $(date)"
git push


echo "=== All done! Shutting down instance ==="
vastai stop instance 23860683 --api-key 8ae7bf55c0e0d706ec35e022dfcc991547b70da038b169380ef994c802f32b43

