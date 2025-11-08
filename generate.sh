#!/bin/bash

# Script to generate images using trained gradients
# Usage: bash generate.sh [checkpoint_path] [num_images]

# Default values
CHECKPOINT=${1:-"checkpoints/model_babies.pth"}
NUM_IMAGES=${2:-5}

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found at $CHECKPOINT"
    echo "Please train gradients first using:"
    echo "  python scripts/fs_gradient_train.py --config configs/ffhq_train.yaml"
    exit 1
fi

echo "🎨 Generating images using gradients from: $CHECKPOINT"
echo "📊 Number of images to generate: $NUM_IMAGES"
echo ""

# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run generation
python scripts/fs_gradient_evaluate.py \
    --config configs/ffhq_generate.yaml \
    --experiment_gradient_path "$CHECKPOINT" \
    --num_evaluate "$NUM_IMAGES" \
    --batch_size 1 \
    --use_checkpoint True

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Generation completed!"
    echo "📁 Generated images saved to: arr.npy"
    echo ""
    echo "To view the generated images:"
    echo "  python view_generated.py"
    echo ""
    echo "Or load in Python:"
    echo "  import numpy as np"
    echo "  images = np.load('arr.npy')  # Shape: ($NUM_IMAGES, 3, 256, 256)"
else
    echo ""
    echo "❌ Generation failed!"
fi

