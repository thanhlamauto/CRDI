#!/bin/bash

echo "🌿 Setting up Plant Disease Few-Shot Learning"
echo "=============================================="
echo ""

# Check if running on TPU
if python -c "import torch_xla.core.xla_model as xm; xm.xla_device()" 2>/dev/null; then
    echo "✅ TPU detected - using pre-installed PyTorch/XLA"
    echo "📦 Installing additional dependencies..."
    pip install -r requirements_tpu.txt
else
    echo "🖥️  TPU not detected - installing full requirements"
    pip install -r requirements.txt
fi

echo ""

# Step 1: Prepare dataset
echo "📦 Step 1: Preparing dataset (sampling 10 images + creating FID reference)..."
python scripts/prepare_plant_dataset.py

if [ $? -ne 0 ]; then
    echo "❌ Dataset preparation failed!"
    exit 1
fi

echo ""
echo "✅ Dataset preparation complete!"
echo ""

# Step 2: Train gradients
echo "🎓 Step 2: Training few-shot gradients..."
echo "This will take approximately 30-60 minutes on TPU..."
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PJRT_DEVICE=TPU

python scripts/fs_gradient_train.py --config configs/plant_disease_train.yaml

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

echo ""
echo "✅ Training complete!"
echo ""

# Step 3: Generate images
echo "🎨 Step 3: Generating 5000 images..."
echo "This will take approximately 20-40 minutes on TPU..."
echo ""

python scripts/fs_gradient_evaluate.py --config configs/plant_disease_generate.yaml

if [ $? -ne 0 ]; then
    echo "❌ Generation failed!"
    exit 1
fi

echo ""
echo "✅ Generation complete!"
echo ""

# Step 4: Visualize results
echo "📊 Step 4: Creating visualization..."
python view_generated.py

echo ""
echo "🎉 Pipeline complete!"
echo ""
echo "Results:"
echo "  - Generated images: arr.npy (5000 images)"
echo "  - Preview: generated_preview.png"
echo "  - Individual images: generated_images/"
echo "  - Trained gradients: checkpoints/model_plant_disease.pth"
echo ""
echo "FID score and metrics should be displayed above."

