#!/bin/bash

echo "🎮 Plant Disease Few-Shot Generation Pipeline (GPU T4)"
echo "========================================================="
echo ""

# Check GPU availability
echo "🔍 Checking GPU availability..."
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}'); [print(f'  - GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo ""

# Step 1: Prepare dataset (if not already done)
if [ ! -d "/kaggle/working/CRDI/datasets/plant_disease_target" ]; then
    echo "📦 Step 1: Preparing dataset..."
    python scripts/prepare_plant_dataset.py
    if [ $? -ne 0 ]; then
        echo "❌ Dataset preparation failed!"
        exit 1
    fi
    echo "✅ Dataset prepared"
    echo ""
else
    echo "✅ Dataset already prepared, skipping..."
    echo ""
fi

# Step 2: Train gradients
echo "🎓 Step 2: Training few-shot gradients on GPU..."
echo "This will take approximately 40-60 minutes on 2x T4 GPUs..."
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
echo "This will take approximately 30-50 minutes on GPU..."
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
echo "📊 Step 4: Creating visualizations..."
python view_generated.py

if [ $? -eq 0 ]; then
    echo "✅ Visualization complete!"
else
    echo "⚠️  Visualization had issues (non-critical)"
fi

echo ""
echo "🎉 Pipeline complete!"
echo ""
echo "📁 Results:"
echo "  - Generated images: arr.npy (5000 images, 256x256x3)"
echo "  - Preview: generated_preview.png"
echo "  - Individual images: generated_images/"
echo "  - Trained gradients: checkpoints/model_plant_disease.pth"
echo ""
echo "📊 Metrics (FID and Intra-LPIPS) should be displayed above"

