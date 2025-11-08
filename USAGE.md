# CRDI - Few-Shot Image Generation

## Quick Start Guide

### Step 1: Train Gradients on Your Images

Train few-shot gradients using your 10 baby images:

```bash
# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Train gradients (uses configs/ffhq_train.yaml)
python scripts/fs_gradient_train.py --config configs/ffhq_train.yaml
```

**What this does:**
- Loads pre-trained FFHQ diffusion model
- Learns gradient directions from your 10 baby images
- Saves gradients to `checkpoints/model_babies.pth`

**Training time:** ~30-60 minutes (50 epochs)

---

### Step 2: Generate New Images Using Trained Gradients

Once training completes, generate new baby-like images:

```bash
# Option 1: Use the helper script (easiest)
bash generate.sh checkpoints/model_babies.pth 100

# Option 2: Run directly with config
python scripts/fs_gradient_evaluate.py \
    --config configs/ffhq_generate.yaml \
    --experiment_gradient_path checkpoints/model_babies.pth \
    --num_evaluate 100

# Option 3: With custom parameters
python scripts/fs_gradient_evaluate.py \
    --experiment_gradient_path checkpoints/model_babies.pth \
    --num_evaluate 5000 \
    --t_start 5 --t_end 15 --num_gradient 1 \
    --batch_size 1 --use_checkpoint True
```

**Output:**
- Generated images saved to `arr.npy`
- FID and Intra-LPIPS metrics printed

---

### Step 3: View Generated Images

```python
import numpy as np
import matplotlib.pyplot as plt

# Load generated images
images = np.load('arr.npy')  # Shape: (N, 3, 256, 256)

# Visualize first 10 images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for idx, ax in enumerate(axes.flat):
    img = images[idx].transpose(1, 2, 0)  # CHW -> HWC
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.savefig('generated_samples.png')
```

---

## Full Workflow (Training + Generation)

Use the `main.sh` script for the complete pipeline:

```bash
bash main.sh
```

This will:
1. Train gradients (120 epochs, extended training)
2. Generate 5000 images for evaluation
3. Calculate FID and LPIPS metrics

---

## Configuration Files

### `configs/ffhq_train.yaml` - Training Configuration
- `num_samples`: 10 (number of training images)
- `epochs`: 50 (training iterations)
- `batch_size`: 1 (for P100 GPU memory)
- `learning_rate`: 0.05
- `use_checkpoint`: True (gradient checkpointing for memory)

### `configs/ffhq_generate.yaml` - Generation Configuration
- `num_evaluate`: 5000 (images to generate)
- `experiment_gradient_path`: Path to trained gradients
- `batch_size`: 1 (for memory efficiency)
- `category`: "babies" (must match training)

---

## File Paths (Kaggle Environment)

- **Pre-trained Model**: `/kaggle/input/ffhq/pytorch/default/1/ffhq.pt`
- **Training Images**: `/kaggle/working/CRDI/datasets/babies_target/`
- **Image List**: `/kaggle/working/CRDI/datasets/babies_target/babies.csv`
- **Trained Gradients**: `checkpoints/model_babies.pth`
- **Generated Images**: `arr.npy`

---

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size` to 1 in config
- Enable `use_checkpoint: True`
- Set `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### Checkpoint Not Found
- Make sure training completed successfully
- Check that `checkpoints/model_babies.pth` exists
- Verify path in `experiment_gradient_path` parameter

### Type Mismatch Error
- Ensure `use_fp16: False` in config (FP16 not compatible with this setup)

---

## Key Parameters Explained

- **t_start, t_end**: Diffusion timestep range (5-15 works well)
- **num_gradient**: Number of gradient vectors (1 for simple cases)
- **classifier_scale**: Gradient strength (2.0 = balanced)
- **anneal_ptb**: Add noise during generation for diversity
- **anneal_scale**: Amount of noise to add (0.2 = mild)

---

## Memory Requirements

- **P100 GPU (16GB)**: batch_size=1, use_checkpoint=True
- **Estimated usage**: ~7-8 GB with current settings

