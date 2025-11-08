# TPU Setup Guide for Few-Shot Image Generation

## 🚀 Quick Start on Kaggle TPU v5e-8

### Prerequisites
1. Create a new Kaggle notebook
2. Select **TPU v5e-8** as accelerator
3. Add datasets:
   - Plant Disease Dataset: `/kaggle/input/plantdisease/PlantVillage/Pepper__bell___Bacterial_spot`
   - FFHQ Model: `/kaggle/input/ffhq/pytorch/default/1/ffhq.pt`

### One-Command Setup

```bash
# Clone/upload the CRDI repository to /kaggle/working/CRDI
cd /kaggle/working/CRDI

# Install dependencies
pip install -r requirements.txt

# Run complete pipeline (prepare dataset + train + generate)
bash setup_plant_disease.sh
```

This will:
1. ✅ Sample 10 random plant disease images for few-shot training
2. ✅ Create FID reference dataset (2000 images)
3. ✅ Train gradients on TPU (~30-60 min)
4. ✅ Generate 5000 images (~20-40 min)
5. ✅ Calculate FID and LPIPS metrics
6. ✅ Create visualizations

---

## 📋 Step-by-Step Manual Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key TPU dependency:** `torch-xla==2.2.0` (must match PyTorch 2.2.0)

### Step 2: Prepare Plant Disease Dataset

```bash
python scripts/prepare_plant_dataset.py
```

**What this does:**
- Randomly samples 10 images from Pepper Bell Bacterial Spot dataset
- Copies them to `/kaggle/working/CRDI/datasets/plant_disease_target/`
- Creates CSV file with image paths
- Generates FID reference dataset (2000 images) for evaluation

**Output:**
- Training images: `datasets/plant_disease_target/00000.jpg` ... `00009.jpg`
- CSV: `datasets/plant_disease_target/plant_disease.csv`
- FID reference: `datasets/fid_npz/plant_disease.npz`

### Step 3: Train Few-Shot Gradients

```bash
# Set TPU environment
export PJRT_DEVICE=TPU

# Train
python scripts/fs_gradient_train.py --config configs/plant_disease_train.yaml
```

**Configuration** (`plant_disease_train.yaml`):
- `batch_size: 8` (TPU can handle larger batches than GPU)
- `num_workers: 4` (parallel data loading)
- `epochs: 50`
- `use_checkpoint: True` (gradient checkpointing for memory)

**Output:**
- Trained gradients saved to: `checkpoints/model_plant_disease.pth`

### Step 4: Generate 5000 Images

```bash
python scripts/fs_gradient_evaluate.py --config configs/plant_disease_generate.yaml
```

**Output:**
- Generated images: `arr.npy` (5000, 3, 256, 256)
- FID score and Intra-LPIPS metrics printed

### Step 5: Visualize Results

```bash
python view_generated.py
```

**Output:**
- Preview grid: `generated_preview.png`
- Individual images: `generated_images/image_001.png` ... `image_5000.png`

---

## 🔧 TPU Optimizations Applied

### 1. **Auto-Detection**
The code automatically detects TPU availability:
```python
import torch_xla.core.xla_model as xm
device = xm.xla_device()  # Returns 'xla:0' on TPU
```

### 2. **Device-Agnostic Code**
Removed hardcoded `.cuda()` calls:
```python
# Before: x_0.cuda()
# After:  x_0.to(device)
```

### 3. **Larger Batch Sizes**
TPU has more memory than P100 GPU:
- GPU: `batch_size: 1-2`
- TPU: `batch_size: 8` (4-8x faster)

### 4. **Lightning Fabric TPU Support**
```python
fabric = Fabric(
    accelerator="tpu",
    devices="auto",  # Uses all 8 TPU cores
)
```

### 5. **Parallel Data Loading**
```python
num_workers: 4  # Faster data loading on TPU
```

---

## 📊 Expected Performance

### Training (50 epochs, 10 images)
- **GPU P100**: ~45-60 minutes (batch_size=1)
- **TPU v5e-8**: ~25-40 minutes (batch_size=8)
- **Speedup**: ~1.5-2x faster

### Generation (5000 images)
- **GPU P100**: ~40-60 minutes (batch_size=1)
- **TPU v5e-8**: ~20-30 minutes (batch_size=8)
- **Speedup**: ~2x faster

---

## 🎯 Configuration Differences: GPU vs TPU

| Setting | GPU (P100) | TPU (v5e-8) |
|---------|------------|-------------|
| batch_size | 1-2 | 8 |
| num_workers | 0 | 4 |
| use_fp16 | False | False |
| use_checkpoint | True | True |
| accelerator | "cuda" | "tpu" |
| devices | 1 | "auto" (8 cores) |

---

## ⚠️ TPU Limitations & Notes

### 1. **No Mixed Precision (FP16)**
TPU doesn't support the same FP16 implementation as CUDA. Keep `use_fp16: False`.

### 2. **Synchronization**
TPU requires explicit synchronization for metrics:
```python
import torch_xla.core.xla_model as xm
xm.mark_step()  # Sync TPU operations
```

### 3. **First Run Slower**
First TPU run includes compilation time. Subsequent runs are faster.

### 4. **Data Transfer**
Keep data on local disk (`/kaggle/working/`) for faster access.

---

## 🐛 Troubleshooting

### Error: "No module named 'torch_xla'"
```bash
pip install torch-xla==2.2.0
```

### Error: "PJRT device not found"
```bash
export PJRT_DEVICE=TPU
```

### Slow Training
- Increase `batch_size` (try 16 or 32)
- Increase `num_workers` (try 8)
- Ensure data is on local disk

### Out of Memory
- Decrease `batch_size` to 4
- Keep `use_checkpoint: True`

---

## 📈 Metrics Interpretation

### FID Score (Fréchet Inception Distance)
- **Lower is better**
- < 50: Excellent quality
- 50-100: Good quality
- > 100: Poor quality

### Intra-LPIPS (Diversity)
- **Higher is better**
- > 0.3: Good diversity
- 0.2-0.3: Moderate diversity
- < 0.2: Low diversity (mode collapse)

---

## 🔄 Switching Back to GPU

To use GPU instead of TPU, the code auto-detects:

```bash
# Just run without TPU environment
python scripts/fs_gradient_train.py --config configs/plant_disease_train.yaml
```

It will automatically fall back to GPU/CPU if TPU is not available.

---

## 📝 Summary

✅ **Automatic TPU detection** - No code changes needed  
✅ **8x larger batch size** - Faster training/generation  
✅ **Device-agnostic** - Works on TPU, GPU, or CPU  
✅ **Complete pipeline** - One script does everything  
✅ **FID evaluation** - Automatic quality metrics  

For questions or issues, check the main README.md or USAGE.md files.

