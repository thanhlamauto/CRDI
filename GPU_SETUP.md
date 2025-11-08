# GPU Setup Guide (Kaggle 2x T4)

## 🎮 Quick Start - Complete Pipeline

```bash
cd /kaggle/working/CRDI
bash run_pipeline_gpu.sh
```

This single command will:
1. ✅ Prepare dataset (10 plant disease images + FID reference)
2. ✅ Train gradients on 2x T4 GPUs (~40-60 min)
3. ✅ Generate 5000 images (~30-50 min)
4. ✅ Calculate FID and LPIPS metrics
5. ✅ Create visualizations

---

## 📋 Step-by-Step Manual Run

### 1. Prepare Dataset (If Not Already Done)
```bash
python scripts/prepare_plant_dataset.py
```

**Output:**
- 10 sampled images: `datasets/plant_disease_target/`
- FID reference: `datasets/fid_npz/plant_disease.npz` (997 images)

### 2. Train Gradients
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/fs_gradient_train.py --config configs/plant_disease_train.yaml
```

**Time:** ~40-60 minutes on 2x T4
**Output:** `checkpoints/model_plant_disease.pth`

### 3. Generate Images
```bash
python scripts/fs_gradient_evaluate.py --config configs/plant_disease_generate.yaml
```

**Time:** ~30-50 minutes
**Output:** `arr.npy` (5000 images, 256x256x3)

### 4. Visualize Results
```bash
python view_generated.py
```

**Output:**
- `generated_preview.png` (grid preview)
- `generated_images/` (5000 individual PNGs)

---

## ⚙️ GPU T4 Optimizations Applied

### Configuration Changes

| Setting | TPU (v5e-8) | GPU (2x T4) |
|---------|-------------|-------------|
| **Training batch_size** | 8 | 2 |
| **Generation batch_size** | 8 | 4 |
| **num_workers** | 4 | 2 |
| **Strategy** | Single device | DDP (2 GPUs) |
| **use_checkpoint** | True | True |
| **use_fp16** | False | False |

### Why These Settings?

**T4 GPU Specs:**
- Memory: 16GB per GPU
- Memory Bandwidth: 320 GB/s
- FP32 Performance: 8.1 TFLOPS

**Batch Size:**
- Training: `batch_size=2` per GPU (effective batch=4 with 2 GPUs)
- Generation: `batch_size=4` (single GPU for generation)
- Gradient checkpointing enabled to save memory

**Multi-GPU Strategy:**
- Uses **DDP (Distributed Data Parallel)** for 2 GPUs
- Automatically syncs gradients across GPUs
- Lightning Fabric handles distribution automatically

---

## 🔧 Auto-Detection Logic

The code automatically detects your hardware:

```python
# Detection order:
1. TPU (torch_xla) → Use XLA backend
2. GPU (CUDA) → Use DDP if 2+ GPUs
3. CPU → Fallback
```

**On Kaggle 2x T4:**
```
🎮 Running on 2x GPU
  - GPU 0: Tesla T4
  - GPU 1: Tesla T4
Strategy: ddp
```

---

## 📊 Expected Performance

### Training (50 epochs, 10 images)
- **1x T4**: ~60-80 minutes
- **2x T4**: ~40-60 minutes
- **Speedup**: ~1.3-1.5x (DDP overhead reduces scaling)

### Generation (5000 images)
- **1x T4**: ~40-60 minutes  
- **2x T4**: ~30-50 minutes
- Uses single GPU (no DDP for generation)

---

## 📈 Metrics Explanation

### FID (Fréchet Inception Distance)
**What it measures:** Image quality compared to reference dataset

- **< 50**: Excellent quality
- **50-100**: Good quality
- **100-200**: Moderate quality (expected for domain transfer)
- **> 200**: Poor quality

**Note:** FFHQ model (trained on faces) generating plant images is a **domain transfer** task, so FID may be 100-200, which is acceptable.

### Intra-LPIPS (Diversity)
**What it measures:** Diversity between generated images

- **> 0.3**: High diversity (good)
- **0.2-0.3**: Moderate diversity
- **< 0.2**: Low diversity (mode collapse risk)

---

## 🐛 Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size in config
batch_size: 1  # For training
batch_size: 2  # For generation
```

### Slow Training
```bash
# Check GPU utilization
nvidia-smi

# Should show ~90%+ GPU utilization
# If low, increase batch_size
```

### Only 1 GPU Detected
```bash
# Check available GPUs
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Verify in Kaggle settings:
# Accelerator: GPU T4 x2
```

### DDP Hanging
```bash
# If training hangs with 2 GPUs, try single GPU:
fabric = Fabric(accelerator="cuda", devices=1)
```

---

## 💡 Pro Tips

1. **Monitor GPU usage:**
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Check memory:**
   ```bash
   # Should use ~10-14GB per GPU during training
   ```

3. **Batch size tuning:**
   - Start with 2, increase if memory allows
   - Training: Max ~4 per GPU
   - Generation: Max ~8 per GPU

4. **Speed vs Quality:**
   - More epochs → Better quality (50-100 recommended)
   - More timesteps (t_start to t_end) → More control

5. **Save checkpoints:**
   - Gradients auto-save to `checkpoints/`
   - Small file size (~few KB)
   - Reusable for multiple generations

---

## 🔄 Switching Between Accelerators

The code auto-detects hardware. To force specific accelerator:

**Force GPU:**
```python
fabric = Fabric(accelerator="cuda", devices=2)
```

**Force Single GPU:**
```python
fabric = Fabric(accelerator="cuda", devices=1)
```

**Force CPU (slow):**
```python
fabric = Fabric(accelerator="cpu", devices=1)
```

---

## 📁 Output Files

```
checkpoints/
└── model_plant_disease.pth  (~few KB, trained gradients)

datasets/
├── plant_disease_target/  (10 training images)
└── fid_npz/
    └── plant_disease.npz  (997 images for FID)

arr.npy  (5000 generated images, ~1.5GB)

generated_preview.png  (grid visualization)

generated_images/
├── image_001.png
├── image_002.png
├── ...
└── image_5000.png
```

---

## ⏱️ Total Time Estimate

| Stage | Time (2x T4) |
|-------|--------------|
| Dataset prep | ~1-2 min |
| Training (50 epochs) | ~40-60 min |
| Generation (5000) | ~30-50 min |
| Visualization | ~1-2 min |
| **Total** | **~75-115 min** |

---

## 🎯 Success Criteria

✅ **Training completes** without OOM errors  
✅ **FID score** < 200 (acceptable for domain transfer)  
✅ **Intra-LPIPS** > 0.2 (good diversity)  
✅ **Generated images** look plant-like (not pure noise)  
✅ **5000 images** saved successfully  

For questions, see main `README.md` or `USAGE.md`.

