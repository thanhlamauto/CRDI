# Kaggle TPU Installation Guide

## 🚨 Important: Pre-installed Packages on Kaggle TPU

Kaggle TPU environments come with **PyTorch and torch-xla pre-installed**. You should **NOT** reinstall them as it may cause conflicts.

## ✅ Correct Installation on Kaggle TPU

### Option 1: Using TPU-specific requirements (Recommended)

```bash
cd /kaggle/working/CRDI
pip install -r requirements_tpu.txt
```

This installs only the additional dependencies, skipping torch/torchvision/torch-xla.

### Option 2: Using main requirements (GPU/Local)

```bash
pip install -r requirements.txt
```

This includes PyTorch but uses flexible version constraints (`>=2.0.0`).

### Option 3: Automatic detection (Best)

The `setup_plant_disease.sh` script automatically detects TPU:

```bash
bash setup_plant_disease.sh
```

It will:
- Detect if TPU is available
- Use `requirements_tpu.txt` on TPU
- Use `requirements.txt` on GPU/CPU

---

## 🔍 Verifying Your Environment

Check what's installed in your Kaggle environment:

```python
import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    print(f"✅ TPU available: {device}")
    print(f"torch-xla version: {torch_xla.__version__}")
except Exception as e:
    print(f"❌ TPU not available: {e}")
```

---

## 📋 What's in Each Requirements File

### `requirements_tpu.txt` (For Kaggle TPU)
- ✅ Excludes: torch, torchvision, torch-xla
- ✅ Includes: All other dependencies (lightning, lpips, etc.)
- ✅ Use when: Running on Kaggle TPU

### `requirements.txt` (For GPU/Local)
- ✅ Includes: torch>=2.0.0, torchvision>=0.15.0
- ✅ Flexible versions to avoid conflicts
- ✅ Use when: Running on GPU or local machine

---

## 🐛 Troubleshooting

### Error: "No matching distribution found for torch-xla"

**Solution:** Use `requirements_tpu.txt` instead:
```bash
pip install -r requirements_tpu.txt
```

### Error: "torch_xla module not found"

**Check 1:** Are you on a TPU notebook?
- Go to Kaggle notebook settings
- Ensure **TPU v5e-8** is selected as accelerator

**Check 2:** Verify torch-xla is installed:
```python
import torch_xla
print(torch_xla.__version__)
```

### Error: Version conflicts

**Solution:** Don't mix requirements files. Use only one:
- TPU: `requirements_tpu.txt`
- GPU/Local: `requirements.txt`

---

## 🚀 Quick Start Commands

### For Kaggle TPU (Most Common)

```bash
# Navigate to project
cd /kaggle/working/CRDI

# Install dependencies (TPU-specific)
pip install -r requirements_tpu.txt

# Run complete pipeline
bash setup_plant_disease.sh
```

### For Kaggle GPU

```bash
cd /kaggle/working/CRDI
pip install -r requirements.txt
bash setup_plant_disease.sh
```

---

## 📊 Expected Behavior

### On TPU ✅
```
🚀 Running on TPU: xla:0
torch-xla version: 2.x.x (pre-installed)
```

### On GPU ✅
```
🖥️  Running on: cuda
CUDA available: True
```

### On CPU (Fallback) ⚠️
```
🖥️  Running on: cpu
Warning: This will be very slow
```

---

## 💡 Pro Tips

1. **Always check accelerator first** before installing
2. **Use `setup_plant_disease.sh`** - it handles everything automatically
3. **Don't reinstall PyTorch on Kaggle** - use pre-installed versions
4. **Check TPU cores**: Should show 8 cores available for TPU v5e-8
5. **Monitor memory**: Use `!nvidia-smi` for GPU, XLA profiler for TPU

---

## 🔗 Useful Resources

- [Kaggle TPU Documentation](https://www.kaggle.com/docs/tpu)
- [PyTorch XLA Guide](https://pytorch.org/xla/)
- [Lightning TPU Support](https://lightning.ai/docs/pytorch/stable/accelerators/tpu.html)

