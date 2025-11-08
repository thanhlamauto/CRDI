# 🌿 Plant Disease Few-Shot Generation - Quick Start

## TL;DR - Run This On Kaggle TPU

```bash
cd /kaggle/working/CRDI
pip install -r requirements.txt
bash setup_plant_disease.sh
```

**Done!** This will:
- Sample 10 plant disease images
- Train gradients on TPU
- Generate 5000 new images
- Calculate FID metrics
- Create visualizations

---

## What Gets Created

### Input (Automatically Sampled)
```
datasets/plant_disease_target/
├── 00000.jpg  (sampled from Pepper Bell Bacterial Spot)
├── 00001.jpg
├── ...
└── 00009.jpg  (10 random images total)
```

### Output (After Running)
```
checkpoints/
└── model_plant_disease.pth  (trained gradients, ~few KB)

arr.npy  (5000 generated images, 256x256x3)

generated_preview.png  (grid visualization)

generated_images/
├── image_001.png
├── image_002.png
├── ...
└── image_5000.png
```

---

## Configuration Files Used

### `configs/plant_disease_train.yaml`
- Dataset: 10 plant disease images
- Batch size: 8 (optimized for TPU)
- Epochs: 50
- Output: `checkpoints/model_plant_disease.pth`

### `configs/plant_disease_generate.yaml`
- Generates: 5000 images
- Batch size: 8 (optimized for TPU)
- Uses: `checkpoints/model_plant_disease.pth`
- Evaluates: FID score using reference dataset

---

## Manual Step-by-Step

If you want to run steps individually:

### 1. Prepare Dataset
```bash
python scripts/prepare_plant_dataset.py
```
Output: 10 images + CSV + FID reference

### 2. Train Gradients
```bash
export PJRT_DEVICE=TPU
python scripts/fs_gradient_train.py --config configs/plant_disease_train.yaml
```
Time: ~30-40 minutes on TPU v5e-8

### 3. Generate Images
```bash
python scripts/fs_gradient_evaluate.py --config configs/plant_disease_generate.yaml
```
Time: ~20-30 minutes on TPU v5e-8
Output: `arr.npy` with 5000 images + FID metrics

### 4. Visualize
```bash
python view_generated.py
```
Output: `generated_preview.png` + individual PNGs

---

## Expected Results

### Metrics
- **FID Score**: ~50-150 (depends on domain gap between FFHQ faces and plant leaves)
- **Intra-LPIPS**: ~0.2-0.4 (diversity measure)

### Notes
- FFHQ model is trained on faces, so generating plant images is a **domain transfer** task
- Results may look like abstract/organic patterns rather than realistic plants
- Lower FID = better quality, but domain gap makes this challenging

---

## Customization

### Use Different Number of Images
Edit `scripts/prepare_plant_dataset.py`:
```python
num_samples=10  # Change to 5, 20, 50, etc.
```

### Generate More/Fewer Images
Edit `configs/plant_disease_generate.yaml`:
```yaml
num_evaluate: 5000  # Change to 100, 1000, 10000, etc.
```

### Change Training Duration
Edit `configs/plant_disease_train.yaml`:
```yaml
epochs: 50  # Change to 100, 200, etc.
```

### Use Different Dataset
Edit `scripts/prepare_plant_dataset.py`:
```python
SOURCE_DIR = "/kaggle/input/your-dataset-path"
```

---

## Troubleshooting

### "No such file or directory: /kaggle/input/plantdisease/..."
Make sure you've added the Plant Disease dataset to your Kaggle notebook

### "Checkpoint not found"
Training didn't complete successfully. Check training logs for errors.

### TPU not detected
Make sure you selected **TPU v5e-8** in notebook settings

### Out of memory
Reduce `batch_size` in configs (try 4 or 2)

---

## What's Different from Baby Faces?

| Aspect | Baby Faces | Plant Disease |
|--------|------------|---------------|
| Dataset | `/kaggle/working/CRDI/datasets/babies_target/` | Pepper Bell Bacterial Spot |
| Images | 10 baby faces | 10 plant leaf images |
| Config | `configs/ffhq_train.yaml` | `configs/plant_disease_train.yaml` |
| Category | "babies" | "plant_disease" |
| Checkpoint | `model_babies.pth` | `model_plant_disease.pth` |
| Batch Size (TPU) | 1-2 (GPU optimized) | 8 (TPU optimized) |

---

## Next Steps

After generation completes:

1. **View Results**: Check `generated_preview.png`
2. **Analyze Metrics**: Look at FID and LPIPS scores in terminal output
3. **Adjust Settings**: Increase epochs or learning rate if quality is poor
4. **Try Different Data**: Sample different plants from the dataset

For detailed TPU information, see `TPU_SETUP.md`.

