#!/usr/bin/env python3
"""
Script to prepare plant disease dataset for few-shot learning
- Randomly samples 10 images from the dataset
- Creates CSV file with image paths
- Optionally creates FID reference dataset
"""
import os
import random
import glob
import shutil
from pathlib import Path

def sample_images_from_directory(
    source_dir: str,
    output_csv: str,
    output_dir: str,
    num_samples: int = 10,
    seed: int = 2024
):
    """Sample random images from a directory"""
    
    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))
    
    print(f"Found {len(all_images)} images in {source_dir}")
    
    if len(all_images) < num_samples:
        print(f"⚠️  Warning: Only {len(all_images)} images found, but {num_samples} requested")
        num_samples = len(all_images)
    
    # Random sampling
    random.seed(seed)
    sampled_images = random.sample(all_images, num_samples)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy images and create CSV
    csv_lines = []
    for idx, img_path in enumerate(sampled_images):
        # Copy image to output directory
        img_name = f"{idx:05d}{Path(img_path).suffix}"
        dest_path = os.path.join(output_dir, img_name)
        shutil.copy2(img_path, dest_path)
        csv_lines.append(dest_path)
        print(f"Copied: {img_path} -> {dest_path}")
    
    # Write CSV
    with open(output_csv, 'w') as f:
        f.write('\n'.join(csv_lines))
    
    print(f"\n✅ Created dataset:")
    print(f"   - {num_samples} images in: {output_dir}")
    print(f"   - CSV file: {output_csv}")
    
    return csv_lines

def create_fid_reference(source_dir: str, output_npz: str, max_images: int = 2000):
    """Create FID reference dataset from images"""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from tqdm import tqdm
    
    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))
    
    print(f"\n📊 Creating FID reference dataset from {len(all_images)} images...")
    
    # Limit number of images
    if len(all_images) > max_images:
        random.seed(2024)
        all_images = random.sample(all_images, max_images)
        print(f"   Using {max_images} random images")
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Load and process images
    images_array = []
    for img_path in tqdm(all_images, desc="Processing images"):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            images_array.append(img_tensor.numpy())
        except Exception as e:
            print(f"⚠️  Skipped {img_path}: {e}")
            continue
    
    # Save as npz
    images_array = np.array(images_array)
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    np.savez(output_npz, images=images_array)
    
    print(f"\n✅ Created FID reference: {output_npz}")
    print(f"   Shape: {images_array.shape}")

def main():
    # Configuration
    SOURCE_DIR = "/kaggle/input/plantdisease/PlantVillage/Pepper__bell___Bacterial_spot"
    OUTPUT_DIR = "/kaggle/working/CRDI/datasets/plant_disease_target"
    OUTPUT_CSV = "/kaggle/working/CRDI/datasets/plant_disease_target/plant_disease.csv"
    FID_NPZ = "/kaggle/working/CRDI/datasets/fid_npz/plant_disease.npz"
    
    print("🌿 Preparing Plant Disease Dataset for Few-Shot Learning\n")
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")
    
    # Sample 10 images for few-shot training
    sampled_images = sample_images_from_directory(
        source_dir=SOURCE_DIR,
        output_csv=OUTPUT_CSV,
        output_dir=OUTPUT_DIR,
        num_samples=10,
        seed=2024
    )
    
    # Create FID reference dataset
    create_fid_reference(
        source_dir=SOURCE_DIR,
        output_npz=FID_NPZ,
        max_images=2000
    )
    
    print("\n✅ Dataset preparation complete!")
    print(f"\nNext steps:")
    print(f"1. Train: python scripts/fs_gradient_train.py --config configs/plant_disease_train.yaml")
    print(f"2. Generate: python scripts/fs_gradient_evaluate.py --config configs/plant_disease_generate.yaml")

if __name__ == "__main__":
    main()

