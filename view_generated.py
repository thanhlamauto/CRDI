#!/usr/bin/env python3
"""
Quick script to visualize generated images from arr.npy
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def view_generated_images(npy_path='arr.npy', save_path='generated_preview.png'):
    """Load and display generated images"""
    
    if not os.path.exists(npy_path):
        print(f"❌ Error: {npy_path} not found!")
        print("Please run generation first:")
        print("  python scripts/fs_gradient_evaluate.py --config configs/ffhq_generate.yaml")
        return
    
    # Load images
    images = np.load(npy_path)
    print(f"✅ Loaded {len(images)} images from {npy_path}")
    print(f"📊 Shape: {images.shape}")
    
    # Determine grid layout
    n_images = len(images)
    n_cols = min(5, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flat if n_rows > 1 else axes
    
    # Display images
    for idx in range(n_images):
        img = images[idx].transpose(1, 2, 0)  # CHW -> HWC
        img = np.clip(img, 0, 1)  # Ensure values are in [0, 1]
        axes[idx].imshow(img)
        axes[idx].set_title(f"Image {idx+1}")
        axes[idx].axis('off')
    
    # Hide extra subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved preview to: {save_path}")
    
    # Also save individual images
    os.makedirs('generated_images', exist_ok=True)
    for idx in range(n_images):
        img = images[idx].transpose(1, 2, 0)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        img_path = f'generated_images/image_{idx+1:03d}.png'
        pil_img.save(img_path)
    
    print(f"📁 Saved {n_images} individual images to: generated_images/")
    print("\nTo view:")
    print(f"  - Preview: {save_path}")
    print(f"  - Individual images: generated_images/image_001.png, etc.")

if __name__ == "__main__":
    view_generated_images()

