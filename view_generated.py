#!/usr/bin/env python3
"""
Quick script to visualize generated images from arr.npy
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse

def view_generated_images(npy_path='arr.npy', save_path='generated_preview.png', num_display=20):
    """Load and display generated images"""
    
    if not os.path.exists(npy_path):
        print(f"❌ Error: {npy_path} not found!")
        print("Please run generation first:")
        print("  python scripts/fs_gradient_evaluate.py --config configs/plant_disease_generate.yaml")
        return
    
    # Load images
    images = np.load(npy_path)
    print(f"✅ Loaded {len(images)} images from {npy_path}")
    print(f"📊 Shape: {images.shape}")
    
    # Show first few for preview
    num_display = min(num_display, len(images))
    print(f"🖼️  Displaying first {num_display} images...")
    
    # Determine grid layout for display
    n_display = num_display
    n_cols = 5
    n_rows = (n_display + n_cols - 1) // n_cols
    
    # Create figure for preview
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_display == 1:
        axes = [axes]
    else:
        axes = axes.flat if n_rows > 1 else axes
    
    # Display first N images
    for idx in range(n_display):
        if idx < len(images):
            img = images[idx].transpose(1, 2, 0)  # CHW -> HWC
            img = np.clip(img, 0, 1)  # Ensure values are in [0, 1]
            axes[idx].imshow(img)
            axes[idx].set_title(f"Image {idx+1}", fontsize=10)
            axes[idx].axis('off')
    
    # Hide extra subplots
    for idx in range(n_display, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved preview to: {save_path}")
    
    # Also save individual images (save all of them)
    os.makedirs('generated_images', exist_ok=True)
    print(f"💾 Saving {len(images)} individual images...")
    
    for idx in range(len(images)):
        img = images[idx].transpose(1, 2, 0)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        img_path = f'generated_images/image_{idx+1:04d}.png'
        pil_img.save(img_path)
        
        # Progress indicator every 100 images
        if (idx + 1) % 100 == 0:
            print(f"  Saved {idx + 1}/{len(images)} images...")
    
    print(f"✅ Saved {len(images)} individual images to: generated_images/")
    print("\n📊 Summary:")
    print(f"  - Total images generated: {len(images)}")
    print(f"  - Preview grid: {save_path} (first {num_display} images)")
    print(f"  - Individual images: generated_images/image_0001.png to image_{len(images):04d}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize generated images")
    parser.add_argument("--input", default="arr.npy", help="Input NPY file")
    parser.add_argument("--output", default="generated_preview.png", help="Output preview image")
    parser.add_argument("--num_display", type=int, default=20, help="Number of images to display in preview")
    
    args = parser.parse_args()
    view_generated_images(args.input, args.output, args.num_display)

