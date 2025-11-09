#!/usr/bin/env python3
"""
Compare two datasets: visualize samples and compute FID score
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import os
import argparse
from tqdm import tqdm


def visualize_samples(images, title, num_samples=10, save_path=None):
    """Visualize samples from a dataset"""
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Total images: {len(images)}")
    print(f"Shape: {images.shape}")
    print(f"Value range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Sample random images
    num_samples = min(num_samples, len(images))
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    # Create figure
    n_cols = 5
    n_rows = (num_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flat if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for idx, img_idx in enumerate(indices):
        img = images[img_idx]
        
        # Convert from CHW to HWC if needed
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        
        # Ensure values are in [0, 1]
        img = np.clip(img, 0, 1)
        
        axes[idx].imshow(img)
        axes[idx].set_title(f"Sample {img_idx}", fontsize=10)
        axes[idx].axis('off')
    
    # Hide extra subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved visualization to: {save_path}")
    
    return fig


def compute_fid_statistics(images, batch_size=50):
    """Compute FID statistics (mean and covariance) from images"""
    
    print("\n📊 Computing FID statistics...")
    
    # Load InceptionV3 model
    from src.fs_gradients.fid_score import InceptionV3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    model.eval()
    
    # Compute activations in batches
    activations = []
    num_batches = (len(images) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(images), batch_size), total=num_batches, desc="Computing activations"):
        batch = images[i:i+batch_size]
        
        # Convert to torch tensor
        batch_tensor = torch.from_numpy(batch).to(device)
        
        # Ensure correct format (B, C, H, W)
        if batch_tensor.shape[1] != 3:
            batch_tensor = batch_tensor.permute(0, 3, 1, 2)
        
        with torch.no_grad():
            pred = model(batch_tensor)[0]
        
        # Squeeze spatial dimensions
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
        
        activations.append(pred.squeeze(3).squeeze(2).cpu().numpy())
    
    activations = np.concatenate(activations, axis=0)
    
    # Compute statistics
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    
    print(f"✅ Statistics computed:")
    print(f"   - Activations shape: {activations.shape}")
    print(f"   - Mean shape: {mu.shape}")
    print(f"   - Covariance shape: {sigma.shape}")
    
    return mu, sigma, activations


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate FID score between two distributions"""
    
    print("\n🧮 Calculating FID score...")
    
    from scipy import linalg
    
    # Calculate mean difference
    diff = mu1 - mu2
    
    # Product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Check for numerical stability
    if not np.isfinite(covmean).all():
        print("⚠️  Warning: covmean contains inf or NaN, adding epsilon to diagonal")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Handle complex numbers
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print(f"⚠️  Warning: Imaginary component {m}")
        covmean = covmean.real
    
    # Calculate FID
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    
    return fid


def main():
    parser = argparse.ArgumentParser(description="Compare two datasets and compute FID")
    parser.add_argument("--generated", default="/kaggle/input/plant-village-generated/arr.npy", 
                        help="Path to generated images (NPY)")
    parser.add_argument("--reference", default="/kaggle/input/reference-data/plant_disease.npz",
                        help="Path to reference images (NPZ)")
    parser.add_argument("--num_samples", type=int, default=10, 
                        help="Number of samples to visualize from each dataset")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Batch size for FID computation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("="*80)
    print("  DATASET COMPARISON AND FID CALCULATION")
    print("="*80)
    
    # Load generated images
    print("\n📂 Loading generated images...")
    if not os.path.exists(args.generated):
        print(f"❌ Error: {args.generated} not found!")
        return
    
    generated_images = np.load(args.generated)
    print(f"✅ Loaded generated images: {generated_images.shape}")
    
    # Load reference images
    print("\n📂 Loading reference dataset...")
    if not os.path.exists(args.reference):
        print(f"❌ Error: {args.reference} not found!")
        return
    
    with np.load(args.reference) as f:
        if "images" in f:
            reference_images = f["images"]
            print(f"✅ Loaded reference images: {reference_images.shape}")
        elif "mu" in f and "sigma" in f:
            print("✅ Reference contains precomputed statistics")
            reference_mu = f["mu"]
            reference_sigma = f["sigma"]
            reference_images = None
        else:
            print(f"❌ Error: Unknown format in {args.reference}")
            return
    
    # Visualize samples from generated dataset
    print("\n" + "="*80)
    print("STEP 1: VISUALIZE GENERATED IMAGES")
    print("="*80)
    visualize_samples(
        generated_images, 
        "Generated Images (Plant Village)",
        num_samples=args.num_samples,
        save_path="generated_samples_preview.png"
    )
    
    # Visualize samples from reference dataset
    if reference_images is not None:
        print("\n" + "="*80)
        print("STEP 2: VISUALIZE REFERENCE IMAGES")
        print("="*80)
        visualize_samples(
            reference_images,
            "Reference Images (Plant Disease Dataset)", 
            num_samples=args.num_samples,
            save_path="reference_samples_preview.png"
        )
    
    # Compute FID
    print("\n" + "="*80)
    print("STEP 3: COMPUTE FID SCORE")
    print("="*80)
    
    # Compute statistics for generated images
    print("\n[1/2] Processing generated images...")
    gen_mu, gen_sigma, gen_acts = compute_fid_statistics(generated_images, args.batch_size)
    
    # Compute or load statistics for reference images
    if reference_images is not None:
        print("\n[2/2] Processing reference images...")
        ref_mu, ref_sigma, ref_acts = compute_fid_statistics(reference_images, args.batch_size)
    else:
        print("\n[2/2] Using precomputed reference statistics...")
        ref_mu = reference_mu
        ref_sigma = reference_sigma
    
    # Calculate FID
    fid_score = calculate_fid(gen_mu, gen_sigma, ref_mu, ref_sigma)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\n📊 Dataset Statistics:")
    print(f"   Generated images:  {len(generated_images)}")
    if reference_images is not None:
        print(f"   Reference images:  {len(reference_images)}")
    
    print(f"\n🎯 FID Score: {fid_score:.2f}")
    
    print("\n📈 Interpretation:")
    if fid_score < 50:
        print("   ✅ Excellent - Very similar to reference dataset")
    elif fid_score < 100:
        print("   ✅ Good - High quality generation")
    elif fid_score < 200:
        print("   ⚠️  Moderate - Acceptable for domain transfer")
    else:
        print("   ❌ Poor - Large difference from reference")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Generated samples visualization: generated_samples_preview.png")
    if reference_images is not None:
        print(f"✅ Reference samples visualization: reference_samples_preview.png")
    print(f"✅ FID Score: {fid_score:.2f}")
    print("="*80 + "\n")
    
    # Save results to file
    with open("fid_results.txt", "w") as f:
        f.write(f"FID Score: {fid_score:.2f}\n")
        f.write(f"Generated images: {len(generated_images)}\n")
        if reference_images is not None:
            f.write(f"Reference images: {len(reference_images)}\n")
    
    print("📁 Results saved to: fid_results.txt")


if __name__ == "__main__":
    main()

