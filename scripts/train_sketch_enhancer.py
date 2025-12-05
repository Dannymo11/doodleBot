#!/usr/bin/env python3
"""
Train the Sketch Enhancer model.

This script trains a U-Net model to enhance rough sketches into cleaner ones.
It uses synthetic training pairs created from QuickDraw data.

Usage:
    python scripts/train_sketch_enhancer.py --categories fish cat dog --epochs 50
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.inverse_clipasso.models.sketch_enhancer import create_sketch_enhancer
from src.inverse_clipasso.data.sketch_pairs import SketchPairDataset


def save_sample_grid(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    output_path: Path,
    num_samples: int = 8,
):
    """Save a grid of input/output/target samples."""
    model.eval()
    
    # Get samples
    inputs, targets = next(iter(dataloader))
    inputs = inputs[:num_samples].to(device)
    targets = targets[:num_samples].to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
    
    # Create grid
    fig, axes = plt.subplots(3, num_samples, figsize=(2 * num_samples, 6))
    
    for i in range(num_samples):
        # Input
        inp = (inputs[i, 0].cpu().numpy() + 1) / 2  # [-1,1] -> [0,1]
        axes[0, i].imshow(inp, cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Input", fontsize=12)
        
        # Output
        out = (outputs[i, 0].cpu().numpy() + 1) / 2
        axes[1, i].imshow(out, cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Output", fontsize=12)
        
        # Target
        tgt = (targets[i, 0].cpu().numpy() + 1) / 2
        axes[2, i].imshow(tgt, cmap="gray")
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_ylabel("Target", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    model.train()


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    output_dir: Path,
    save_every: int = 5,
):
    """Train the model."""
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # History
    history = {"train_loss": [], "val_loss": []}
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            loss, losses = model.compute_generator_loss(inputs, targets)
            loss.backward()
            
            optimizer.step()
            
            train_losses.append(losses["l1"])
            pbar.set_postfix({"L1": f"{losses['l1']:.4f}"})
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                _, losses = model.compute_generator_loss(inputs, targets)
                val_losses.append(losses["l1"])
        
        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        
        print(f"Epoch {epoch+1}: Train L1={avg_train:.4f}, Val L1={avg_val:.4f}")
        
        # Save samples and checkpoint
        if (epoch + 1) % save_every == 0:
            save_sample_grid(
                model, val_loader, device,
                output_dir / f"samples_epoch{epoch+1:03d}.png"
            )
            
            torch.save(
                model.state_dict(),
                output_dir / f"checkpoint_epoch{epoch+1:03d}.pt"
            )
    
    # Save final model
    torch.save(model.state_dict(), output_dir / "model_final.pt")
    
    # Plot history
    plt.figure(figsize=(10, 4))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "training_history.png", dpi=150)
    plt.close()
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train Sketch Enhancer")
    parser.add_argument(
        "--categories", nargs="+", default=["fish", "cat", "dog", "bird"],
        help="QuickDraw categories to train on"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--samples", type=int, default=2000, help="Samples per category")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"\n{'='*60}")
    print("SKETCH ENHANCER TRAINING")
    print(f"{'='*60}")
    print(f"Categories: {args.categories}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Samples per category: {args.samples}")
    print(f"Device: {device}")
    
    # Data
    data_dir = Path(PROJECT_ROOT) / "data" / "quickdraw" / "raw" / "strokes"
    
    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        print("Please download QuickDraw data first.")
        return
    
    # Create datasets
    print("\n📦 Creating datasets...")
    
    full_dataset = SketchPairDataset(
        data_dir=str(data_dir),
        categories=args.categories,
        samples_per_category=args.samples,
    )
    
    # Split train/val
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Model
    print("\n🏗️ Creating model...")
    model = create_sketch_enhancer(use_discriminator=False)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(PROJECT_ROOT) / "outputs" / "sketch_enhancer" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    
    # Train
    print("\n🚀 Starting training...")
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=output_dir,
    )
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Model saved to: {output_dir / 'model_final.pt'}")
    print(f"\nTo use the model:")
    print(f"  from src.inverse_clipasso.models import create_sketch_enhancer")
    print(f"  model = create_sketch_enhancer(pretrained='{output_dir / 'model_final.pt'}')")


if __name__ == "__main__":
    main()

