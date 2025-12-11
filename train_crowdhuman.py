#!/usr/bin/env python3
"""
Train YOLO11n on CrowdHuman dataset - Memory Optimized for 16GB RAM

Features:
- Safe to stop with Ctrl+C (saves checkpoint)
- Resume from last checkpoint with --resume
- Memory optimized settings for Apple Silicon

Usage:
    # Start fresh training
    python train_crowdhuman.py
    
    # Resume from last checkpoint
    python train_crowdhuman.py --resume
    
    # Custom epochs
    python train_crowdhuman.py --epochs 50
"""

import argparse
import gc
import torch
from pathlib import Path


def clear_memory():
    """Clear GPU/MPS memory before training."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("✅ Memory cleared")


def train(epochs: int = 25, resume: bool = False, batch: int = 4):
    """Run training with memory-optimized settings."""
    
    from ultralytics import YOLO
    
    print("=" * 70)
    print("🏋️  YOLO11n Training on CrowdHuman Dataset")
    print("=" * 70)
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch}")
    print(f"   Resume: {resume}")
    print(f"   Device: MPS (Apple Silicon)")
    print("=" * 70)
    print()
    print("💡 Press Ctrl+C to safely stop training (checkpoint will be saved)")
    print("   Then run with --resume to continue from where you left off")
    print()
    
    # Clear memory first
    clear_memory()
    
    # Paths
    run_dir = Path("runs/crowdhuman/yolo11n_crowdhuman")
    last_checkpoint = run_dir / "weights" / "last.pt"
    
    # Load model
    if resume and last_checkpoint.exists():
        print(f"📂 Resuming from: {last_checkpoint}")
        model = YOLO(str(last_checkpoint))
    else:
        print("📦 Loading pretrained yolo11n.pt")
        model = YOLO("models/yolo11n.pt")
    
    # Training with memory-optimized settings
    try:
        results = model.train(
            data="datasets/crowdhuman_yolo/dataset.yaml",
            epochs=epochs,
            batch=batch,           # Small batch for 16GB RAM
            imgsz=640,
            device="mps",          # Apple Silicon GPU
            workers=0,             # No extra workers (saves RAM)
            project="runs/crowdhuman",
            name="yolo11n_crowdhuman",
            exist_ok=True,         # Allow resuming into same folder
            resume=resume,         # Resume training state
            
            # Disable heavy augmentation (causes issues with dense images)
            mosaic=0.0,            # Disabled - problematic with 400+ box images
            mixup=0.0,             # Disabled
            copy_paste=0.0,        # Disabled
            
            # Keep light augmentation
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            scale=0.5,
            translate=0.1,
            
            # Training settings
            optimizer="AdamW",
            lr0=0.001,             # Lower LR for fine-tuning
            lrf=0.01,
            weight_decay=0.0005,
            warmup_epochs=3,
            
            # Memory optimization
            cache=False,           # Don't cache images in RAM
            amp=True,              # Mixed precision (saves memory)
            
            # Checkpointing
            save=True,
            save_period=5,         # Save every 5 epochs
            
            # Validation & logging
            val=True,
            verbose=True,
            plots=True,
        )
        
        print()
        print("=" * 70)
        print("✅ Training Complete!")
        print("=" * 70)
        print(f"📁 Results saved to: {run_dir}")
        print(f"🏆 Best model: {run_dir}/weights/best.pt")
        
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("⏸️  Training interrupted by user")
        print("=" * 70)
        print(f"💾 Checkpoint saved to: {run_dir}/weights/last.pt")
        print()
        print("To resume training, run:")
        print("   python train_crowdhuman.py --resume")
        
    finally:
        # Clean up memory
        clear_memory()


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO11n on CrowdHuman (Memory Optimized)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=25,
        help="Number of epochs (default: 25)"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=4,
        help="Batch size (default: 4, use 2 if still OOM)"
    )
    
    args = parser.parse_args()
    train(epochs=args.epochs, resume=args.resume, batch=args.batch)


if __name__ == "__main__":
    main()

