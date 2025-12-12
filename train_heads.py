#!/usr/bin/env python3
"""
Train YOLO11s on CrowdHuman HEAD detection dataset.

Head detection is better for CCTV/crowds because:
- Heads are rarely occluded (even in dense crowds)
- Consistent shape/size for tracking
- Better for re-identification anchoring

Usage:
    # Train head detection model
    python train_heads.py
    
    # Resume training
    python train_heads.py --resume
    
    # Use different model size
    python train_heads.py --model yolo11m
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✅ Memory cleared")


def train(
    model_size: str = "yolo11s",
    epochs: int = 25,
    resume: bool = False,
    batch: int = 8,
    device: str = "auto",
):
    """Train head detection model."""
    
    from ultralytics import YOLO
    
    # Auto-detect device
    if device == "auto":
        if torch.cuda.is_available():
            device = "0"  # CUDA GPU
        elif torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon
        else:
            device = "cpu"
    
    print("=" * 70)
    print(f"🧠 Training {model_size.upper()} for HEAD Detection")
    print("=" * 70)
    print(f"   Model: {model_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch}")
    print(f"   Device: {device}")
    print(f"   Resume: {resume}")
    print("=" * 70)
    print()
    print("💡 Press Ctrl+C to safely stop (checkpoint saved)")
    print()
    
    clear_memory()
    
    # Paths
    run_name = f"{model_size}_heads"
    run_dir = Path(f"runs/crowdhuman/{run_name}")
    last_checkpoint = run_dir / "weights" / "last.pt"
    data_yaml = Path("datasets/crowdhuman_heads/dataset.yaml")
    
    # Check if head dataset exists
    if not data_yaml.exists():
        print(f"❌ Head dataset not found: {data_yaml}")
        print("   Run first: python convert_to_yolo.py --box-type hbox --output-dir ./datasets/crowdhuman_heads")
        return None
    
    # Load model
    if resume and last_checkpoint.exists():
        print(f"📂 Resuming from: {last_checkpoint}")
        model = YOLO(str(last_checkpoint))
    else:
        model_path = f"models/{model_size}.pt"
        if not Path(model_path).exists():
            model_path = f"{model_size}.pt"  # Download from ultralytics
        print(f"📦 Loading pretrained {model_path}")
        model = YOLO(model_path)
    
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch,
            imgsz=640,
            device=device,
            workers=2 if device != "mps" else 0,
            project="runs/crowdhuman",
            name=run_name,
            exist_ok=True,
            resume=resume,
            
            # Augmentation (light for head detection)
            mosaic=0.0,       # Disabled for dense annotations
            mixup=0.0,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            scale=0.5,
            translate=0.1,
            
            # Training settings
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            weight_decay=0.0005,
            warmup_epochs=3,
            
            # Memory optimization
            cache=False,
            amp=True,
            
            # Checkpointing
            save=True,
            save_period=5,
            
            # Validation
            val=True,
            verbose=True,
            plots=True,
        )
        
        print()
        print("=" * 70)
        print("✅ Training Complete!")
        print("=" * 70)
        print(f"📁 Results: {run_dir}")
        print(f"🏆 Best model: {run_dir}/weights/best.pt")
        print()
        print("To run head tracking on video:")
        print(f"   python track_heads.py --source Footage/your_video.mp4 --model {run_dir}/weights/best.pt")
        
        return results
        
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("⏸️  Training interrupted")
        print("=" * 70)
        print(f"💾 Checkpoint: {run_dir}/weights/last.pt")
        print("   Resume with: python train_heads.py --resume")
        
    finally:
        clear_memory()


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO for head detection on CrowdHuman"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolo11s",
        choices=["yolo11n", "yolo11s", "yolo11m", "yolo11l"],
        help="Model size (default: yolo11s)"
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
        default=8,
        help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        help="Device: auto, cpu, mps, 0 (CUDA)"
    )
    
    args = parser.parse_args()
    train(
        model_size=args.model,
        epochs=args.epochs,
        resume=args.resume,
        batch=args.batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()

