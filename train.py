#!/usr/bin/env python3
"""
Train YOLO11 models on CrowdHuman dataset for improved crowd detection.

This script fine-tunes YOLO11 models (nano, small, medium, large) on the 
CrowdHuman dataset to improve detection of individuals in crowded scenes.

Usage:
    # Train with YOLO11n (fastest, smallest)
    python train.py --model yolo11n
    
    # Train with YOLO11s (balanced)
    python train.py --model yolo11s
    
    # Train with YOLO11m (more accurate)
    python train.py --model yolo11m
    
    # Custom training
    python train.py --model yolo11n --epochs 100 --batch 16 --imgsz 640
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def get_model_path(model_name: str) -> str:
    """
    Get the path to a pretrained model or download it.
    
    Args:
        model_name: Model name (yolo11n, yolo11s, yolo11m, yolo11l)
    
    Returns:
        Path to model weights
    """
    # Check for local pretrained models first
    local_models = Path("./models")
    
    model_mapping = {
        "yolo11n": "yolo11n.pt",
        "yolo11s": "yolo11s.pt", 
        "yolo11m": "yolo11m.pt",
        "yolo11l": "yolo11l.pt",
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(model_mapping.keys())}")
    
    model_file = model_mapping[model_name]
    local_path = local_models / model_file
    
    if local_path.exists():
        print(f"📦 Using local model: {local_path}")
        return str(local_path)
    else:
        print(f"📥 Model {model_file} not found locally, will download from Ultralytics")
        return model_file


def train(
    model_name: str = "yolo11n",
    data_yaml: str = "./datasets/crowdhuman_yolo/dataset.yaml",
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "",
    workers: int = 8,
    project: str = "runs/crowdhuman",
    name: str = None,
    resume: bool = False,
    pretrained: bool = True,
):
    """
    Train a YOLO11 model on CrowdHuman dataset.
    
    Args:
        model_name: Model variant (yolo11n, yolo11s, yolo11m, yolo11l)
        data_yaml: Path to dataset configuration YAML
        epochs: Number of training epochs
        batch_size: Batch size for training
        imgsz: Input image size
        device: Device to use ('' for auto, 'cpu', '0', '0,1', 'mps')
        workers: Number of data loading workers
        project: Project directory for saving runs
        name: Run name (auto-generated if None)
        resume: Resume training from last checkpoint
        pretrained: Use pretrained weights (recommended for fine-tuning)
    """
    print("=" * 70)
    print("🚀 YOLO11 Training on CrowdHuman Dataset")
    print("=" * 70)
    print(f"📊 Model: {model_name}")
    print(f"📁 Dataset: {data_yaml}")
    print(f"⚙️  Epochs: {epochs}")
    print(f"📦 Batch size: {batch_size}")
    print(f"🖼️  Image size: {imgsz}")
    print(f"💻 Device: {device if device else 'auto'}")
    print("=" * 70)
    
    # Verify dataset exists
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"\n❌ Error: Dataset config not found: {data_yaml}")
        print("   Please run the following first:")
        print("   1. python download_crowdhuman.py")
        print("   2. Extract the zip files")
        print("   3. python convert_to_yolo.py")
        return None
    
    # Get model path
    model_path = get_model_path(model_name)
    
    # Generate run name if not provided
    if name is None:
        name = f"{model_name}_crowdhuman_e{epochs}_b{batch_size}"
    
    # Load model
    print(f"\n📥 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Training configuration optimized for crowd detection
    print("\n🏋️ Starting training...")
    print("   This may take several hours depending on your hardware.\n")
    
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device if device else None,
        workers=workers,
        project=project,
        name=name,
        resume=resume,
        pretrained=pretrained,
        
        # Augmentation settings good for crowd detection
        hsv_h=0.015,  # Hue augmentation
        hsv_s=0.7,    # Saturation augmentation  
        hsv_v=0.4,    # Value augmentation
        degrees=0.0,  # Rotation (disabled for upright people)
        translate=0.1,  # Translation
        scale=0.5,    # Scale augmentation
        shear=0.0,    # Shear (disabled)
        perspective=0.0,  # Perspective (disabled)
        flipud=0.0,   # Vertical flip (disabled for people)
        fliplr=0.5,   # Horizontal flip
        mosaic=1.0,   # Mosaic augmentation (good for crowds)
        mixup=0.1,    # Mixup augmentation
        
        # Training settings
        optimizer="auto",  # Auto-select optimizer
        lr0=0.01,     # Initial learning rate
        lrf=0.01,     # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Loss weights - slightly emphasize box loss for crowded scenes
        box=7.5,      # Box loss weight
        cls=0.5,      # Classification loss weight  
        dfl=1.5,      # Distribution focal loss weight
        
        # Other settings
        patience=50,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every N epochs
        val=True,
        plots=True,
        verbose=True,
    )
    
    print("\n" + "=" * 70)
    print("✅ Training complete!")
    print(f"📁 Results saved to: {project}/{name}")
    print(f"🏆 Best model: {project}/{name}/weights/best.pt")
    print("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO11 on CrowdHuman for improved crowd detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model selection
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolo11n",
        choices=["yolo11n", "yolo11s", "yolo11m", "yolo11l"],
        help="YOLO11 model variant to train"
    )
    
    # Dataset
    parser.add_argument(
        "--data", "-d",
        type=str,
        default="./datasets/crowdhuman_yolo/dataset.yaml",
        help="Path to dataset YAML configuration"
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=16,
        help="Batch size (reduce if OOM)"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device: '' (auto), 'cpu', '0', '0,1', 'mps' (Apple Silicon)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of data loading workers"
    )
    
    # Output
    parser.add_argument(
        "--project",
        type=str,
        default="runs/crowdhuman",
        help="Project directory for saving runs"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name (auto-generated if not specified)"
    )
    
    # Other options
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from scratch (not recommended)"
    )
    
    args = parser.parse_args()
    
    train(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        pretrained=not args.no_pretrained,
    )


if __name__ == "__main__":
    main()

