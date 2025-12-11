"""
YOLO11 Training on CrowdHuman - Google Colab Optimized

Copy this entire script into a Colab cell and run it.
Or upload this file and run: !python train_colab.py

INSTRUCTIONS:
1. Go to Google Colab (colab.research.google.com)
2. Runtime → Change runtime type → GPU (T4 or A100)
3. Copy/paste the cells below or upload this file

ESTIMATED COMPUTE TIME:
- 50% dataset, 25 epochs: ~1.5 hours (T4) / ~20 min (A100)
- 75% dataset, 25 epochs: ~2 hours (T4) / ~30 min (A100)  
- 100% dataset, 25 epochs: ~3 hours (T4) / ~45 min (A100)

TPU NOTE: PyTorch/YOLO has limited TPU support. Use GPU instead.
"""

# =============================================================================
# CONFIGURATION - Adjust these settings
# =============================================================================

TRAIN_FRACTION = 0.5      # 0.5 = 50% of training data (saves compute)
MODEL_SIZE = "yolo11n"    # Options: yolo11n, yolo11s, yolo11m
EPOCHS = 25
BATCH_SIZE = 16           # Reduce to 8 if OOM on T4
IMAGE_SIZE = 640
BOX_TYPE = "fbox"         # fbox=full body, vbox=visible, hbox=head

# =============================================================================
# CELL 1: Setup & Installation
# =============================================================================

def setup():
    """Install dependencies and check GPU."""
    import subprocess
    import sys
    
    print("=" * 60)
    print("🔧 Setting up environment...")
    print("=" * 60)
    
    # Install packages
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                          "ultralytics", "datasets", "huggingface_hub", "tqdm"])
    
    import torch
    print(f"\n✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU Memory: {gpu_mem:.1f} GB")
        
        # Adjust batch size based on GPU
        global BATCH_SIZE
        if gpu_mem < 12:  # T4 has ~15GB but leave room
            BATCH_SIZE = min(BATCH_SIZE, 16)
        print(f"✅ Using batch size: {BATCH_SIZE}")
    else:
        print("⚠️  No GPU detected! Training will be very slow.")
        print("   Go to Runtime → Change runtime type → GPU")

# =============================================================================
# CELL 2: Download CrowdHuman Dataset
# =============================================================================

def download_dataset():
    """Download CrowdHuman from HuggingFace."""
    import os
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm
    import zipfile
    
    print("\n" + "=" * 60)
    print("📦 Downloading CrowdHuman dataset...")
    print("=" * 60)
    
    # Create directories
    BASE_DIR = Path("/content/crowdhuman")
    RAW_DIR = BASE_DIR / "raw"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    REPO_ID = "sshao0516/CrowdHuman"
    
    # Always download annotations and validation
    files_to_download = [
        "annotation_train.odgt",
        "annotation_val.odgt", 
        "CrowdHuman_val.zip",
    ]
    
    # Download training zips based on fraction
    if TRAIN_FRACTION >= 0.25:
        files_to_download.append("CrowdHuman_train01.zip")
    if TRAIN_FRACTION >= 0.5:
        files_to_download.append("CrowdHuman_train02.zip")
    if TRAIN_FRACTION >= 0.75:
        files_to_download.append("CrowdHuman_train03.zip")
    
    print(f"   Training fraction: {int(TRAIN_FRACTION * 100)}%")
    print(f"   Files to download: {len(files_to_download)}")
    
    for filename in tqdm(files_to_download, desc="Downloading"):
        local_path = RAW_DIR / filename
        if local_path.exists():
            print(f"   ⏭️  {filename} (exists)")
            continue
            
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=RAW_DIR,
        )
    
    # Extract zips
    print("\n📂 Extracting images...")
    IMAGES_DIR = RAW_DIR / "Images"
    IMAGES_DIR.mkdir(exist_ok=True)
    
    for zip_file in RAW_DIR.glob("*.zip"):
        print(f"   Extracting {zip_file.name}...")
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(RAW_DIR)
    
    num_images = len(list(IMAGES_DIR.glob("*.jpg")))
    print(f"\n✅ Extracted {num_images} images")
    
    return RAW_DIR, IMAGES_DIR

# =============================================================================
# CELL 3: Convert to YOLO Format
# =============================================================================

def convert_to_yolo(raw_dir, images_dir):
    """Convert CrowdHuman ODGT to YOLO format."""
    import json
    import random
    from pathlib import Path
    from PIL import Image
    import shutil
    from tqdm import tqdm
    
    print("\n" + "=" * 60)
    print("🔄 Converting to YOLO format...")
    print("=" * 60)
    
    YOLO_DIR = Path("/content/crowdhuman/yolo")
    
    def parse_odgt(filepath):
        annotations = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    annotations.append(json.loads(line))
        return annotations
    
    def convert_bbox(bbox, img_w, img_h):
        x, y, w, h = bbox
        x_c = max(0, min(1, (x + w / 2) / img_w))
        y_c = max(0, min(1, (y + h / 2) / img_h))
        w_n = max(0, min(1, w / img_w))
        h_n = max(0, min(1, h / img_h))
        return x_c, y_c, w_n, h_n
    
    def process_split(annotations, split_name, fraction=1.0):
        images_out = YOLO_DIR / "images" / split_name
        labels_out = YOLO_DIR / "labels" / split_name
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        
        # Sample data
        if fraction < 1.0:
            random.seed(42)
            annotations = random.sample(annotations, int(len(annotations) * fraction))
        
        total_boxes = 0
        processed = 0
        
        for ann in tqdm(annotations, desc=f"Converting {split_name}"):
            image_id = ann["ID"]
            
            # Find image
            img_path = None
            for ext in [".jpg", ".png"]:
                candidate = images_dir / f"{image_id}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            
            if img_path is None:
                continue
            
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except:
                continue
            
            # Process boxes
            yolo_lines = []
            for gtbox in ann.get("gtboxes", []):
                if gtbox.get("tag") == "mask":
                    continue
                if gtbox.get("extra", {}).get("ignore", 0) == 1:
                    continue
                
                bbox = gtbox.get(BOX_TYPE)
                if bbox is None or bbox[2] <= 0 or bbox[3] <= 0:
                    continue
                
                x_c, y_c, w, h = convert_bbox(bbox, img_w, img_h)
                if w < 0.001 or h < 0.001:
                    continue
                
                yolo_lines.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
                total_boxes += 1
            
            # Write label
            with open(labels_out / f"{image_id}.txt", 'w') as f:
                f.write("\n".join(yolo_lines))
            
            # Copy image
            dst = images_out / img_path.name
            if not dst.exists():
                shutil.copy2(img_path, dst)
            
            processed += 1
        
        return processed, total_boxes
    
    # Load and process
    train_ann = parse_odgt(raw_dir / "annotation_train.odgt")
    val_ann = parse_odgt(raw_dir / "annotation_val.odgt")
    
    print(f"   Original: {len(train_ann)} train, {len(val_ann)} val")
    print(f"   Using {int(TRAIN_FRACTION * 100)}% of training data")
    
    train_imgs, train_boxes = process_split(train_ann, "train", TRAIN_FRACTION)
    val_imgs, val_boxes = process_split(val_ann, "val", 1.0)  # Full validation
    
    print(f"\n   Final: {train_imgs} train, {val_imgs} val")
    print(f"   Boxes: {train_boxes} train, {val_boxes} val")
    
    # Create dataset.yaml
    yaml_content = f"""# CrowdHuman Dataset for YOLO
path: {YOLO_DIR}
train: images/train
val: images/val

names:
  0: person
"""
    yaml_path = YOLO_DIR / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Dataset ready: {yaml_path}")
    return yaml_path, train_imgs

# =============================================================================
# CELL 4: Test Base Model
# =============================================================================

def test_base_model(yaml_path):
    """Evaluate pretrained model before fine-tuning."""
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print(f"📊 Testing BASE {MODEL_SIZE} (before training)...")
    print("=" * 60)
    
    model = YOLO(f"{MODEL_SIZE}.pt")
    results = model.val(data=str(yaml_path), split="val", verbose=True)
    
    print("\n" + "=" * 60)
    print("📈 BASE MODEL METRICS:")
    print("=" * 60)
    print(f"  mAP50:      {results.box.map50:.4f}")
    print(f"  mAP50-95:   {results.box.map:.4f}")
    print(f"  Precision:  {results.box.mp:.4f}")
    print(f"  Recall:     {results.box.mr:.4f}")
    print("=" * 60)
    
    return results

# =============================================================================
# CELL 5: Train Model
# =============================================================================

def train_model(yaml_path, train_imgs):
    """Fine-tune on CrowdHuman."""
    from ultralytics import YOLO
    import gc
    import torch
    
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print(f"🏋️ Training {MODEL_SIZE} on CrowdHuman")
    print("=" * 60)
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Training images: {train_imgs}")
    print("=" * 60)
    
    model = YOLO(f"{MODEL_SIZE}.pt")
    
    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        device=0,
        workers=2,
        project="/content/runs",
        name="crowdhuman_training",
        exist_ok=True,
        
        # Augmentation (mosaic disabled for dense images)
        mosaic=0.0,
        mixup=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
        
        # Training
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        
        # Logging
        verbose=True,
        plots=True,
        save=True,
        val=True,
    )
    
    print("\n✅ Training complete!")
    return results

# =============================================================================
# CELL 6: Evaluate & Compare
# =============================================================================

def evaluate_trained_model(yaml_path, base_results):
    """Compare base vs fine-tuned model."""
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("📊 Evaluating fine-tuned model...")
    print("=" * 60)
    
    best_model = YOLO("/content/runs/crowdhuman_training/weights/best.pt")
    fine_results = best_model.val(data=str(yaml_path), split="val", verbose=True)
    
    print("\n" + "=" * 60)
    print("📈 COMPARISON: Before vs After Training")
    print("=" * 60)
    print(f"{'Metric':<15} {'Base':<12} {'Fine-tuned':<12} {'Change':<10}")
    print("-" * 50)
    
    comparisons = [
        ("mAP50", base_results.box.map50, fine_results.box.map50),
        ("mAP50-95", base_results.box.map, fine_results.box.map),
        ("Precision", base_results.box.mp, fine_results.box.mp),
        ("Recall", base_results.box.mr, fine_results.box.mr),
    ]
    
    for name, base, fine in comparisons:
        change = fine - base
        sign = "+" if change >= 0 else ""
        print(f"{name:<15} {base:<12.4f} {fine:<12.4f} {sign}{change:.4f}")
    
    print("=" * 60)
    return fine_results

# =============================================================================
# CELL 7: Download Model
# =============================================================================

def download_model():
    """Package and download the trained model."""
    try:
        from google.colab import files
        import shutil
        from pathlib import Path
        
        print("\n📦 Packaging trained model...")
        
        model_dir = Path("/content/runs/crowdhuman_training/weights")
        output_zip = "/content/yolo11_crowdhuman_trained.zip"
        
        shutil.make_archive(
            output_zip.replace(".zip", ""),
            'zip',
            model_dir
        )
        
        print(f"✅ Model packaged: {output_zip}")
        print("\n📥 Downloading...")
        files.download(output_zip)
        
    except ImportError:
        print("⚠️  Not running in Colab. Model saved at:")
        print("   /content/runs/crowdhuman_training/weights/best.pt")

# =============================================================================
# MAIN: Run All Steps
# =============================================================================

def main():
    """Run the complete training pipeline."""
    print("=" * 60)
    print("🚀 YOLO11 CrowdHuman Training Pipeline")
    print("=" * 60)
    print(f"  Model: {MODEL_SIZE}")
    print(f"  Training data: {int(TRAIN_FRACTION * 100)}%")
    print(f"  Epochs: {EPOCHS}")
    print("=" * 60)
    
    # Step 1: Setup
    setup()
    
    # Step 2: Download dataset
    raw_dir, images_dir = download_dataset()
    
    # Step 3: Convert to YOLO format
    yaml_path, train_imgs = convert_to_yolo(raw_dir, images_dir)
    
    # Step 4: Test base model
    base_results = test_base_model(yaml_path)
    
    # Step 5: Train
    train_model(yaml_path, train_imgs)
    
    # Step 6: Evaluate
    evaluate_trained_model(yaml_path, base_results)
    
    # Step 7: Download
    download_model()
    
    print("\n" + "=" * 60)
    print("✅ ALL DONE!")
    print("=" * 60)
    print("Your trained model is at:")
    print("  /content/runs/crowdhuman_training/weights/best.pt")
    print("\nTo use it:")
    print("  from ultralytics import YOLO")
    print("  model = YOLO('best.pt')")
    print("  results = model.predict('image.jpg')")

if __name__ == "__main__":
    main()

