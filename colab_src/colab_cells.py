"""
YOLO11 CrowdHuman Training - Colab Cells

Copy each cell (marked with # %% CELL) into a Google Colab notebook.
Run them in order from top to bottom.

BEFORE RUNNING: 
  Runtime → Change runtime type → GPU (T4 or A100)
"""

# %% CELL 1: Configuration
# ============================================================================
# 📋 CONFIGURATION - Adjust these settings!
# ============================================================================

TRAIN_FRACTION = 0.5      # 0.5 = 50% of training data (faster, saves compute)
MODEL_SIZE = "yolo11n"    # Options: yolo11n, yolo11s, yolo11m
EPOCHS = 25               # Number of training epochs
BATCH_SIZE = 16           # Reduce to 8 if you get OOM errors
IMAGE_SIZE = 640          # Input image size
BOX_TYPE = "fbox"         # fbox=full body, vbox=visible, hbox=head

print("=" * 60)
print("📋 Training Configuration")
print("=" * 60)
print(f"  Model: {MODEL_SIZE}")
print(f"  Training data: {int(TRAIN_FRACTION * 100)}% ({int(15000 * TRAIN_FRACTION)} images)")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print("=" * 60)


# %% CELL 2: Setup & Installation
# ============================================================================
# 🔧 Install dependencies and check GPU
# ============================================================================

!pip install -q ultralytics datasets huggingface_hub tqdm

import torch
print(f"\n✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU Memory: {gpu_mem:.1f} GB")
else:
    print("⚠️  No GPU detected! Go to Runtime → Change runtime type → GPU")

!nvidia-smi


# %% CELL 3: Download Dataset
# ============================================================================
# 📦 Download CrowdHuman from HuggingFace
# ============================================================================

import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm.notebook import tqdm
import zipfile

# Create directories
BASE_DIR = Path("/content/crowdhuman")
RAW_DIR = BASE_DIR / "raw"
YOLO_DIR = BASE_DIR / "yolo"
RAW_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = "sshao0516/CrowdHuman"

# Files to download based on fraction
files_to_download = [
    "annotation_train.odgt",
    "annotation_val.odgt",
    "CrowdHuman_val.zip",
]

if TRAIN_FRACTION >= 0.25:
    files_to_download.append("CrowdHuman_train01.zip")
if TRAIN_FRACTION >= 0.5:
    files_to_download.append("CrowdHuman_train02.zip")
if TRAIN_FRACTION >= 0.75:
    files_to_download.append("CrowdHuman_train03.zip")

print(f"📦 Downloading {len(files_to_download)} files...")
print(f"   Training fraction: {int(TRAIN_FRACTION * 100)}%")

for filename in tqdm(files_to_download, desc="Downloading"):
    local_path = RAW_DIR / filename
    if local_path.exists():
        print(f"   ⏭️  {filename} (exists)")
        continue
    hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset", local_dir=RAW_DIR)

print("\n✅ Download complete!")


# %% CELL 4: Extract Images
# ============================================================================
# 📂 Extract zip files
# ============================================================================

print("📂 Extracting images...")

IMAGES_DIR = RAW_DIR / "Images"
IMAGES_DIR.mkdir(exist_ok=True)

for zip_file in RAW_DIR.glob("*.zip"):
    print(f"   Extracting {zip_file.name}...")
    with zipfile.ZipFile(zip_file, 'r') as zf:
        zf.extractall(RAW_DIR)

num_images = len(list(IMAGES_DIR.glob("*.jpg")))
print(f"\n✅ Extracted {num_images} images")


# %% CELL 5: Convert to YOLO Format
# ============================================================================
# 🔄 Convert CrowdHuman ODGT to YOLO format
# ============================================================================

import json
import random
from PIL import Image
import shutil

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
    
    if fraction < 1.0:
        random.seed(42)
        annotations = random.sample(annotations, int(len(annotations) * fraction))
    
    total_boxes = 0
    processed = 0
    
    for ann in tqdm(annotations, desc=f"Converting {split_name}"):
        image_id = ann["ID"]
        
        img_path = None
        for ext in [".jpg", ".png"]:
            candidate = IMAGES_DIR / f"{image_id}{ext}"
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
        
        with open(labels_out / f"{image_id}.txt", 'w') as f:
            f.write("\n".join(yolo_lines))
        
        dst = images_out / img_path.name
        if not dst.exists():
            shutil.copy2(img_path, dst)
        
        processed += 1
    
    return processed, total_boxes

print("🔄 Converting to YOLO format...")

train_ann = parse_odgt(RAW_DIR / "annotation_train.odgt")
val_ann = parse_odgt(RAW_DIR / "annotation_val.odgt")

print(f"   Original: {len(train_ann)} train, {len(val_ann)} val")

train_imgs, train_boxes = process_split(train_ann, "train", TRAIN_FRACTION)
val_imgs, val_boxes = process_split(val_ann, "val", 1.0)

print(f"\n✅ Final dataset:")
print(f"   Train: {train_imgs} images, {train_boxes} boxes")
print(f"   Val: {val_imgs} images, {val_boxes} boxes")

# Create dataset.yaml
yaml_content = f"""path: {YOLO_DIR}
train: images/train
val: images/val

names:
  0: person
"""
yaml_path = YOLO_DIR / "dataset.yaml"
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"\n✅ Dataset ready: {yaml_path}")


# %% CELL 6: Test Base Model (Before Training)
# ============================================================================
# 📊 See how the pretrained model performs (before fine-tuning)
# ============================================================================

from ultralytics import YOLO

print(f"📊 Testing BASE {MODEL_SIZE} on CrowdHuman...")
print("   (Performance BEFORE fine-tuning)\n")

base_model = YOLO(f"{MODEL_SIZE}.pt")
base_results = base_model.val(data=str(yaml_path), split="val", verbose=True)

print("\n" + "=" * 60)
print("📈 BASE MODEL METRICS (before training):")
print("=" * 60)
print(f"  mAP50:      {base_results.box.map50:.4f}")
print(f"  mAP50-95:   {base_results.box.map:.4f}")
print(f"  Precision:  {base_results.box.mp:.4f}")
print(f"  Recall:     {base_results.box.mr:.4f}  ← This is what we want to improve!")
print("=" * 60)


# %% CELL 7: Train Model
# ============================================================================
# 🏋️ Fine-tune on CrowdHuman
# ============================================================================

import gc
import torch

# Clear memory
gc.collect()
torch.cuda.empty_cache()

print("=" * 60)
print(f"🏋️ Training {MODEL_SIZE} on CrowdHuman")
print("=" * 60)
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
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
    
    # Augmentation (mosaic disabled for ultra-dense images)
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
print(f"📁 Model saved to: /content/runs/crowdhuman_training/weights/best.pt")


# %% CELL 8: Evaluate & Compare
# ============================================================================
# 📊 Compare base model vs fine-tuned model
# ============================================================================

print("📊 Evaluating fine-tuned model...")

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
    emoji = "✅" if change > 0 else "⚠️"
    print(f"{name:<15} {base:<12.4f} {fine:<12.4f} {sign}{change:.4f} {emoji}")

print("=" * 60)


# %% CELL 9: Visualize Results
# ============================================================================
# 📊 Display training curves and metrics
# ============================================================================

from IPython.display import Image, display
from pathlib import Path

results_dir = Path("/content/runs/crowdhuman_training")

plots = ["results.png", "confusion_matrix.png", "F1_curve.png", "PR_curve.png"]

for plot in plots:
    plot_path = results_dir / plot
    if plot_path.exists():
        print(f"\n📊 {plot}:")
        display(Image(filename=str(plot_path), width=800))


# %% CELL 10: Download Trained Model
# ============================================================================
# 📥 Download your trained model
# ============================================================================

from google.colab import files
import shutil

print("📦 Packaging trained model...")

model_dir = Path("/content/runs/crowdhuman_training/weights")
output_zip = "/content/yolo11_crowdhuman_trained.zip"

shutil.make_archive(output_zip.replace(".zip", ""), 'zip', model_dir)

print(f"✅ Model packaged: {output_zip}")
print("\n📥 Downloading...")
files.download(output_zip)

print("\n" + "=" * 60)
print("✅ ALL DONE!")
print("=" * 60)
print("To use your model:")
print("  from ultralytics import YOLO")
print("  model = YOLO('best.pt')")
print("  results = model.predict('crowd_image.jpg')")

