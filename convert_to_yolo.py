#!/usr/bin/env python3
"""
Convert CrowdHuman ODGT annotations to YOLO format.

CrowdHuman annotation format (ODGT - JSON per line):
- fbox: [x, y, w, h] - full body bounding box
- vbox: [x, y, w, h] - visible region bounding box  
- hbox: [x, y, w, h] - head bounding box

YOLO format:
- class_id x_center y_center width height (all normalized 0-1)

Usage:
    python convert_to_yolo.py --raw-dir ./datasets/crowdhuman_raw --output-dir ./datasets/crowdhuman_yolo
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_odgt_line(line: str) -> dict:
    """Parse a single line from ODGT file."""
    return json.loads(line.strip())


def convert_bbox_to_yolo(bbox: list, img_width: int, img_height: int) -> tuple:
    """
    Convert [x, y, w, h] to YOLO format [x_center, y_center, width, height] normalized.
    
    Args:
        bbox: [x, y, width, height] in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        Tuple of (x_center, y_center, width, height) normalized to 0-1
    """
    x, y, w, h = bbox
    
    # Calculate center coordinates
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    
    # Normalize width and height
    width = w / img_width
    height = h / img_height
    
    # Clamp values to valid range
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return x_center, y_center, width, height


def process_annotation(annotation: dict, images_dir: Path, labels_dir: Path, 
                       box_type: str = "fbox", class_id: int = 0) -> dict:
    """
    Process a single image annotation and create YOLO label file.
    
    Args:
        annotation: Dict with image ID and gtboxes
        images_dir: Directory containing source images
        labels_dir: Directory to save YOLO label files
        box_type: Type of box to use (fbox, vbox, hbox)
        class_id: Class ID for person (default 0)
    
    Returns:
        Dict with processing stats
    """
    image_id = annotation["ID"]
    gtboxes = annotation.get("gtboxes", [])
    
    stats = {"image_id": image_id, "boxes": 0, "skipped": 0, "error": None}
    
    # Find the image file (could be .jpg or .png)
    image_path = None
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = images_dir / f"{image_id}{ext}"
        if candidate.exists():
            image_path = candidate
            break
    
    if image_path is None:
        stats["error"] = f"Image not found: {image_id}"
        return stats
    
    # Get image dimensions
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        stats["error"] = f"Error reading image {image_id}: {e}"
        return stats
    
    # Process bounding boxes
    yolo_lines = []
    
    for gtbox in gtboxes:
        # Skip if tagged as mask (crowd/reflection/etc)
        tag = gtbox.get("tag", "person")
        if tag == "mask":
            stats["skipped"] += 1
            continue
        
        # Skip if marked as ignore
        extra = gtbox.get("extra", {})
        if extra.get("ignore", 0) == 1:
            stats["skipped"] += 1
            continue
        
        # Get the specified box type
        bbox = gtbox.get(box_type)
        if bbox is None:
            stats["skipped"] += 1
            continue
        
        # Skip invalid boxes
        if bbox[2] <= 0 or bbox[3] <= 0:
            stats["skipped"] += 1
            continue
        
        # Convert to YOLO format
        x_center, y_center, width, height = convert_bbox_to_yolo(
            bbox, img_width, img_height
        )
        
        # Skip boxes that are too small or invalid after normalization
        if width < 0.001 or height < 0.001:
            stats["skipped"] += 1
            continue
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        stats["boxes"] += 1
    
    # Write label file
    label_path = labels_dir / f"{image_id}.txt"
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))
    
    return stats


def convert_split(odgt_path: Path, images_dir: Path, output_dir: Path, 
                  split_name: str, box_type: str = "fbox"):
    """
    Convert a dataset split (train/val) to YOLO format.
    
    Args:
        odgt_path: Path to annotation ODGT file
        images_dir: Directory containing images
        output_dir: Base output directory
        split_name: Name of split (train/val)
        box_type: Type of box to use (fbox, vbox, hbox)
    """
    print(f"\n📂 Processing {split_name} split...")
    print(f"   Annotations: {odgt_path}")
    print(f"   Images: {images_dir}")
    
    # Create output directories
    images_output = output_dir / "images" / split_name
    labels_output = output_dir / "labels" / split_name
    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)
    
    # Read annotations
    print(f"   Reading annotations...")
    with open(odgt_path, "r") as f:
        annotations = [parse_odgt_line(line) for line in f if line.strip()]
    
    print(f"   Found {len(annotations)} images to process")
    
    # Process annotations
    total_boxes = 0
    total_skipped = 0
    errors = []
    
    for annotation in tqdm(annotations, desc=f"   Converting {split_name}"):
        stats = process_annotation(annotation, images_dir, labels_output, box_type)
        
        if stats["error"]:
            errors.append(stats["error"])
            continue
        
        total_boxes += stats["boxes"]
        total_skipped += stats["skipped"]
        
        # Copy/symlink image to output directory
        image_id = annotation["ID"]
        for ext in [".jpg", ".jpeg", ".png"]:
            src_path = images_dir / f"{image_id}{ext}"
            if src_path.exists():
                dst_path = images_output / f"{image_id}{ext}"
                if not dst_path.exists():
                    # Use symlink for efficiency, or copy for portability
                    try:
                        dst_path.symlink_to(src_path.absolute())
                    except (OSError, NotImplementedError):
                        shutil.copy2(src_path, dst_path)
                break
    
    print(f"\n   ✅ {split_name} conversion complete:")
    print(f"      - Images processed: {len(annotations) - len(errors)}")
    print(f"      - Total boxes: {total_boxes}")
    print(f"      - Skipped boxes: {total_skipped}")
    if errors:
        print(f"      - Errors: {len(errors)}")
        for err in errors[:5]:
            print(f"        • {err}")
        if len(errors) > 5:
            print(f"        • ... and {len(errors) - 5} more")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CrowdHuman ODGT annotations to YOLO format"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="./datasets/crowdhuman_raw",
        help="Directory containing downloaded CrowdHuman files"
    )
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="./datasets/crowdhuman_yolo",
        help="Output directory for YOLO format dataset"
    )
    parser.add_argument(
        "--box-type",
        type=str,
        default="fbox",
        choices=["fbox", "vbox", "hbox"],
        help="Box type to use: fbox (full body), vbox (visible), hbox (head)"
    )
    parser.add_argument(
        "--images-subdir",
        type=str,
        default="Images",
        help="Subdirectory name containing extracted images (default: Images)"
    )
    
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("🔄 CrowdHuman to YOLO Format Converter")
    print("=" * 60)
    print(f"📁 Raw directory: {raw_dir.absolute()}")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print(f"📦 Box type: {args.box_type}")
    
    # Verify raw directory exists
    if not raw_dir.exists():
        print(f"\n❌ Error: Raw directory not found: {raw_dir}")
        print("   Please run download_crowdhuman.py first.")
        return
    
    # Check for annotation files
    train_odgt = raw_dir / "annotation_train.odgt"
    val_odgt = raw_dir / "annotation_val.odgt"
    
    if not train_odgt.exists():
        print(f"\n❌ Error: Training annotations not found: {train_odgt}")
        return
    
    if not val_odgt.exists():
        print(f"\n❌ Error: Validation annotations not found: {val_odgt}")
        return
    
    # Find images directory (after extraction)
    images_dir = raw_dir / args.images_subdir
    if not images_dir.exists():
        print(f"\n❌ Error: Images directory not found: {images_dir}")
        print("   Please extract the zip files first:")
        print(f"   cd {raw_dir} && unzip '*.zip'")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert training set
    convert_split(train_odgt, images_dir, output_dir, "train", args.box_type)
    
    # Convert validation set
    convert_split(val_odgt, images_dir, output_dir, "val", args.box_type)
    
    # Create dataset.yaml
    yaml_content = f"""# CrowdHuman Dataset for YOLO Training
# Converted from CrowdHuman ODGT format
# Box type used: {args.box_type}

path: {output_dir.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: person

# Dataset info
# - Training images: ~15,000
# - Validation images: ~4,370
# - Average persons per image: ~23
# - Source: https://huggingface.co/datasets/sshao0516/CrowdHuman
"""
    
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    print("\n" + "=" * 60)
    print("✅ Conversion complete!")
    print(f"📁 YOLO dataset saved to: {output_dir.absolute()}")
    print(f"📄 Dataset config: {yaml_path}")
    print("\n📝 Next step: Run train.py to start training")
    print(f"   python train.py --data {yaml_path}")


if __name__ == "__main__":
    main()

