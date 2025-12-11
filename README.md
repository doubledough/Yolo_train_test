# YOLO11 Training on CrowdHuman Dataset

Fine-tune YOLO11 models for improved person detection in crowded scenes using the [CrowdHuman dataset](https://huggingface.co/datasets/sshao0516/CrowdHuman).

## 🎯 Goal

Enhance YOLO11's ability to detect and track individuals in crowded environments. The CrowdHuman dataset contains:
- **15,000** training images
- **4,370** validation images
- **~470K** human instances
- **~23** persons per image on average
- High occlusion scenarios

## 📦 Project Structure

```
Yolo_train_test/
├── models/                     # Pretrained YOLO11 weights
│   ├── yolo11n.pt             # Nano (fastest, 2.6M params)
│   ├── yolo11s.pt             # Small (6.5M params)
│   ├── yolo11m.pt             # Medium (20.1M params)
│   └── yolo11l.pt             # Large (25.3M params)
├── Footage/                    # Your test videos
├── datasets/                   # Downloaded & converted datasets
│   ├── crowdhuman_raw/        # Raw downloaded files
│   └── crowdhuman_yolo/       # Converted YOLO format
├── runs/                       # Training outputs
├── download_crowdhuman.py      # Download dataset
├── convert_to_yolo.py          # Convert annotations
├── train.py                    # Training script
├── test_model.py               # Test/compare models
└── requirements.txt
```

## 🚀 Quick Start

### 1. Set Up Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download CrowdHuman Dataset

```bash
python download_crowdhuman.py
```

This downloads ~20GB of data to `./datasets/crowdhuman_raw/`:
- Training images (3 zip files)
- Validation images (1 zip file)
- Annotation files (.odgt format)

### 3. Extract Images

```bash
cd datasets/crowdhuman_raw
unzip "*.zip"
cd ../..
```

After extraction, images will be in `datasets/crowdhuman_raw/Images/`.

### 4. Convert to YOLO Format

```bash
python convert_to_yolo.py
```

Options:
- `--box-type fbox` - Full body bounding box (default, recommended)
- `--box-type vbox` - Visible region only
- `--box-type hbox` - Head only

### 5. Train Model

```bash
# Quick training with YOLO11n (fastest)
python train.py --model yolo11n --epochs 50

# More accurate with YOLO11s
python train.py --model yolo11s --epochs 100

# Best accuracy with YOLO11m
python train.py --model yolo11m --epochs 100 --batch 8
```

Training parameters:
- `--model`: yolo11n, yolo11s, yolo11m, or yolo11l
- `--epochs`: Number of training epochs (default: 50)
- `--batch`: Batch size (default: 16, reduce if OOM)
- `--imgsz`: Input image size (default: 640)
- `--device`: Device selection ('', 'cpu', '0', 'mps')

### 6. Test Your Model

```bash
# Test on your footage
python test_model.py --source Footage/entrance.mp4 --model runs/crowdhuman/yolo11n_crowdhuman/weights/best.pt

# Compare base vs fine-tuned
python test_model.py --source Footage/entrance.mp4 --compare
```

## 💻 Hardware Requirements

| Model | VRAM | Training Time (50 epochs) |
|-------|------|---------------------------|
| yolo11n | 4GB+ | ~4-6 hours |
| yolo11s | 6GB+ | ~6-10 hours |
| yolo11m | 8GB+ | ~12-18 hours |
| yolo11l | 12GB+ | ~20-30 hours |

*Times estimated for RTX 3080. Apple Silicon (MPS) will be slower.*

## 🍎 Apple Silicon (M1/M2/M3)

Training will automatically use MPS acceleration. To explicitly set:

```bash
python train.py --model yolo11n --device mps --batch 8
```

Reduce batch size if you encounter memory issues.

## 📊 Expected Results

Fine-tuning on CrowdHuman typically improves:
- **Detection recall** in crowded scenes (+10-20%)
- **Occlusion handling** - better at detecting partially hidden people
- **Small person detection** at a distance

Tradeoffs:
- May have slightly reduced performance on non-person classes
- Optimized specifically for person detection

## 📁 Output Files

After training, find your results in:

```
runs/crowdhuman/<run_name>/
├── weights/
│   ├── best.pt          # Best model (use this!)
│   └── last.pt          # Last checkpoint
├── results.csv          # Training metrics
├── confusion_matrix.png
├── F1_curve.png
├── PR_curve.png
└── ...
```

## 🔧 Advanced Configuration

### Custom Training Settings

Edit `train.py` to modify:
- Learning rate schedule
- Data augmentation
- Loss function weights
- Early stopping patience

### Using Your Own Dataset

1. Prepare images in `datasets/your_dataset/images/{train,val}/`
2. Create YOLO format labels in `datasets/your_dataset/labels/{train,val}/`
3. Create a `dataset.yaml` pointing to your data
4. Run: `python train.py --data path/to/your/dataset.yaml`

## 📚 References

- [CrowdHuman Paper](https://arxiv.org/pdf/1805.00123)
- [CrowdHuman HuggingFace](https://huggingface.co/datasets/sshao0516/CrowdHuman)
- [Ultralytics YOLO11](https://docs.ultralytics.com/)
- [YOLO Training Guide](https://docs.ultralytics.com/modes/train/)

## ❓ Troubleshooting

**Out of Memory (OOM)**
```bash
# Reduce batch size
python train.py --batch 8  # or even 4
```

**Slow Training on Mac**
```bash
# Ensure MPS is used
python train.py --device mps
```

**Dataset Not Found**
```bash
# Verify extraction
ls datasets/crowdhuman_raw/Images/
# Should show .jpg files
```

**Permission Denied (Downloads)**
```bash
# May need to login to HuggingFace
huggingface-cli login
```

## 📄 License

- CrowdHuman dataset: CC BY-NC 4.0 (non-commercial use only)
- YOLO11: AGPL-3.0 (or Ultralytics Enterprise License)

