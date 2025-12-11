# 🚀 YOLO11 CrowdHuman Training - Google Colab

Train YOLO11 on CrowdHuman dataset using Google Colab's free GPU.

## ⚡ Quick Start

### Option 1: Copy-Paste (Easiest)

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. **Runtime → Change runtime type → GPU (T4 or A100)**
4. Copy the cells from `colab_cells.py` into your notebook
5. Run each cell in order

### Option 2: Upload Script

1. Upload `train_colab.py` to Colab
2. Run: `!python train_colab.py`

## ⚙️ Configuration

Edit these variables at the top of `train_colab.py`:

```python
TRAIN_FRACTION = 0.5      # 0.5 = use 50% of training data
MODEL_SIZE = "yolo11n"    # yolo11n, yolo11s, or yolo11m
EPOCHS = 25               # Number of training epochs
BATCH_SIZE = 16           # Reduce to 8 if out of memory
IMAGE_SIZE = 640          # Input image size
BOX_TYPE = "fbox"         # fbox=full body, vbox=visible, hbox=head
```

## 🖥️ Hardware Recommendations

| Runtime | Speed | Memory | Cost |
|---------|-------|--------|------|
| **T4 GPU** | ~3 min/epoch | 15GB | Free |
| **A100 GPU** | ~45 sec/epoch | 40GB | Colab Pro |
| **TPU** | ❌ Not recommended | - | - |

> **Why not TPU?** PyTorch/YOLO has very limited TPU support. Google's TPUs are optimized for TensorFlow. Use GPU for best results.

## ⏱️ Estimated Training Time

| Dataset % | Epochs | T4 GPU | A100 GPU |
|-----------|--------|--------|----------|
| 25% | 25 | ~45 min | ~10 min |
| 50% | 25 | **~1.5 hours** | ~20 min |
| 75% | 25 | ~2 hours | ~30 min |
| 100% | 25 | ~3 hours | ~45 min |

## 📊 Expected Results

Fine-tuning on CrowdHuman typically improves:

| Metric | Base Model | After Training | Improvement |
|--------|------------|----------------|-------------|
| mAP50 | ~83% | ~88-92% | +5-9% |
| Recall | ~66% | ~80-85% | +14-19% |

The main improvement is **recall** - detecting more people in crowded scenes.

## 💾 Output Files

After training, your model will be at:
```
/content/runs/crowdhuman_training/
├── weights/
│   ├── best.pt          # Best model (use this!)
│   └── last.pt          # Last checkpoint
├── results.csv          # Training metrics
├── confusion_matrix.png
├── F1_curve.png
└── PR_curve.png
```

## 📥 Download Your Model

The script automatically downloads `yolo11_crowdhuman_trained.zip` containing your trained weights.

## 🔧 Troubleshooting

### Out of Memory (OOM)
```python
BATCH_SIZE = 8  # or even 4
```

### Slow Training
- Make sure GPU is enabled: Runtime → Change runtime type → GPU
- Use A100 if available (Colab Pro)

### Session Disconnects
Colab free tier disconnects after ~90 minutes of inactivity. For long training:
- Keep the browser tab active
- Consider Colab Pro for longer sessions
- Use `TRAIN_FRACTION = 0.5` for faster training

### Dataset Download Fails
If HuggingFace download fails:
```python
# Login to HuggingFace (may be required for large downloads)
!huggingface-cli login
```

## 📁 Files in This Folder

| File | Description |
|------|-------------|
| `train_colab.py` | Complete training script (run as one file) |
| `colab_cells.py` | Same code split into cells for copy-paste |
| `README.md` | This file |

## 🎯 Using Your Trained Model

After downloading `best.pt`:

```python
from ultralytics import YOLO

# Load fine-tuned model
model = YOLO("best.pt")

# Run inference
results = model.predict("crowd_image.jpg", conf=0.25)

# Or for video
results = model.predict("crowd_video.mp4", conf=0.25, save=True)
```

## 📚 References

- [CrowdHuman Dataset](https://huggingface.co/datasets/sshao0516/CrowdHuman)
- [CrowdHuman Paper](https://arxiv.org/pdf/1805.00123)
- [Ultralytics YOLO11](https://docs.ultralytics.com/)
- [Google Colab](https://colab.research.google.com/)

