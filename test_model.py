#!/usr/bin/env python3
"""
Test trained YOLO11 models on videos or images.

Compare performance between:
- Base YOLO11 (pretrained on COCO)
- Fine-tuned YOLO11 (trained on CrowdHuman)

Usage:
    # Test on a video with both base and fine-tuned models
    python test_model.py --video Footage/entrance.mp4
    
    # Test only the fine-tuned model
    python test_model.py --video Footage/entrance.mp4 --model runs/crowdhuman/yolo11n_crowdhuman/weights/best.pt
    
    # Compare base vs fine-tuned
    python test_model.py --video Footage/entrance.mp4 --compare
"""

import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO


def run_detection(
    source: str,
    model_path: str = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    show: bool = True,
    save: bool = True,
    save_dir: str = "runs/detect",
    classes: list = None,
):
    """
    Run YOLO detection on a video or image.
    
    Args:
        source: Path to video/image or camera index
        model_path: Path to model weights (default: base yolo11n)
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        show: Display results in real-time
        save: Save results to disk
        save_dir: Directory to save results
        classes: Filter to specific classes (None = all, [0] = person only)
    """
    # Load model
    if model_path is None:
        model_path = "yolo11n.pt"
        print(f"📦 Using base model: {model_path}")
    else:
        print(f"📦 Using model: {model_path}")
    
    model = YOLO(model_path)
    
    # For CrowdHuman fine-tuned model, we only have person class (0)
    # For base COCO model, person is also class 0
    if classes is None:
        classes = [0]  # Default to person detection only
    
    print(f"\n🎬 Running detection on: {source}")
    print(f"   Confidence threshold: {conf_threshold}")
    print(f"   IoU threshold: {iou_threshold}")
    print(f"   Classes: {classes}")
    
    # Run inference
    results = model.predict(
        source=source,
        conf=conf_threshold,
        iou=iou_threshold,
        show=show,
        save=save,
        project=save_dir,
        classes=classes,
        stream=True,  # Use streaming for videos
        verbose=False,
    )
    
    # Process results
    frame_count = 0
    total_detections = 0
    
    for result in results:
        frame_count += 1
        num_detections = len(result.boxes) if result.boxes is not None else 0
        total_detections += num_detections
        
        if frame_count % 30 == 0:  # Print every 30 frames
            print(f"   Frame {frame_count}: {num_detections} persons detected")
    
    print(f"\n✅ Detection complete!")
    print(f"   Total frames: {frame_count}")
    print(f"   Total detections: {total_detections}")
    print(f"   Average per frame: {total_detections / max(1, frame_count):.1f}")
    
    return results


def compare_models(
    source: str,
    base_model: str = "yolo11n.pt",
    finetuned_model: str = None,
    conf_threshold: float = 0.25,
    save_comparison: bool = True,
):
    """
    Compare base YOLO with fine-tuned model side by side.
    
    Args:
        source: Path to video/image
        base_model: Path to base YOLO model
        finetuned_model: Path to fine-tuned model
        conf_threshold: Confidence threshold
        save_comparison: Save comparison video
    """
    print("=" * 70)
    print("🔬 Model Comparison: Base YOLO vs CrowdHuman Fine-tuned")
    print("=" * 70)
    
    # Find fine-tuned model if not specified
    if finetuned_model is None:
        # Look for the most recent fine-tuned model
        runs_dir = Path("runs/crowdhuman")
        if runs_dir.exists():
            model_dirs = sorted(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            for d in model_dirs:
                best_pt = d / "weights" / "best.pt"
                if best_pt.exists():
                    finetuned_model = str(best_pt)
                    break
        
        if finetuned_model is None:
            print("❌ No fine-tuned model found. Please train a model first or specify --finetuned")
            return
    
    print(f"\n📦 Base model: {base_model}")
    print(f"📦 Fine-tuned model: {finetuned_model}")
    
    # Load models
    base = YOLO(base_model)
    finetuned = YOLO(finetuned_model)
    
    # Open video
    source_path = Path(source)
    if not source_path.exists():
        print(f"❌ Source not found: {source}")
        return
    
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        print(f"❌ Could not open video: {source}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup output video
    output_path = None
    out = None
    if save_comparison:
        output_path = Path("runs/comparison") / f"{source_path.stem}_comparison.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width * 2, height))
    
    # Process frames
    frame_idx = 0
    base_total = 0
    finetuned_total = 0
    
    print("\n🎬 Processing video... (Press 'q' to quit)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Run both models
        base_results = base.predict(frame, conf=conf_threshold, classes=[0], verbose=False)[0]
        finetuned_results = finetuned.predict(frame, conf=conf_threshold, classes=[0], verbose=False)[0]
        
        base_count = len(base_results.boxes) if base_results.boxes is not None else 0
        finetuned_count = len(finetuned_results.boxes) if finetuned_results.boxes is not None else 0
        
        base_total += base_count
        finetuned_total += finetuned_count
        
        # Draw results on frames
        base_frame = base_results.plot()
        finetuned_frame = finetuned_results.plot()
        
        # Add labels
        cv2.putText(base_frame, f"Base YOLO: {base_count} persons", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(finetuned_frame, f"CrowdHuman: {finetuned_count} persons", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Combine side by side
        combined = cv2.hconcat([base_frame, finetuned_frame])
        
        # Save and display
        if out is not None:
            out.write(combined)
        
        # Resize for display if too large
        display_frame = combined
        if combined.shape[1] > 1920:
            scale = 1920 / combined.shape[1]
            display_frame = cv2.resize(combined, None, fx=scale, fy=scale)
        
        cv2.imshow("Comparison: Base (Left) vs CrowdHuman (Right)", display_frame)
        
        if frame_idx % 30 == 0:
            print(f"   Frame {frame_idx}/{total_frames}: Base={base_count}, CrowdHuman={finetuned_count}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⏹️  Stopped by user")
            break
    
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 Comparison Summary")
    print("=" * 70)
    print(f"   Base YOLO total detections: {base_total}")
    print(f"   CrowdHuman fine-tuned total: {finetuned_total}")
    print(f"   Improvement: {((finetuned_total - base_total) / max(1, base_total)) * 100:+.1f}%")
    
    if output_path:
        print(f"\n📁 Comparison video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test YOLO models for crowd detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="Path to video, image, or directory"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to model weights (default: yolo11n.pt)"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )
    
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare base model with fine-tuned model"
    )
    
    parser.add_argument(
        "--finetuned",
        type=str,
        default=None,
        help="Path to fine-tuned model for comparison"
    )
    
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display results"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(
            source=args.source,
            finetuned_model=args.finetuned,
            conf_threshold=args.conf,
            save_comparison=not args.no_save,
        )
    else:
        run_detection(
            source=args.source,
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            show=not args.no_show,
            save=not args.no_save,
        )


if __name__ == "__main__":
    main()

