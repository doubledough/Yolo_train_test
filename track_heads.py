#!/usr/bin/env python3
"""
Head Tracking with YOLO11 - Track individuals by their heads in video.

Features:
- Persistent ID tracking (re-identification)
- Works great with CCTV footage
- Heads are less occluded than full bodies
- Outputs annotated video with track IDs

Usage:
    # Track with trained head model
    python track_heads.py --source Footage/entrance.mp4 --model runs/crowdhuman/yolo11s_heads/weights/best.pt
    
    # Track with base model (person detection)
    python track_heads.py --source Footage/entrance.mp4
    
    # Process first 5 minutes only
    python track_heads.py --source Footage/entrance.mp4 --duration 300
    
    # Use different tracker
    python track_heads.py --source Footage/entrance.mp4 --tracker bytetrack
"""

import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO
import time


def track_video(
    source: str,
    model_path: str = None,
    output_dir: str = "runs/tracking",
    conf: float = 0.25,
    iou: float = 0.5,
    tracker: str = "botsort",
    show: bool = False,
    save: bool = True,
    duration: int = None,
    device: str = "auto",
):
    """
    Run head tracking on video.
    
    Args:
        source: Path to video file
        model_path: Path to trained model (None = use base yolo11n)
        output_dir: Directory to save results
        conf: Confidence threshold
        iou: IoU threshold for tracking
        tracker: Tracker type (botsort or bytetrack)
        show: Display video while processing
        save: Save annotated video
        duration: Process only first N seconds (None = full video)
        device: Device to use (auto, cpu, mps, 0)
    """
    import torch
    
    # Auto-detect device
    if device == "auto":
        if torch.cuda.is_available():
            device = "0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Load model
    if model_path is None:
        model_path = "yolo11n.pt"
        print("📦 Using base yolo11n (person detection)")
        class_filter = [0]  # Person class in COCO
    else:
        print(f"📦 Loading trained model: {model_path}")
        class_filter = None  # Use all classes from trained model
    
    model = YOLO(model_path)
    
    # Check source exists
    source_path = Path(source)
    if not source_path.exists():
        print(f"❌ Video not found: {source}")
        return
    
    # Get video info
    cap = cv2.VideoCapture(str(source_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Calculate frames to process
    if duration:
        max_frames = int(duration * fps)
        frames_to_process = min(max_frames, total_frames)
    else:
        frames_to_process = total_frames
    
    print("=" * 70)
    print("🎯 Head Tracking with YOLO")
    print("=" * 70)
    print(f"   Video: {source_path.name}")
    print(f"   Resolution: {width}x{height} @ {fps:.1f}fps")
    print(f"   Total frames: {total_frames} ({total_frames/fps:.1f}s)")
    print(f"   Processing: {frames_to_process} frames ({frames_to_process/fps:.1f}s)")
    print(f"   Tracker: {tracker}")
    print(f"   Device: {device}")
    print(f"   Confidence: {conf}")
    print("=" * 70)
    
    # Prepare output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run tracking
    print("\n🔄 Running tracking...")
    start_time = time.time()
    
    # Track using YOLO's built-in tracker
    results = model.track(
        source=str(source_path),
        conf=conf,
        iou=iou,
        tracker=f"{tracker}.yaml",
        device=device,
        stream=True,
        show=show,
        save=save,
        project=str(output_path),
        name=source_path.stem,
        exist_ok=True,
        classes=class_filter,
        verbose=False,
    )
    
    # Process results and collect statistics
    track_ids_seen = set()
    total_detections = 0
    frame_count = 0
    
    for result in results:
        frame_count += 1
        
        if result.boxes is not None and result.boxes.id is not None:
            # Get track IDs
            ids = result.boxes.id.cpu().numpy().astype(int)
            track_ids_seen.update(ids)
            total_detections += len(ids)
        
        # Progress update
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed
            print(f"   Frame {frame_count}/{frames_to_process} | "
                  f"{fps_actual:.1f} fps | "
                  f"Unique IDs: {len(track_ids_seen)}")
        
        # Stop if duration limit reached
        if duration and frame_count >= frames_to_process:
            break
    
    elapsed = time.time() - start_time
    
    print()
    print("=" * 70)
    print("✅ Tracking Complete!")
    print("=" * 70)
    print(f"   Frames processed: {frame_count}")
    print(f"   Processing time: {elapsed:.1f}s ({frame_count/elapsed:.1f} fps)")
    print(f"   Total detections: {total_detections}")
    print(f"   Unique track IDs: {len(track_ids_seen)}")
    print(f"   Average detections/frame: {total_detections/frame_count:.1f}")
    
    if save:
        output_video = output_path / source_path.stem / source_path.name
        print(f"\n📁 Output saved to: {output_video}")
    
    print("=" * 70)
    
    return {
        "frames": frame_count,
        "unique_ids": len(track_ids_seen),
        "total_detections": total_detections,
        "processing_time": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Track heads/people in video with persistent IDs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="Path to video file"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to trained model (default: yolo11n.pt)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="runs/tracking",
        help="Output directory"
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
        default=0.5,
        help="IoU threshold for tracking"
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="botsort",
        choices=["botsort", "bytetrack"],
        help="Tracker algorithm"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display video while processing"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output video"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Process only first N seconds"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, mps, 0 (CUDA)"
    )
    
    args = parser.parse_args()
    
    track_video(
        source=args.source,
        model_path=args.model,
        output_dir=args.output,
        conf=args.conf,
        iou=args.iou,
        tracker=args.tracker,
        show=args.show,
        save=not args.no_save,
        duration=args.duration,
        device=args.device,
    )


if __name__ == "__main__":
    main()

