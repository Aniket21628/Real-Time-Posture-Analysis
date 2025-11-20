#!/usr/bin/env python3
"""
ULTRA-LIGHTWEIGHT Posture Detector for Raspberry Pi 5
Extreme optimizations to prevent crashes:
- Process every 3rd frame only
- Lower resolution (320x240)
- Minimal processing
- Aggressive memory management
"""

import cv2
import sys
import gc
from pathlib import Path
import torch


def main():
    # ULTRA-LOW resource configuration
    MODEL_PATH = "./data/inference_models/small640.pt"
    CAMERA_ID = 0
    WIDTH, HEIGHT = 320, 240  # VERY low resolution for stability
    SKIP_FRAMES = 3  # Process every 3rd frame only
    
    print("=" * 50)
    print("ULTRA-LIGHTWEIGHT MODE")
    print("Optimized for stability on resource-constrained Pi")
    print("=" * 50)
    
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        sys.exit(1)
    
    # Disable PyTorch optimizations that use more memory
    torch.set_num_threads(2)  # Limit CPU threads
    
    print("Loading model (this may take 30-60 seconds)...")
    
    # Fix for PyTorch 2.6+ weights_only security change
    original_torch_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    torch.load = patched_load
    
    # Lazy import to save memory during startup
    import yolov5
    
    try:
        model = yolov5.load(MODEL_PATH, device='cpu')
        model.conf = 0.3  # Higher threshold = less processing
        model.max_det = 1
        print("✓ Model loaded")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)
    
    # Force garbage collection
    gc.collect()
    
    # Open camera with minimal settings
    print("Opening camera...")
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA_ID}")
        sys.exit(1)
    
    print("✓ Camera ready")
    print("\nRunning in ULTRA-LIGHT mode:")
    print(f"  - Resolution: {WIDTH}x{HEIGHT}")
    print(f"  - Processing every {SKIP_FRAMES} frames")
    print("  - Press 'q' to quit\n")
    
    cv2.namedWindow("Posture", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    last_result = None
    last_class = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Only process every Nth frame to reduce load
            if frame_count % SKIP_FRAMES == 0:
                try:
                    # Run inference
                    results = model(frame)
                    
                    # Parse results
                    try:
                        detections = results.pandas().xyxy[0].to_dict(orient="records")
                        if detections:
                            det = detections[0]
                            last_class = int(det['class'])
                            last_result = det
                    except:
                        pass
                    
                    # Clear results to free memory
                    del results
                    
                    # Force garbage collection every 30 frames
                    if frame_count % 30 == 0:
                        gc.collect()
                
                except Exception as e:
                    print(f"Warning: Inference error: {e}")
            
            # Draw based on last known result
            h, w = frame.shape[:2]
            if last_class is not None:
                if last_class == 0:  # Good posture
                    color = (0, 255, 0)
                    text = "GOOD"
                else:  # Bad posture
                    color = (0, 0, 255)
                    text = "BAD"
                
                # Thick border
                cv2.rectangle(frame, (0, 0), (w-1, h-1), color, 8)
                cv2.putText(frame, text, (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Display
            cv2.imshow("Posture", frame)
            
            # Check for quit (minimal delay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Done")


if __name__ == "__main__":
    main()
