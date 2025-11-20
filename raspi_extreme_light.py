#!/usr/bin/env python3
"""
EXTREME LIGHTWEIGHT - Anti-Crash Version
Prevents power spikes by:
- Very slow inference (every 10 frames)
- Tiny resolution (160x120)
- Single thread only
- Gradual warmup
- Continuous monitoring
"""

import cv2
import sys
import gc
import time
from pathlib import Path
import torch


def main():
    # EXTREME low settings
    MODEL_PATH = "./data/inference_models/small640.pt"
    CAMERA_ID = 0
    WIDTH, HEIGHT = 160, 120  # VERY tiny
    SKIP_FRAMES = 10  # Process every 10th frame only!
    
    print("=" * 60)
    print("EXTREME LIGHTWEIGHT MODE - Anti-Crash")
    print("=" * 60)
    print("This version is designed to prevent power spike crashes")
    print("by running inference very slowly and carefully.")
    print("=" * 60)
    
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        sys.exit(1)
    
    # Extreme CPU limiting
    torch.set_num_threads(1)  # SINGLE thread only
    torch.set_num_interop_threads(1)
    
    print("\n[1/5] Loading model (60-90 seconds, please wait)...")
    print("      This is slow to prevent power spikes...")
    
    # Fix for PyTorch 2.6+
    original_torch_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    torch.load = patched_load
    
    # Lazy import
    import yolov5
    
    try:
        model = yolov5.load(MODEL_PATH, device='cpu')
        model.conf = 0.4  # Even higher threshold
        model.max_det = 1
        print("      ✓ Model loaded")
    except Exception as e:
        print(f"      ERROR: {e}")
        sys.exit(1)
    
    # Aggressive cleanup
    gc.collect()
    
    print("\n[2/5] Warming up model (prevents first-inference spike)...")
    # Run a dummy inference to "warm up" the model
    # This prevents a huge power spike on first real inference
    dummy_frame = torch.zeros((120, 160, 3), dtype=torch.uint8).numpy()
    try:
        _ = model(dummy_frame)
        time.sleep(2)  # Let system stabilize
        gc.collect()
        print("      ✓ Warmup complete")
    except:
        print("      ⚠ Warmup failed, continuing anyway...")
    
    print("\n[3/5] Opening camera...")
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 10)  # Very low FPS
    
    if not cap.isOpened():
        print(f"      ERROR: Cannot open camera {CAMERA_ID}")
        sys.exit(1)
    
    print("      ✓ Camera ready")
    
    print("\n[4/5] Testing camera capture...")
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            print(f"      ERROR: Cannot read from camera")
            sys.exit(1)
        time.sleep(0.1)
    print("      ✓ Camera working")
    
    print("\n[5/5] Starting main loop...")
    print("\nSettings:")
    print(f"  - Resolution: {WIDTH}x{HEIGHT} (TINY)")
    print(f"  - Inference: Every {SKIP_FRAMES} frames (~1 per second)")
    print(f"  - CPU threads: 1 (minimal load)")
    print(f"  - Press 'q' to quit")
    print("\nIf this crashes, the issue is definitely power supply!")
    print("=" * 60)
    print()
    
    cv2.namedWindow("Posture", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    last_class = None
    inference_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Lost camera connection")
                break
            
            frame_count += 1
            
            # Only process every Nth frame
            if frame_count % SKIP_FRAMES == 0:
                inference_count += 1
                print(f"Running inference #{inference_count}... ", end='', flush=True)
                
                try:
                    # Run inference
                    start_time = time.time()
                    results = model(frame)
                    inference_time = time.time() - start_time
                    
                    # Parse results
                    try:
                        detections = results.pandas().xyxy[0].to_dict(orient="records")
                        if detections:
                            det = detections[0]
                            last_class = int(det['class'])
                            conf = det['confidence']
                            print(f"Done ({inference_time:.2f}s) - Class: {last_class}, Conf: {conf:.2f}")
                        else:
                            print(f"Done ({inference_time:.2f}s) - No detection")
                    except Exception as e:
                        print(f"Parse error: {e}")
                    
                    # Cleanup
                    del results
                    
                    # Garbage collect after every inference
                    gc.collect()
                    
                    # Small delay to let system stabilize
                    time.sleep(0.1)
                
                except Exception as e:
                    print(f"INFERENCE ERROR: {e}")
                    # Continue anyway
            
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
                cv2.rectangle(frame, (0, 0), (w-1, h-1), color, 4)
                cv2.putText(frame, text, (5, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Show frame count
            cv2.putText(frame, f"Frame: {frame_count}", (5, h-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Display
            cv2.imshow("Posture", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        print("Done")


if __name__ == "__main__":
    main()
