#!/usr/bin/env python3
"""
ULTRA-MINIMAL Posture Detector for Raspberry Pi 5
Bare bones version - maximum performance, minimum features
Only shows: Good/Bad posture with colored border
"""

import cv2
import sys
from pathlib import Path
import torch
import yolov5


def main():
    # Configuration
    MODEL_PATH = "./data/inference_models/small640.pt"
    CAMERA_ID = 0
    WIDTH, HEIGHT = 640, 360  # Lower resolution for better performance
    
    print("Loading model...")
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        sys.exit(1)
    
    # Fix for PyTorch 2.6+ weights_only security change
    # Monkey-patch torch.load to use weights_only=False for YOLOv5 models
    original_torch_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    torch.load = patched_load
    
    # Load model (CPU only)
    model = yolov5.load(MODEL_PATH, device='cpu')
    model.conf = 0.25
    model.max_det = 1
    print("✓ Model loaded")
    
    # Open camera
    print("Opening camera...")
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA_ID}")
        sys.exit(1)
    print("✓ Camera ready")
    
    # Create window
    cv2.namedWindow("Posture", cv2.WINDOW_NORMAL)
    print("\nRunning... Press 'q' to quit\n")
    
    # Main loop
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect
            results = model(frame)
            
            # Parse and display
            try:
                detections = results.pandas().xyxy[0].to_dict(orient="records")
                if detections:
                    det = detections[0]
                    class_id = int(det['class'])
                    conf = det['confidence']
                    
                    # Draw based on posture
                    h, w = frame.shape[:2]
                    if class_id == 0:  # Good posture
                        color = (0, 255, 0)  # Green
                        text = "GOOD POSTURE"
                    else:  # Bad posture
                        color = (0, 0, 255)  # Red
                        text = "BAD POSTURE"
                    
                    # Thick colored border
                    cv2.rectangle(frame, (0, 0), (w-1, h-1), color, 12)
                    
                    # Simple text overlay
                    cv2.putText(frame, text, (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    cv2.putText(frame, f"{conf*100:.0f}%", (20, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            except:
                pass  # No detection, just show frame
            
            # Display
            cv2.imshow("Posture", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        pass
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()
