#!/usr/bin/env python3
"""
Lightweight Sitting Posture Detection for Raspberry Pi 5
Optimized version with OpenCV only (no PyQt5)
"""

import cv2
import time
import sys
from pathlib import Path
import torch
import yolov5
import numpy as np


class PostureDetector:
    """Lightweight posture detection optimized for Raspberry Pi"""
    
    def __init__(self, model_path="./data/inference_models/small640.pt"):
        print("=" * 60)
        print("Initializing Raspberry Pi Posture Detector")
        print("=" * 60)
        
        # Model configuration
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            print(f"ERROR: Model not found at {self.model_path}")
            sys.exit(1)
        
        # Load YOLOv5 model
        print(f"Loading model: {self.model_path}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Fix for PyTorch 2.6+ weights_only security change
        # Monkey-patch torch.load to use weights_only=False for YOLOv5 models
        original_torch_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        torch.load = patched_load
        
        try:
            # Always use CPU on Raspberry Pi for stability
            self.model = yolov5.load(str(self.model_path), device='cpu')
            print("✓ Model loaded successfully (CPU mode)")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            sys.exit(1)
        
        # Optimized model settings for Raspberry Pi
        self.model.conf = 0.25      # Confidence threshold
        self.model.iou = 0.45       # IoU threshold
        self.model.classes = [0, 1] # Only show these classes
        self.model.agnostic = False
        self.model.multi_label = False
        self.model.max_det = 1      # Only detect one person
        self.model.amp = False      # Disable AMP for CPU stability
        
        # Visual settings
        self.box_thickness = 2
        self.text_font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_font_scale = 0.6
        self.text_thickness = 2
        
        # Colors (BGR format)
        self.color_good = (0, 255, 0)    # Green for good posture
        self.color_bad = (0, 0, 255)     # Red for bad posture
        self.color_text = (255, 255, 255) # White text
        self.color_bg = (0, 0, 0)        # Black background
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        print("=" * 60)
    
    def draw_info_box(self, frame, class_name, confidence, fps):
        """Draw information overlay on frame"""
        height, width = frame.shape[:2]
        
        # Prepare text
        if class_name == 0:
            status_text = "GOOD POSTURE"
            border_color = self.color_good
            text_color = self.color_good
        else:
            status_text = "BAD POSTURE"
            border_color = self.color_bad
            text_color = self.color_bad
        
        conf_text = f"Confidence: {confidence*100:.1f}%"
        fps_text = f"FPS: {fps:.1f}"
        
        # Draw thick border around entire frame
        cv2.rectangle(frame, (0, 0), (width-1, height-1), border_color, 8)
        
        # Info box dimensions
        box_height = 100
        box_width = 300
        margin = 10
        
        # Draw semi-transparent background box
        overlay = frame.copy()
        cv2.rectangle(overlay, (margin, margin), 
                     (margin + box_width, margin + box_height), 
                     self.color_bg, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        y_offset = margin + 30
        cv2.putText(frame, status_text, (margin + 10, y_offset),
                   self.text_font, self.text_font_scale, text_color, 
                   self.text_thickness, cv2.LINE_AA)
        
        y_offset += 30
        cv2.putText(frame, conf_text, (margin + 10, y_offset),
                   self.text_font, 0.5, self.color_text, 1, cv2.LINE_AA)
        
        y_offset += 25
        cv2.putText(frame, fps_text, (margin + 10, y_offset),
                   self.text_font, 0.5, self.color_text, 1, cv2.LINE_AA)
        
        return frame
    
    def draw_bounding_box(self, frame, bbox, class_name):
        """Draw bounding box around detected person"""
        x1, y1, x2, y2 = bbox
        color = self.color_good if class_name == 0 else self.color_bad
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)
        return frame
    
    def predict(self, frame):
        """Run inference on frame"""
        results = self.model(frame)
        return results
    
    def parse_results(self, results):
        """Extract detection info from YOLO results"""
        bbox = None
        class_name = None
        confidence = None
        
        try:
            results_dict = results.pandas().xyxy[0].to_dict(orient="records")
            if results_dict:
                result = results_dict[0]  # Get first (and only) detection
                confidence = result['confidence']
                class_name = int(result['class'])
                bbox = (
                    int(result['xmin']),
                    int(result['ymin']),
                    int(result['xmax']),
                    int(result['ymax'])
                )
        except Exception as e:
            print(f"Warning: Error parsing results: {e}")
        
        return bbox, class_name, confidence
    
    def update_fps(self):
        """Calculate FPS"""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
        return self.fps
    
    def run(self, camera_id=0, width=640, height=480):
        """Main detection loop"""
        print("\nStarting camera...")
        print(f"Camera ID: {camera_id}")
        print(f"Resolution: {width}x{height}")
        print("\nControls:")
        print("  - Press 'q' or ESC to quit")
        print("  - Press 's' to save current frame")
        print("=" * 60)
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"ERROR: Cannot open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✓ Camera opened: {actual_width}x{actual_height}")
        
        # Create window
        window_name = "Posture Detector - Raspberry Pi"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        frame_skip = 0  # For frame skipping to improve performance
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("ERROR: Failed to read frame")
                    break
                
                # Process every frame (no skipping by default)
                # If performance is too slow, uncomment frame skipping:
                # frame_skip += 1
                # if frame_skip % 2 != 0:  # Process every 2nd frame
                #     continue
                
                # Run inference
                results = self.predict(frame)
                
                # Parse results
                bbox, class_name, confidence = self.parse_results(results)
                
                # Draw on frame
                if bbox and class_name is not None and confidence:
                    frame = self.draw_bounding_box(frame, bbox, class_name)
                    frame = self.draw_info_box(frame, class_name, confidence, self.fps)
                else:
                    # No detection - show neutral state
                    height_f, width_f = frame.shape[:2]
                    cv2.rectangle(frame, (0, 0), (width_f-1, height_f-1), (128, 128, 128), 8)
                    cv2.putText(frame, "No detection", (20, 40),
                               self.text_font, 0.8, self.color_text, 2, cv2.LINE_AA)
                
                # Update and display FPS
                fps = self.update_fps()
                
                # Display frame
                cv2.imshow(window_name, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("\nExiting...")
                    break
                elif key == ord('s'):  # Save screenshot
                    filename = f"posture_capture_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"✓ Saved: {filename}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            # Cleanup
            print("Cleaning up...")
            cap.release()
            cv2.destroyAllWindows()
            print("✓ Done")


def main():
    """Entry point"""
    # Parse command line arguments
    camera_id = 0
    model_path = "./data/inference_models/small640.pt"
    width = 640
    height = 480
    
    if len(sys.argv) > 1:
        try:
            camera_id = int(sys.argv[1])
        except:
            model_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        try:
            width = int(sys.argv[2])
            height = int(sys.argv[3])
        except:
            pass
    
    # Create and run detector
    detector = PostureDetector(model_path)
    detector.run(camera_id=camera_id, width=width, height=height)


if __name__ == "__main__":
    main()
