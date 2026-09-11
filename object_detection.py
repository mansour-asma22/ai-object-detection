"""
AI Object Detection - Modern Design
Clean, minimal style with subtle boxes and labels
"""

import cv2
import numpy as np
import time

from ultralytics import YOLO

# ==================== CONFIGURATION ====================

#  CHANGE THIS to your video file name
video_file = "  cafe.mp4"  # <--- Put your file name here

# Recording starts/stops with the R key
recording = False

# ==================== LOAD AI MODEL ====================

print(" Loading AI Model...")
model = YOLO("yolov8n.pt")
print(" Model loaded successfully!")

# ==================== OPEN VIDEO ====================

cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print(" Error: Could not open video file.")
    print(" Make sure the file exists and is named correctly.")
    exit()

# Get video info
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f" Video: {width}x{height}, {fps}fps, {total_frames} frames")

# ==================== SETUP VIDEO WRITER ====================

output_file = "output_" + video_file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

print(f" Output: {output_file}")

# ==================== MODERN DRAWING FUNCTIONS ====================

# Color palette - subtle modern colors
COLORS = {
    'person': (0, 200, 255),      # Cyan
    'car': (0, 255, 150),          # Mint green
    'truck': (255, 180, 0),        # Amber
    'bus': (255, 120, 50),         # Orange
    'bicycle': (200, 100, 255),    # Purple
    'motorcycle': (200, 100, 255), # Purple
    'traffic light': (255, 50, 100), # Rose
    'stop sign': (255, 50, 100),   # Rose
    'default': (180, 180, 200)     # Soft blue-grey
}

def get_color(class_name):
    """Get modern color for each class"""
    for key, color in COLORS.items():
        if key in class_name.lower():
            return color
    return COLORS['default']

def draw_modern_box(frame, x1, y1, x2, y2, label, confidence):
    """Draw a clean, modern bounding box with subtle style"""
    
    # Get color for this class
    color = get_color(label)
    
    # === Draw subtle box ===
    # Thin, clean line (2px instead of thick)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # === Small corner accents (modern touch) ===
    corner_length = min(15, (x2 - x1) // 4)
    
    # Top-left corner
    cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, 2)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, 2)
    
    # Top-right corner
    cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, 2)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, 2)
    
    # Bottom-left corner
    cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, 2)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, 2)
    
    # Bottom-right corner
    cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, 2)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, 2)
    
    # === Small label with background ===
    label_text = f"{label}"
    confidence_text = f"{int(confidence * 100)}%"
    full_text = f"{label_text} {confidence_text}"
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        full_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    
    # Background for label (subtle, not too dark)
    label_bg_x1 = x1
    label_bg_y1 = y1 - text_height - 10
    label_bg_x2 = x1 + text_width + 12
    label_bg_y2 = y1 + 4
    
    # Only draw label background if it fits above the box
    if label_bg_y1 > 0:
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (label_bg_x1, label_bg_y1),
                     (label_bg_x2, label_bg_y2),
                     (20, 22, 30), -1)  # Dark but not black
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border around label
        cv2.rectangle(overlay,
                     (label_bg_x1, label_bg_y1),
                     (label_bg_x2, label_bg_y2),
                     color, 1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Text
        cv2.putText(frame, full_text,
                   (x1 + 6, y1 - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        # If label doesn't fit above, put it below
        label_bg_y1 = y2 + 2
        label_bg_y2 = y2 + text_height + 12
        
        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (label_bg_x1, label_bg_y1),
                     (label_bg_x2, label_bg_y2),
                     (20, 22, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, full_text,
                   (x1 + 6, y2 + text_height + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# ==================== PROCESS VIDEO ====================

print("\n🎬 Processing... Press 'q' to quit, 's' for screenshot\n")

frame_count = 0
prev_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1

    # Run detection
    results = model(frame)
    
    # Create a copy for drawing
    annotated_frame = frame.copy()
    
    # Process detections
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            # Draw modern box with label
            draw_modern_box(annotated_frame, x1, y1, x2, y2, class_name, confidence)

    # Save frame only when recording
    if recording:
        out.write(annotated_frame)
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display FPS
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    # Show frame
    cv2.imshow("AI Street Detection", annotated_frame)

   # Keyboard controls
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or cv2.getWindowProperty("AI Street Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

    elif key == ord('s'):
        cv2.imwrite(f"screenshot_frame_{frame_count}.png", annotated_frame)
        print(f" Screenshot saved (frame {frame_count})")

    elif key == ord('r'):
        recording = not recording

        if recording:
            print(" Recording started")
        else:
            print(" Recording stopped")

# ==================== CLEANUP ====================

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n Done! Processed {frame_count} frames")
print(f" Saved to: {output_file}")