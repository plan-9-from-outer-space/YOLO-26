
# Import the Required Libraries
import cv2
import time
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo26n.pt")

# Video Input 
video_path = "input_video/video3.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")
    exit()

width, height = 1080, 640

fps_input = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc (*"mp4v")
output_path = "output_video/output_1.mp4"
# fps_input = 1.0
out = cv2.VideoWriter(output_path, fourcc, fps_input, (width, height))

previous_time = 0

# Heatmap Intensity Array
global_img_array = np.zeros((height, width), dtype=np.float32)

# COCO vehicle class ID's: 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
vehicle_ids = [2, 3, 5, 7]

# Main Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.resize(frame, (width, height))

    results = model(frame, conf=0.25)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])

        # Filter for the vehicle classes
        if cls_id not in vehicle_ids:
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Safe bounding box clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        # Draw the detection boxes and labels
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        label = f"{model.names[cls_id]}: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Center Point Intensity Method
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Add intensity to the center point
        global_img_array[cy, cx] += 50
    
    # Apply Gaussian Blur on float map
    heatmap_blurred = cv2.GaussianBlur(global_img_array, (25, 25), 0)

    # Normalize using cv2 (0 - 255)
    heatmap_norm = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to 8 bit
    heatmap_uint8 = heatmap_norm.astype(np.uint8)

    # Apply Color Map
    heatmap_img = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Overlay Heatmap on original frame
    super_imposed_img = cv2.addWeighted (
        src1 = heatmap_img, 
        alpha = 0.6, 
        src2 = frame, 
        beta = 0.4, 
        gamma = 0)
    
    # FPS Calculation
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(super_imposed_img, f"FPS: {fps:.1f}", (30, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Display the output
    out.write(super_imposed_img)
    cv2.imshow("Vehicle Intensity", super_imposed_img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

