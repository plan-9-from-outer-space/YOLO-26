# Import the Required Libraries
import cv2
import time
import numpy as np 
from ultralytics import YOLO

# Load the YOLO model
model = YOLO ("yolo26n.pt")

width = 1080
height = 640

# Vehicles Class ID's
vehicle_ids = [2, 3, 5, 7]

video_path = "input_videos/video2.mp4"

cap = cv2.VideoCapture(video_path) 

# Output video settings
fps_input = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_videos/output.mp4", fourcc, fps_input, (width, height))

previous_time = 0

# BEV Size (Bird's Eye View)
BEV_W = 400
BEV_H = 400

# Perspective Points
src_points = np.float32([(285, 319), (787, 321), (1075, 621), (5, 618)])
dst_points = np.float32([(0, 0), (BEV_W, 0), (BEV_W, BEV_H), (0, BEV_H)])

# Perspective Transform Matrix
M = cv2.getPerspectiveTransform (src_points, dst_points)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (width, height))

    # Draw the road polygon
    cv2.polylines(frame, [np.int32(src_points)], True, (0, 255, 0), 2)
    
    # Empty the BEV each frame
    bev = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)

    # Call the YOLO model on the frame
    results = model(frame, conf=0.25)[0]

    for box in results.boxes:

        # Only Vehicles
        cls_id = int(box.cls[0])
        if cls_id not in vehicle_ids: continue

        conf = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{model.names[cls_id]}: {conf:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Bottom Center
        cx = int((x1 + x2)/2)
        cy = int(y2)

        # Convert point to BEV coordinates
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        bev_pt = cv2.perspectiveTransform(pt, M)
        bx, by = bev_pt[0][0]
        bx, by = int(bx), int(by)

        # Draw only if inside BEV canvas
        if  0 <= bx < BEV_W and 0 <= by < BEV_H:
            cv2.circle(bev, (bx, by), 5, (0, 0, 255), -1)
    
    # Overlay BEV on Frame
    # Resize BEV for Overlay
    bev_small = cv2.resize(bev, (300, 300))
    x_offset = 750
    y_offset = 60

    # Place BEV directly on Frame
    frame[y_offset:y_offset + 300, x_offset:x_offset + 300] = bev_small

    # Border around BEV
    cv2.rectangle(frame, (x_offset, y_offset), (x_offset + 300, y_offset + 300), (0, 255, 0), 2)

    # Heading above BEV
    cv2.putText(frame, "Bird's Eye View", (x_offset, y_offset - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    out.write(frame)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()            

