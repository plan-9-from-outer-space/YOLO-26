import cv2
import numpy as np

points = []

# Mouse callback function
def select_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point Selected: {x}, {y}")

# Load video
video_path = "input_videos/video2.mp4"
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
cap.release()

frame = cv2.resize(frame, (1080, 640))

cv2.namedWindow("Select Road Points")
cv2.setMouseCallback("Select Road Points", select_points)

print("\n✅ Click 4 points in this order:")
print("1. Top-Left")
print("2. Top-Right")
print("3. Bottom-Right")
print("4. Bottom-Left\n")

while True:
    temp = frame.copy()

    # Draw selected points
    for p in points:
        cv2.circle(temp, p, 6, (0, 255, 255), -1)

    cv2.imshow("Select Road Points", temp)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if len(points) == 4:
        print("\n✅ Final Road Points:")
        print(points)
        break

cv2.destroyAllWindows()
