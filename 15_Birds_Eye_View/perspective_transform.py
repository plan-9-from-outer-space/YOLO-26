import cv2
import numpy as np

cap = cv2.VideoCapture("input_videos/video2.mp4")

ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Could not read video")
    exit()

frame = cv2.resize(frame, (1080, 640))

# ✅ Road Points (Your Selected Points)
# points = np.float32([(425, 338), (649, 340), (1079, 598), (4, 586)])
points = np.float32([(284, 324), (786, 327), (1077, 635), (5, 634)])

# ✅ Draw trapezoid on original frame for verification
cv2.polylines(frame, [np.int32(points)], True, (0, 255, 0), 3)
cv2.imshow("Selected Road Area", frame)
cv2.waitKey(0)

# Destination rectangle (Bird Eye View)
dst = np.float32([
    [0, 0],
    [400, 0],
    [400, 400],
    [0, 400]
])

# Perspective transform matrix
M = cv2.getPerspectiveTransform(points, dst)

# Warp road into Bird Eye View
warped = cv2.warpPerspective(frame, M, (400, 400))

cv2.imshow("Bird Eye View Result", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
