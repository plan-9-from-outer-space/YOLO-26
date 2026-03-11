
# Import All the Required Libraries
from ultralytics import YOLO

# Load the Pre-trained Model
model = YOLO ("yolo26x-obb.pt")

# Predict on an Image
result = model.predict ("resources/boats.jpg", conf=0.25, save=True, show_labels=False, show_conf=False)
# result = model.predict ("resources/video2.mp4", conf=0.25, save=True, show=True)

