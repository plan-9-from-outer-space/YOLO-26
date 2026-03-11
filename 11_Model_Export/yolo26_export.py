# Import All the Required Libraries
from ultralytics import YOLO

# Load the YOLO26 Model
model = YOLO("yolo26s.pt")

# Export the Model to ONNX Format
model.export(format="onnx")
model.export(format="openvino")
