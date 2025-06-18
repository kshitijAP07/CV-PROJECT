# STEP 1: Download Dataset from Roboflow
from roboflow import Roboflow

# Initialize Roboflow with your API key
rf = Roboflow(api_key="rOnSwugln8OKvmb75pZb")

# Load your project and dataset (version 4)
project = rf.workspace().project("syrup_manufacturing")
dataset = project.version(4).download("yolov8")  # Download version 4 in yolov8 format

# STEP 2: Train the model using Ultralytics YOLOv8
from ultralytics import YOLO

# Load a YOLOv8 model (you can choose yolov8n.pt, yolov8s.pt, etc.)
model = YOLO("yolov8n.pt")  # You can use 'yolov8s.pt', 'yolov8m.pt', etc., depending on your needs

# Start training the model
model.train(
    data=dataset.location + "/data.yaml",  # Path to Roboflow's data.yaml
    epochs=50,  # You can change this number depending on how many epochs you want
    imgsz=640  # Set the image size for training
)
