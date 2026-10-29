from ultralytics import YOLO

# Load trained model
model = YOLO("Data/best.pt")

# Predict all images in folder
results = model.predict(
    source="Video/VideoBasket03",   # Folder of images to test
    save=True,                          # Save images with drawn bounding boxes
    iou=0.5,
    conf=0.1,                           # Minimum confidence threshold (adjustable 0-1)
    imgsz=640, 
    project="runs/predict",
    name="test_result",
)