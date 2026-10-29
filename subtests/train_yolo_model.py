from ultralytics import YOLO

# Load a model
# model = YOLO("yolo26n.pt")
model = YOLO("Data/best.pt")  # Continue training from previous model

# Train using NDJSON dataset
results = model.train(
    data="Data/eggboxes2.ndjson", 
    epochs=200, 
    imgsz=416, 
    batch=8, 
    device="cpu", 
    # workers=4, 
    patience=25, 
    # cache=True,
)