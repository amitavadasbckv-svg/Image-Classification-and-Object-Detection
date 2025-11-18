# train_yolov8.py
from ultralytics import YOLO
import os

# Path to your data.yaml
DATA_YAML = "data.yaml"           # adjust path if needed
PRETRAINED = "yolov8n.pt"        # small model; change to yolov8s.pt, yolov8m.pt, etc.
EPOCHS = 50
IMGSZ = 640
BATCH = 16

# Load model (will download weights if not present)
model = YOLO(PRETRAINED)

# Train. This will create runs/detect/train (or runs/detect/expX) and save weights
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    workers=8,
    # save=True,  # default behaviour saves checkpoints and best.pt
)

print("Training finished. Results object:", results)
