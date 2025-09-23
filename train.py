from ultralytics import YOLO
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

model = YOLO(os.path.join(current_dir, "models/9.22/last.pt"))

print("dir right")

model.train(
    data = "/kaggle/input/military-assets-dataset-12-classes-yolo8-format/military_object_dataset/military_dataset.yaml",
    epochs = 120,
    batch = 100,
    imgsz = 640,
)
