from ultralytics import YOLO
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

model = YOLO(os.path.join(current_dir, "models/9.29/last.pt"))

print("dir right")

model.train(
    data = "/kaggle/input/newdatayaml/military_dataset.yaml",
    epochs = 60,
    batch = 64,
    imgsz = 640
)

model = model.eval()

model.export(format="onnx")
