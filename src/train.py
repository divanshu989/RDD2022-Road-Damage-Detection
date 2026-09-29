from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = PROJECT_ROOT / "dataset" / "yolo_balanced" / "data.yaml"

model = YOLO("yolo11n.pt")

model.train(
    data=str(DATASET),
    epochs=50,
    imgsz=640,
    batch=8,
    workers=2,
    project=str(PROJECT_ROOT / "runs"),
    name="rdd2022_balanced",
)

print("\nTraining completed.")
print(
    f"Results saved in: "
    f"{PROJECT_ROOT / 'runs' / 'rdd2022_balanced'}"
)