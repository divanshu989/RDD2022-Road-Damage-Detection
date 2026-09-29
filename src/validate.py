from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL = PROJECT_ROOT / "runs" / "rdd2022_yolo" / "weights" / "best.pt"
DATASET = PROJECT_ROOT / "dataset" / "yolo" / "data.yaml"

model = YOLO(str(MODEL))

results = model.val(
    data=str(DATASET),
    imgsz=640,
    batch=8,
    project=str(PROJECT_ROOT / "runs"),
    name="rdd2022_validation",
)

print("\nValidation completed.")
print(f"Model: {MODEL}")