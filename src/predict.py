from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL = PROJECT_ROOT / "runs" / "rdd2022_yolo" / "weights" / "best.pt"
SOURCE = PROJECT_ROOT / "dataset" / "yolo" / "images" / "val"

model = YOLO(str(MODEL))

results = model.predict(
    source=str(SOURCE),
    imgsz=640,
    conf=0.20,
    save=True,
    save_txt=True,
    save_conf=True,
    project=str(PROJECT_ROOT / "runs"),
    name="rdd2022_predictions",
)

print()
print("Prediction completed.")
print(
    f"Predictions saved to: "
    f"{PROJECT_ROOT / 'runs' / 'rdd2022_predictions'}"
)