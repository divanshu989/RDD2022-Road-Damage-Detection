from pathlib import Path
import shutil
import random
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE = PROJECT_ROOT / "dataset" / "yolo"
OUTPUT = PROJECT_ROOT / "dataset" / "yolo_balanced"

TRAIN_IMAGES = SOURCE / "images" / "train"
TRAIN_LABELS = SOURCE / "labels" / "train"

OUT_IMAGES = OUTPUT / "images" / "train"
OUT_LABELS = OUTPUT / "labels" / "train"

SEED = 42
random.seed(SEED)

# Target number of TRAINING IMAGES containing each class.
# We do not touch validation data.
TARGETS = {
    "0": 300,   # D00
    "1": 100,   # D01
    "2": 100,   # D10
    "3": 60,    # D11
    "4": 400,   # D20
    "5": 1224,  # D40 - keep original amount
    "6": 60,    # D43
    "7": 150,   # D44
    "8": 70,    # D50
}


def get_classes(label_file):
    classes = set()

    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()

        if parts:
            classes.add(parts[0])

    return classes


def main():
    if OUTPUT.exists():
        print(f"Removing existing balanced dataset: {OUTPUT}")
        shutil.rmtree(OUTPUT)

    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)

    image_by_class = defaultdict(list)

    label_files = list(TRAIN_LABELS.glob("*.txt"))

    print(f"Found {len(label_files)} training label files.")

    # ---------------------------------------------------------
    # Find which images contain which classes
    # ---------------------------------------------------------

    for label_file in label_files:

        classes = get_classes(label_file)

        image_name = label_file.stem

        image_file = TRAIN_IMAGES / f"{image_name}.jpg"

        if not image_file.exists():
            continue

        for cls in classes:
            image_by_class[cls].append(
                (image_file, label_file)
            )

    # ---------------------------------------------------------
    # Start with every original training image exactly once
    # ---------------------------------------------------------

    selected = {}

    for label_file in label_files:

        image_name = label_file.stem
        image_file = TRAIN_IMAGES / f"{image_name}.jpg"

        if image_file.exists():
            selected[image_name] = (
                image_file,
                label_file
            )

    print(f"Original training images: {len(selected)}")

    # ---------------------------------------------------------
    # Oversample minority classes
    # ---------------------------------------------------------

    for cls, target in TARGETS.items():

        candidates = image_by_class.get(cls, [])

        if not candidates:
            print(f"WARNING: no images found for class {cls}")
            continue

        current = sum(
            1
            for image_file, label_file in selected.values()
            if cls in get_classes(label_file)
        )

        if current >= target:
            print(
                f"Class {cls}: {current} images "
                f"(target {target}) - no oversampling needed"
            )
            continue

        needed = target - current

        print(
            f"Class {cls}: {current} images "
            f"-> adding {needed}"
        )

        for i in range(needed):

            image_file, label_file = random.choice(candidates)

            unique_name = (
                f"oversample_{cls}_{i:05d}_"
                f"{image_file.stem}"
            )

            selected[unique_name] = (
                image_file,
                label_file
            )

    # ---------------------------------------------------------
    # Copy files
    # ---------------------------------------------------------

    print()
    print("Creating balanced training dataset...")

    for new_name, (image_file, label_file) in selected.items():

        output_image = OUT_IMAGES / f"{new_name}.jpg"
        output_label = OUT_LABELS / f"{new_name}.txt"

        shutil.copy2(image_file, output_image)
        shutil.copy2(label_file, output_label)

    print()
    print("=" * 60)
    print("Balanced dataset created")
    print("=" * 60)
    print(f"Output: {OUTPUT}")
    print(f"Training images: {len(selected)}")
    print("=" * 60)


if __name__ == "__main__":
    main()