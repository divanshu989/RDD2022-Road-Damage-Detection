from pathlib import Path
import xml.etree.ElementTree as ET
import random
import shutil

# -----------------------------
# Paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "dataset" / "original" / "train" / "images"
XML_DIR = PROJECT_ROOT / "dataset" / "original" / "train" / "annotations" / "xmls"

YOLO_DIR = PROJECT_ROOT / "dataset" / "yolo"

TRAIN_IMAGES = YOLO_DIR / "images" / "train"
VAL_IMAGES = YOLO_DIR / "images" / "val"

TRAIN_LABELS = YOLO_DIR / "labels" / "train"
VAL_LABELS = YOLO_DIR / "labels" / "val"


# -----------------------------
# RDD2022 classes
# -----------------------------

CLASSES = [
    "D00",
    "D01",
    "D10",
    "D11",
    "D20",
    "D40",
    "D43",
    "D44",
    "D50",
]

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASSES)
}


# -----------------------------
# Create directories
# -----------------------------

for directory in [
    TRAIN_IMAGES,
    VAL_IMAGES,
    TRAIN_LABELS,
    VAL_LABELS,
]:
    directory.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Convert XML → YOLO
# -----------------------------

def convert_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find("size")

    width = int(size.find("width").text)
    height = int(size.find("height").text)

    labels = []

    for obj in root.findall("object"):

        class_name = obj.find("name").text.strip()

        if class_name not in CLASS_TO_ID:
            print(f"WARNING: Unknown class {class_name}")
            continue

        bbox = obj.find("bndbox")

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # YOLO format
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height

        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        class_id = CLASS_TO_ID[class_name]

        labels.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

    return labels


# -----------------------------
# Get images
# -----------------------------

images = list(IMAGE_DIR.glob("*.jpg"))

print(f"Found {len(images)} images")


# -----------------------------
# Train / validation split
# -----------------------------

random.seed(42)
random.shuffle(images)

split_index = int(len(images) * 0.8)

train_images = images[:split_index]
val_images = images[split_index:]

print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")


# -----------------------------
# Process dataset
# -----------------------------

def process_images(image_list, image_destination, label_destination):

    converted = 0

    for image_file in image_list:

        xml_file = XML_DIR / f"{image_file.stem}.xml"

        if not xml_file.exists():
            print(f"WARNING: Missing XML: {xml_file.name}")
            continue

        labels = convert_xml(xml_file)

        # Copy image
        shutil.copy2(
            image_file,
            image_destination / image_file.name
        )

        # Create YOLO label
        label_file = label_destination / f"{image_file.stem}.txt"

        label_file.write_text(
            "\n".join(labels),
            encoding="utf-8"
        )

        converted += 1

    return converted


train_count = process_images(
    train_images,
    TRAIN_IMAGES,
    TRAIN_LABELS
)

val_count = process_images(
    val_images,
    VAL_IMAGES,
    VAL_LABELS
)


# -----------------------------
# Create data.yaml
# -----------------------------

yaml_content = f"""path: {YOLO_DIR.as_posix()}
train: images/train
val: images/val

nc: {len(CLASSES)}

names:
"""

for index, class_name in enumerate(CLASSES):
    yaml_content += f"  {index}: {class_name}\n"


(YOLO_DIR / "data.yaml").write_text(
    yaml_content,
    encoding="utf-8"
)


# -----------------------------
# Final output
# -----------------------------

print()
print("=" * 50)
print("RDD2022 YOLO conversion completed")
print("=" * 50)
print(f"Training images : {train_count}")
print(f"Validation images: {val_count}")
print(f"Classes          : {len(CLASSES)}")
print(f"YOLO dataset     : {YOLO_DIR}")
print("=" * 50)