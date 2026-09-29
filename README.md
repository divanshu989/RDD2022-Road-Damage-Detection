# RDD2022 Road Damage Detection

An AI-powered road damage detection system built using YOLO11, Flask, and the RDD2022 dataset.

The application allows users to upload road images and automatically detects and localizes different types of road damage using a trained YOLO object detection model.

## Project Overview

Road damage detection is useful for identifying road defects that may require maintenance or repair.

This project uses the RDD2022 dataset to train an object detection model capable of identifying nine road damage categories.

The trained model is integrated into a Flask web application that provides an easy-to-use interface for uploading images and viewing detection results.

## Key Features

- Image upload through a web interface
- Automatic road damage detection
- YOLO11 object detection model
- Bounding boxes around detected damage
- Damage class identification
- Confidence scores
- Detection statistics
- Original and detected image comparison
- Detection report generation
- Flask-based web application

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| YOLO11 | Object detection |
| Ultralytics | YOLO framework |
| PyTorch | Deep learning framework |
| Flask | Web application |
| OpenCV | Image processing |
| Pillow | Image handling |
| HTML/CSS | Frontend |

## Dataset

This project uses the RDD2022 Road Damage Detection Dataset.

The dataset contains road images with annotations for different types of road damage.

The original dataset is not included in this repository because of its size.

## Road Damage Classes

The model uses the following nine RDD2022 class labels:

| Class ID | Label |
|---|---|
| 0 | D00 |
| 1 | D01 |
| 2 | D10 |
| 3 | D11 |
| 4 | D20 |
| 5 | D40 |
| 6 | D43 |
| 7 | D44 |
| 8 | D50 |

## Model

The project uses a trained YOLO11 model.

The trained model is stored at:

```text
models/best.pt