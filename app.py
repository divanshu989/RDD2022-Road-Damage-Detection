from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO
from pathlib import Path
from werkzeug.utils import secure_filename
import uuid
import csv
import json

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"
REPORT_FOLDER = BASE_DIR / "static" / "reports"

MODEL_PATH = BASE_DIR / "models" / "best.pt"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)
REPORT_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


# ============================================================
# ROAD DAMAGE CLASSES
# ============================================================

CLASS_NAMES = {
    0: "D00",
    1: "D01",
    2: "D10",
    3: "D11",
    4: "D20",
    5: "D40",
    6: "D43",
    7: "D44",
    8: "D50",
}

CLASS_DESCRIPTIONS = {
    "D00": "Longitudinal crack",
    "D01": "Longitudinal crack",
    "D10": "Transverse crack",
    "D11": "Transverse crack",
    "D20": "Alligator crack",
    "D40": "Pothole",
    "D43": "Crosswalk blur / road surface damage",
    "D44": "Other road surface damage",
    "D50": "Other damage",
}


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("RDD2022 ROAD DAMAGE DETECTION")
print("=" * 60)

print("Model:", MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model not found:\n{MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))

print("YOLO model loaded successfully.")
print("=" * 60)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# DETECTION
# ============================================================

@app.route("/detect", methods=["POST"])
def detect():

    try:

        # ----------------------------------------------------
        # CHECK FILE
        # ----------------------------------------------------

        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image was uploaded."
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No image was selected."
            }), 400


        # ----------------------------------------------------
        # SAVE UPLOADED IMAGE
        # ----------------------------------------------------

        original_name = secure_filename(file.filename)

        unique_id = uuid.uuid4().hex[:8]

        saved_name = (
            unique_id
            + "_"
            + original_name
        )

        upload_path = UPLOAD_FOLDER / saved_name

        file.save(str(upload_path))

        print()
        print("Processing image:", saved_name)


        # ----------------------------------------------------
        # RUN YOLO
        # ----------------------------------------------------

        results = model.predict(
            source=str(upload_path),
            conf=0.20,
            imgsz=640,
            verbose=False
        )


        result = results[0]


        # ----------------------------------------------------
        # SAVE ANNOTATED IMAGE
        # ----------------------------------------------------

        annotated = result.plot()

        result_name = (
            Path(saved_name).stem
            + "_result.jpg"
        )

        result_path = RESULT_FOLDER / result_name

        # result.plot() returns numpy image
        import cv2

        cv2.imwrite(
            str(result_path),
            annotated
        )


        # ----------------------------------------------------
        # EXTRACT DETECTIONS
        # ----------------------------------------------------

        detections = []

        if result.boxes is not None:

            boxes = result.boxes

            for i in range(len(boxes)):

                class_id = int(
                    boxes.cls[i].item()
                )

                confidence = float(
                    boxes.conf[i].item()
                )

                class_name = CLASS_NAMES.get(
                    class_id,
                    f"Class {class_id}"
                )

                description = CLASS_DESCRIPTIONS.get(
                    class_name,
                    "Road damage"
                )

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "description": description,
                    "confidence": round(
                        confidence * 100,
                        1
                    )
                })


        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        total = len(detections)

        unique_types = len(
            set(
                d["class_name"]
                for d in detections
            )
        )

        if total > 0:

            average_confidence = round(
                sum(
                    d["confidence"]
                    for d in detections
                ) / total,
                1
            )

        else:

            average_confidence = 0


        print(
            f"Detected {total} damage instances"
        )


        # ----------------------------------------------------
        # CSV REPORT
        # ----------------------------------------------------

        csv_name = (
            Path(saved_name).stem
            + "_report.csv"
        )

        csv_path = REPORT_FOLDER / csv_name

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "Class",
                "Description",
                "Confidence"
            ])

            for d in detections:

                writer.writerow([
                    d["class_name"],
                    d["description"],
                    d["confidence"]
                ])


        # ----------------------------------------------------
        # JSON DATA FOR REPORT
        # ----------------------------------------------------

        json_name = (
            Path(saved_name).stem
            + "_report.json"
        )

        json_path = REPORT_FOLDER / json_name

        report_data = {
            "image": original_name,
            "total_detections": total,
            "damage_types": unique_types,
            "average_confidence": average_confidence,
            "detections": detections
        }

        with open(
            json_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                report_data,
                f,
                indent=4
            )


        # ----------------------------------------------------
        # FINAL JSON RESPONSE
        # ----------------------------------------------------

        response = {
            "success": True,

            "total": total,

            "types": unique_types,

            "avg": average_confidence,

            "detections": detections,

            "original_url":
                "/static/uploads/" + saved_name,

            "result_url":
                "/static/results/" + result_name,

            "download_url":
                "/download/" + csv_name,

            "report_url":
                "/report/" + json_name
        }


        return jsonify(response)


    except Exception as e:

        print()
        print("=" * 60)
        print("DETECTION ERROR")
        print("=" * 60)
        print(str(e))
        print("=" * 60)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# DOWNLOAD CSV
# ============================================================

@app.route("/download/<filename>")
def download_report(filename):

    return send_from_directory(
        REPORT_FOLDER,
        filename,
        as_attachment=True
    )


# ============================================================
# REPORT
# ============================================================

@app.route("/report/<filename>")
def view_report(filename):

    report_path = REPORT_FOLDER / filename

    if not report_path.exists():

        return "Report not found", 404


    with open(
        report_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)


    html = """
    <!DOCTYPE html>

    <html>

    <head>

        <title>RDD2022 Detection Report</title>

        <style>

            body {
                font-family: Arial;
                background: #f3f6fa;
                padding: 40px;
                color: #172033;
            }

            .container {
                max-width: 1000px;
                margin: auto;
                background: white;
                padding: 35px;
                border-radius: 15px;
                box-shadow: 0 5px 20px rgba(0,0,0,.08);
            }

            h1 {
                color: #172554;
            }

            .metrics {
                display: flex;
                gap: 20px;
                margin: 25px 0;
            }

            .metric {
                flex: 1;
                background: #f8fafc;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }

            .number {
                font-size: 30px;
                font-weight: bold;
            }

            table {
                width: 100%;
                border-collapse: collapse;
            }

            th, td {
                padding: 12px;
                border-bottom: 1px solid #ddd;
                text-align: left;
            }

            th {
                background: #172554;
                color: white;
            }

        </style>

    </head>

    <body>

    <div class="container">

        <h1>RDD2022 Road Damage Detection Report</h1>

        <p>
            Image: <b>
    """

    html += data["image"]

    html += """
            </b>
        </p>

        <div class="metrics">

            <div class="metric">

                <div class="number">
    """

    html += str(
        data["total_detections"]
    )

    html += """
                </div>

                Total Detections

            </div>


            <div class="metric">

                <div class="number">
    """

    html += str(
        data["damage_types"]
    )

    html += """
                </div>

                Damage Types

            </div>


            <div class="metric">

                <div class="number">
    """

    html += str(
        data["average_confidence"]
    ) + "%"

    html += """
                </div>

                Average Confidence

            </div>

        </div>


        <h2>Detected Damage</h2>

        <table>

            <tr>

                <th>Class</th>

                <th>Description</th>

                <th>Confidence</th>

            </tr>
    """


    for d in data["detections"]:

        html += f"""
            <tr>

                <td>
                    <b>{d["class_name"]}</b>
                </td>

                <td>
                    {d["description"]}
                </td>

                <td>
                    {d["confidence"]}%
                </td>

            </tr>
        """


    html += """
        </table>

    </div>

    </body>

    </html>
    """


    return html


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )