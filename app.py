from flask import Flask, render_template, request
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import os
import uuid
import cv2

app = Flask(__name__)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    r"runs\detect\train-2\weights\best.pt"
)

RESULT_FOLDER = "static/results"

os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load trained YOLO model
model = YOLO(MODEL_PATH)


# --------------------------------------------------
# QUALITY INSPECTION RULES
# --------------------------------------------------

DEFECT_CLASSES = {
    "empty_bottle",
    "no_cap",
    "no_label",
    "crooked_cap"
}

PASS_CLASS = "Perfect_bottle"


# --------------------------------------------------
# HOME / INSPECTION
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None

    if request.method == "POST":

        # Check uploaded file
        if "image" not in request.files:

            error = "No image was uploaded."

            return render_template(
                "index.html",
                error=error
            )

        file = request.files["image"]

        if file.filename == "":

            error = "Please select an image."

            return render_template(
                "index.html",
                error=error
            )

        # --------------------------------------------------
        # SAVE UPLOADED IMAGE
        # --------------------------------------------------

        filename = secure_filename(file.filename)

        unique_filename = (
            f"{uuid.uuid4().hex}_{filename}"
        )

        image_path = os.path.join(
            RESULT_FOLDER,
            unique_filename
        )

        file.save(image_path)

        # --------------------------------------------------
        # RUN YOLO
        # --------------------------------------------------

        predictions = model.predict(
            source=image_path,
            conf=0.50,
            verbose=False
        )

        prediction = predictions[0]

        detections = []

        # --------------------------------------------------
        # EXTRACT DETECTIONS
        # --------------------------------------------------

        if prediction.boxes is not None:

            for box in prediction.boxes:

                class_id = int(box.cls[0])

                confidence = float(box.conf[0])

                class_name = model.names[class_id]

                detections.append({
                    "class": class_name,
                    "confidence": round(
                        confidence * 100,
                        2
                    )
                })

        # --------------------------------------------------
        # DETERMINE QUALITY
        # --------------------------------------------------

        defects = [
            detection
            for detection in detections
            if detection["class"] in DEFECT_CLASSES
        ]

        has_perfect_bottle = any(
            detection["class"] == PASS_CLASS
            for detection in detections
        )

        if defects:

            status = "FAIL"

        elif has_perfect_bottle:

            status = "PASS"

        else:

            status = "UNCERTAIN"

        # --------------------------------------------------
        # CREATE ANNOTATED IMAGE
        # --------------------------------------------------

        annotated_image = prediction.plot()

        result_filename = (
            f"result_{uuid.uuid4().hex}.jpg"
        )

        result_path = os.path.join(
            RESULT_FOLDER,
            result_filename
        )

        cv2.imwrite(
            result_path,
            annotated_image
        )

        # --------------------------------------------------
        # SEND RESULT TO HTML
        # --------------------------------------------------

        result = {
            "status": status,
            "detections": detections,
            "image": result_filename
        }

    return render_template(
        "index.html",
        result=result,
        error=error
    )


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )