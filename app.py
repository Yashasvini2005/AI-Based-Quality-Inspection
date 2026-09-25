from flask import Flask, render_template, request
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import os
import uuid

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"runs\detect\train-2\weights\best.pt"

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# Classes used by the Cola bottle dataset
PASS_CLASS = "Perfect_bottle"

DEFECT_CLASSES = {
    "empty_bottle",
    "no_cap",
    "no_label",
    "crooked_cap"
}

# Load YOLO model once when Flask starts
model = YOLO(MODEL_PATH)

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ============================================================
# UPLOAD PAGE
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# ============================================================
# INSPECTION
# ============================================================

@app.route("/inspect", methods=["POST"])
def inspect():

    # Check if a file was submitted
    if "image" not in request.files:
        return render_template(
            "index.html",
            error="Please select a bottle image."
        )

    file = request.files["image"]

    # Check filename
    if file.filename == "":
        return render_template(
            "index.html",
            error="Please select an image before inspecting."
        )

    # Check file type
    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="Unsupported file type. Please upload JPG, JPEG, PNG or WEBP."
        )

    # Generate a unique filename
    original_name = secure_filename(file.filename)

    extension = original_name.rsplit(".", 1)[1].lower()

    unique_id = uuid.uuid4().hex

    upload_filename = f"{unique_id}.{extension}"

    upload_path = os.path.join(
        UPLOAD_FOLDER,
        upload_filename
    )

    file.save(upload_path)

    try:

        # ----------------------------------------------------
        # RUN YOLO
        # ----------------------------------------------------

        results = model.predict(
            source=upload_path,
            conf=0.50,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        detections = []

        # ----------------------------------------------------
        # READ DETECTIONS
        # ----------------------------------------------------

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0])

                confidence = float(box.conf[0])

                class_name = model.names[class_id]

                detections.append({
                    "class": class_name,
                    "confidence": round(confidence * 100, 1)
                })

        # ----------------------------------------------------
        # DETERMINE FINAL RESULT
        # ----------------------------------------------------

        if not detections:

            status = "UNCERTAIN"

            status_message = (
                "No known bottle condition was detected. "
                "Try another image with the bottle clearly visible."
            )

        elif any(
            detection["class"] in DEFECT_CLASSES
            for detection in detections
        ):

            status = "DEFECT"

            status_message = (
                "A bottle defect was detected. "
                "Please review the detected condition below."
            )

        elif any(
            detection["class"] == PASS_CLASS
            for detection in detections
        ):

            status = "PASS"

            status_message = (
                "The bottle was identified as a perfect bottle "
                "according to the trained model."
            )

        else:

            status = "UNCERTAIN"

            status_message = (
                "The model detected the bottle, but the result "
                "could not be classified as PASS or DEFECT."
            )

        # ----------------------------------------------------
        # CREATE ANNOTATED IMAGE
        # ----------------------------------------------------

        annotated_image = result.plot()

        result_filename = f"{unique_id}_result.jpg"

        result_path = os.path.join(
            RESULT_FOLDER,
            result_filename
        )

        # Save annotated image
        import cv2
        cv2.imwrite(result_path, annotated_image)

        # ----------------------------------------------------
        # DISPLAY DETECTION SUMMARY
        # ----------------------------------------------------

        primary_detection = None

        if detections:

            # Highest confidence detection
            primary_detection = max(
                detections,
                key=lambda x: x["confidence"]
            )

        return render_template(
            "result.html",

            status=status,

            status_message=status_message,

            detections=detections,

            primary_detection=primary_detection,

            original_filename=original_name,

            upload_image=url_for_static(
                "uploads",
                upload_filename
            ),

            result_image=url_for_static(
                "results",
                result_filename
            )
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=f"Inspection failed: {str(e)}"
        )


# ============================================================
# STATIC FILE URL HELPER
# ============================================================

def url_for_static(folder, filename):

    return f"/static/{folder}/{filename}"


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )