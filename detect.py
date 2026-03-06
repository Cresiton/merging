from ultralytics import YOLO
import cv2
import numpy as np
from flask import Flask, request, jsonify, make_response
import base64

"""
Flask app that both:
- Serves the UI (index.html, script.js, style.css, etc.)
- Exposes a YOLOv8 detection API at POST /detect

Run with:
    python detect.py

Then open: http://127.0.0.1:5000
"""

# Serve static files (HTML, JS, CSS) from the project root
app = Flask(__name__, static_folder=".", static_url_path="")


@app.after_request
def add_cors_headers(response):
    """
    Allow browser pages (including file:// or other ports)
    to call the /detect endpoint without CORS issues.
    """
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response


@app.route("/", methods=["GET"])
def index():
    # Serve the main UI
    return app.send_static_file("index.html")

# Load model once at startup
model = YOLO("yolov8m.pt")


def decode_image_from_base64(data_url: str):
    """
    Accepts a data URL (e.g. 'data:image/jpeg;base64,...') or raw base64 string
    and returns a BGR OpenCV image.
    """
    # Strip off data URL prefix if present
    if "," in data_url:
        _, base64_str = data_url.split(",", 1)
    else:
        base64_str = data_url

    img_bytes = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


@app.route("/detect", methods=["POST", "OPTIONS"])
def detect():
    """
    Expects JSON: { "image": "<data-url-or-base64>" }
    Returns JSON: { "boxes": [ {x1,y1,x2,y2,label,conf} ] }
    Coordinates are normalized 0-1 relative to image width/height.
    """
    if request.method == "OPTIONS":
        # Preflight request for CORS
        return make_response("", 200)

    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image")

    if not image_b64:
        resp = make_response(jsonify({"error": "Missing 'image' field"}), 400)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    img = decode_image_from_base64(image_b64)
    if img is None:
        resp = make_response(jsonify({"error": "Unable to decode image"}), 400)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    h, w = img.shape[:2]

    # Run YOLO on the received frame
    results = model.predict(
        source=img,
        imgsz=320,
        conf=0.5,
        classes=[0, 2, 3, 5, 7],  # person, car, motorcycle, bus, truck, etc.
        device="cpu",
        verbose=False,
    )

    detections = []
    if results and len(results) > 0:
        r = results[0]
        names = r.names if hasattr(r, "names") else model.names

        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            detections.append(
                {
                    "x1": x1 / w,
                    "y1": y1 / h,
                    "x2": x2 / w,
                    "y2": y2 / h,
                    "label": names.get(cls_id, str(cls_id)),
                    "conf": conf,
                }
            )

    resp = make_response(jsonify({"boxes": detections}))
    # Allow calls from the local HTML file or other origins
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


if __name__ == "__main__":
    # Run on localhost:5000
    app.run(host="127.0.0.1", port=5000, debug=False)