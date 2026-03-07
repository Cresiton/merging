from __future__ import annotations

import logging
import os

# Suppress TensorFlow/Keras deprecation warnings (must be before TF is loaded)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import argparse
import base64
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, make_response, request
from ultralytics import YOLO

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


def _resolve_audio_path() -> str:
    """Resolve path to alarm audio file (searches for 'audio', audio.mpeg, audio.mp3)."""
    candidates = ["audio", "audio.mpeg", "audio.mp3"]
    for p in candidates:
        if os.path.isfile(p):
            return p
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    for name in candidates:
        for base in [script_dir, parent_dir]:
            p = os.path.join(base, name)
            if os.path.isfile(p):
                return p
    return ""


@app.route("/audio/alert")
def serve_alert_audio():
    """Serve alarm audio for breach alerts in the web app. Uses file named 'audio', audio.mpeg, or audio.mp3."""
    path = _resolve_audio_path()
    if not path:
        return "", 404
    from flask import send_file
    return send_file(path, mimetype="audio/mpeg", as_attachment=False)


@app.route("/save_clip", methods=["POST"])
def save_clip():
    """
    Save a single frame (image) from the web client when an alarm is triggered.
    Expects JSON: { "image": "<data-url-or-base64>", "session_id": "...", "timestamp": "<iso8601>" }
    Saves JPEG files under ./saved_clips and returns the relative URL.
    """
    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image")
    if not image_b64:
        return make_response(jsonify({"error": "Missing 'image' field"}), 400)

    img = decode_image_from_base64(image_b64)
    if img is None:
        return make_response(jsonify({"error": "Unable to decode image"}), 400)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    clips_dir = os.path.join(script_dir, "saved_clips")
    os.makedirs(clips_dir, exist_ok=True)

    ts = payload.get("timestamp") or datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    safe_ts = "".join(c for c in ts if c.isalnum() or c in ("-", "_"))
    filename = f"{safe_ts}.jpg"
    path = os.path.join(clips_dir, filename)

    try:
        cv2.imwrite(path, img)
    except Exception:
        return make_response(jsonify({"error": "Failed to write image"}), 500)

    url = f"/saved_clips/{filename}"
    return make_response(jsonify({"url": url}), 200)


@app.route("/saved_clips", methods=["GET"])
def list_saved_clips():
    """
    List saved clip image URLs for the Saved Clips dashboard.
    Returns: { "clips": ["/saved_clips/clip_....jpg", ...] }
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clips_dir = os.path.join(script_dir, "saved_clips")
    clips: List[str] = []
    if os.path.isdir(clips_dir):
        for name in sorted(os.listdir(clips_dir), reverse=True):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                clips.append(f"/saved_clips/{name}")
    return make_response(jsonify({"clips": clips}), 200)


# Load model once at startup
model = YOLO("yolov8m.pt")

# Session state for web detection (tracking across frames)
_session_state: Dict[str, dict] = {}


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


def _run_web_detection(
    img: np.ndarray,
    session_id: str,
    boundary: Optional[List[List[float]]],
    restricted_side: int,
) -> Tuple[List[dict], List[dict]]:
    """
    Run full detection pipeline (YOLO + pose + face + breach) for web.
    Returns (boxes, breaches).
    """
    h, w = img.shape[:2]
    state = _session_state.setdefault(session_id, {
        "next_track_id": 1,
        "last_boxes": [],
        "track_ids": [],
        "person_area": {},
        "ankle_history": defaultdict(lambda: deque(maxlen=15)),
        "last_status": {},
        "auth_tracks": {},
        "frame_idx": 0,
    })
    state["frame_idx"] += 1
    frame_idx = state["frame_idx"]

    # Fast path when boundary is set: skip pose/face, use smaller image for quicker breach detection
    boundary_active = boundary and len(boundary) >= 2 and restricted_side in (0, 1)
    imgsz = 256 if boundary_active else 320

    # YOLO person detection
    results = model.predict(
        source=img,
        imgsz=imgsz,
        conf=0.5,
        classes=[0],
        device="cpu",
        verbose=False,
    )

    boxes_xyxy: List[Tuple[int, int, int, int]] = []
    if results and len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            clamped = clamp_bbox_xyxy(int(x1), int(y1), int(x2), int(y2), w, h)
            if clamped:
                boxes_xyxy.append(clamped)

    # IoU-based tracking: match to previous boxes
    prev_boxes = state.get("last_boxes", [])
    prev_ids = state.get("track_ids", [])
    track_ids: List[int] = []
    used_prev = set()

    iou_thresh = 0.2 if boundary_active else 0.3  # Lower when boundary set to keep track across crossing
    for bbox in boxes_xyxy:
        best_iou, best_idx = 0.0, -1
        for i, prev in enumerate(prev_boxes):
            if i in used_prev:
                continue
            iou = _iou_xyxy(bbox, prev)
            if iou > best_iou and iou > iou_thresh:
                best_iou, best_idx = iou, i
        if best_idx >= 0:
            track_ids.append(prev_ids[best_idx])
            used_prev.add(best_idx)
        else:
            tid = state["next_track_id"]
            state["next_track_id"] += 1
            track_ids.append(tid)

    state["last_boxes"] = boxes_xyxy
    state["track_ids"] = track_ids

    # When boundary is active, skip pose and face to minimize latency for breach alarm
    pose_obj = None
    auth_embeddings: List[np.ndarray] = []
    if not boundary_active:
        try:
            import mediapipe.solutions.pose as mp_pose  # type: ignore[import-not-found]
            pose_obj = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception:
            pass
        try:
            from deepface import DeepFace
            for p in _get_authorized_paths():
                try:
                    faces = DeepFace.extract_faces(img_path=p, enforce_detection=True, detector_backend="opencv", align=True)
                    if faces:
                        emb = _embed_aligned_face(DeepFace, faces[0].get("face"), model_name="Facenet")
                        if emb is not None:
                            auth_embeddings.append(emb)
                except Exception:
                    pass
        except Exception:
            pass

    detections: List[dict] = []
    breaches: List[dict] = []
    enter_thresh, exit_thresh = 0.62, 0.68
    smooth_window = 9
    track_iou_thresh = 0.35

    for idx, (bbox, track_id) in enumerate(zip(boxes_xyxy, track_ids)):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        status = "Person"
        if pose_obj:
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                res = pose_obj.process(rgb)
                if res.pose_landmarks:
                    ankle = ankle_point_from_pose(None, res.pose_landmarks, (x1, y1), (x2 - x1, y2 - y1), 0.3)
                    if ankle:
                        state["ankle_history"][track_id].append(ankle)
            score = movement_score(state["ankle_history"][track_id])
            steps = max(1, len(state["ankle_history"][track_id]) - 1)
            avg = score / float(steps)
            bbox_h = max(1.0, float(y2 - y1))
            norm = avg / bbox_h
            prev_s = state["last_status"].get(track_id, "Standing")
            if prev_s == "Walking":
                status = "Walking" if norm >= 0.02 else "Standing"
            else:
                status = "Walking" if norm >= 0.04 else "Standing"
            state["last_status"][track_id] = status

        authorized = None
        if auth_embeddings:
            if track_id not in state["auth_tracks"]:
                state["auth_tracks"][track_id] = FaceTrack(
                    bbox_xyxy=bbox,
                    dist_history=deque(maxlen=smooth_window),
                    last_seen_frame=frame_idx,
                )
            ft = state["auth_tracks"][track_id]
            ft.bbox_xyxy = bbox
            ft.last_seen_frame = frame_idx
            if frame_idx % 2 == 0:
                try:
                    from deepface import DeepFace
                    crop = img[y1:y2, x1:x2]
                    faces = DeepFace.extract_faces(img_path=crop, enforce_detection=False, detector_backend="opencv", align=True)
                    if faces:
                        emb = _embed_aligned_face(DeepFace, faces[0].get("face"), model_name="Facenet")
                        if emb is not None:
                            best = min(_cosine_distance(emb, a) for a in auth_embeddings)
                            ft.dist_history.append(best)
                except Exception:
                    pass
            authorized = ft.smoothed_authorized(enter_thresh, exit_thresh)

        is_breach = False
        in_restricted_zone = None
        if boundary and len(boundary) >= 2 and restricted_side in (0, 1):
            p1, p2 = boundary[0], boundary[1]
            px1, py1 = p1[0] * w, p1[1] * h
            px2, py2 = p2[0] * w, p2[1] * h
            side = _point_side_of_line((px1, py1), (px2, py2), (cx, cy))
            in_restricted_zone = side == restricted_side
            prev_side = state["person_area"].get(track_id, side)
            state["person_area"][track_id] = side
            # Breach: person moved from restricted (green) → unrestricted (red) (same ID)
            if prev_side == restricted_side and side != restricted_side:
                is_breach = True
                breaches.append({"track_id": track_id})

        label = "Person"
        if authorized is not None:
            label = "AUTHORIZED" if authorized else "UNAUTHORIZED"
        if is_breach:
            label = "BREACH!"

        detections.append({
            "x1": x1 / w, "y1": y1 / h, "x2": x2 / w, "y2": y2 / h,
            "label": label,
            "status": status,
            "authorized": authorized,
            "breach": is_breach,
            "in_restricted_zone": in_restricted_zone,
            "conf": 0.95,
        })

    if pose_obj:
        pose_obj.close()

    # Cleanup stale tracks
    forget_after = 45
    for tid in list(state.get("person_area", {}).keys()):
        if tid not in track_ids:
            state["person_area"].pop(tid, None)
    for tid in list(state.get("auth_tracks", {}).keys()):
        ft = state["auth_tracks"][tid]
        if frame_idx - ft.last_seen_frame > forget_after:
            state["auth_tracks"].pop(tid, None)

    return detections, breaches


@app.route("/detect", methods=["POST", "OPTIONS"])
def detect():
    """
    Expects JSON: {
      "image": "<data-url-or-base64>",
      "session_id": "<optional>",
      "boundary": [[x1,y1],[x2,y2]] optional, normalized 0-1,
      "restricted_side": 0 or 1 optional
    }
    Returns: { "boxes": [...], "breaches": [...] }
    """
    if request.method == "OPTIONS":
        return make_response("", 200)

    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image")
    session_id = payload.get("session_id") or "default"
    boundary = payload.get("boundary")
    restricted_side = int(payload.get("restricted_side", 0))

    if not image_b64:
        resp = make_response(jsonify({"error": "Missing 'image' field"}), 400)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    img = decode_image_from_base64(image_b64)
    if img is None:
        resp = make_response(jsonify({"error": "Unable to decode image"}), 400)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    try:
        boxes, breaches = _run_web_detection(img, session_id, boundary, restricted_side)
    except Exception:
        # Fallback: YOLO-only so user always sees boxes when backend is reachable
        boxes, breaches = _run_web_detection_yolo_only(img)

    resp = make_response(jsonify({"boxes": boxes, "breaches": breaches}))
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


def _run_web_detection_yolo_only(img: np.ndarray) -> Tuple[List[dict], List[dict]]:
    """Minimal YOLO-only detection; used as fallback when full pipeline fails."""
    h, w = img.shape[:2]
    results = model.predict(
        source=img,
        imgsz=320,
        conf=0.5,
        classes=[0],
        device="cpu",
        verbose=False,
    )
    boxes: List[dict] = []
    if results and len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            clamped = clamp_bbox_xyxy(int(x1), int(y1), int(x2), int(y2), w, h)
            if clamped:
                x1, y1, x2, y2 = clamped
                boxes.append({
                    "x1": x1 / w, "y1": y1 / h, "x2": x2 / w, "y2": y2 / h,
                    "label": "Person",
                    "status": "Person",
                    "authorized": None,
                    "breach": False,
                    "conf": float(box.conf[0]),
                })
    return boxes, []


def clamp_bbox_xyxy(
    x1: int, y1: int, x2: int, y2: int, w: int, h: int
) -> Optional[Tuple[int, int, int, int]]:
    """Clamp bbox to image bounds; return None if invalid after clamp."""
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def ankle_point_from_pose(
    mp,
    pose_landmarks,
    crop_origin_xy: Tuple[int, int],
    crop_size_wh: Tuple[int, int],
    visibility_thresh: float = 0.5,
) -> Optional[Tuple[float, float]]:
    """Return ankle center (x,y) in full-frame pixel coordinates, or None if not reliable."""
    x0, y0 = crop_origin_xy
    cw, ch = crop_size_wh

    left_idx = mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value
    right_idx = mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value
    lmk = pose_landmarks.landmark

    la = lmk[left_idx]
    ra = lmk[right_idx]

    candidates = []
    if la.visibility >= visibility_thresh:
        candidates.append((x0 + float(la.x) * cw, y0 + float(la.y) * ch))
    if ra.visibility >= visibility_thresh:
        candidates.append((x0 + float(ra.x) * cw, y0 + float(ra.y) * ch))

    if not candidates:
        return None

    avg_x = sum(p[0] for p in candidates) / len(candidates)
    avg_y = sum(p[1] for p in candidates) / len(candidates)
    return avg_x, avg_y


def _point_side_of_line(
    line_p1: Tuple[float, float],
    line_p2: Tuple[float, float],
    point: Tuple[float, float],
) -> int:
    """Return 0 or 1 indicating which side of the line the point is on.
    Uses cross product: (p2-p1) x (point-p1). Sign determines side."""
    x1, y1 = line_p1
    x2, y2 = line_p2
    px, py = point
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    return 0 if cross >= 0 else 1


def _play_alarm_audio(audio_path: str) -> None:
    """Play alarm audio file (e.g. audio.mpeg). Non-blocking where possible."""
    path = os.path.abspath(audio_path)
    if not os.path.isfile(path):
        # Try relative to script dir
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, os.path.basename(audio_path))
    if not os.path.isfile(path):
        return
    try:
        import pygame  # type: ignore[import-not-found]
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
    except Exception:
        try:
            import subprocess
            import sys
            if sys.platform == "win32":
                os.startfile(path)
            else:
                subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


def movement_score(traj: Deque[Tuple[float, float]]) -> float:
    """Sum of consecutive displacements across the trajectory window."""
    if len(traj) < 2:
        return 0.0
    total = 0.0
    it = iter(traj)
    prev = next(it)
    for cur in it:
        dx = cur[0] - prev[0]
        dy = cur[1] - prev[1]
        total += float(np.hypot(dx, dy))
        prev = cur
    return total


def pose_mode() -> None:
    """Run YOLOv8 person tracking + (optional) MediaPipe Pose ankle motion.

    NOTE: The currently installed mediapipe build on this system does not
    expose the legacy `mediapipe.solutions.pose` API – only the new Tasks
    API is available. To avoid hard crashes, this function will fall back
    to a simple YOLO person tracker when the pose API is missing instead
    of raising ModuleNotFoundError.
    """
    from ultralytics import YOLO as YOLOPose

    try:
        import mediapipe.solutions.pose as mp_pose  # type: ignore[import-not-found]
        pose_available = True
    except ModuleNotFoundError:
        mp_pose = None  # type: ignore[assignment]
        pose_available = False

    model_path = "yolov8m.pt"
    imgsz = 320
    conf = 0.5
    device = "cpu"
    tracker_cfg = "bytetrack.yaml"

    pose_every_n_frames = 1
    history_len = 15
    pose_visibility_thresh = 0.3
    forget_after_frames = 45

    model_pose = YOLOPose(model_path)
    if pose_available:
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    else:
        pose = None

    ankle_history: Dict[int, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=history_len))
    last_seen_frame: Dict[int, int] = {}
    last_pose_frame: Dict[int, int] = {}
    last_status: Dict[int, str] = {}

    # Boundary line for Restricted / Non-Restricted areas (breach detection)
    boundary_pts: List[Tuple[int, int]] = []
    restricted_side: int = 0  # 0 or 1 - which side of the line is restricted
    boundary_ready: bool = False
    person_area: Dict[int, int] = {}  # track_id -> 0 or 1 (current side)
    person_prev_area: Dict[int, int] = {}  # track_id -> previous side (for breach detection)
    breach_triggered_ids: Set[int] = set()
    alarm_audio_path = "audio.mpeg"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isfile(alarm_audio_path):
        alarm_audio_path = os.path.join(script_dir, "audio.mpeg")
    if not os.path.isfile(alarm_audio_path):
        alarm_audio_path = os.path.join(os.path.dirname(script_dir), "audio.mpeg")

    # Mouse callback state for boundary drawing
    boundary_click_state: Dict[str, object] = {"pts": [], "awaiting_restricted": False}

    def _on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        state = boundary_click_state
        if state.get("awaiting_restricted") and len(state.get("pts", [])) >= 2:
            p1, p2 = state["pts"][0], state["pts"][1]
            side = _point_side_of_line((p1[0], p1[1]), (p2[0], p2[1]), (x, y))
            state["restricted_side"] = side
            state["awaiting_restricted"] = False
        else:
            state.setdefault("pts", []).append((x, y))
            if len(state["pts"]) > 2:
                state["pts"] = state["pts"][-2:]
            if len(state["pts"]) == 2:
                state["awaiting_restricted"] = True

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    window_name = "YOLOv8 + ByteTrack + Pose (Restricted Area Breach)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, _on_mouse)

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            h, w = frame.shape[:2]

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("b"):
                boundary_click_state.clear()
                boundary_click_state["pts"] = []
                boundary_click_state["awaiting_restricted"] = False
                boundary_pts = []
                boundary_ready = False
                person_area.clear()
                person_prev_area.clear()
                breach_triggered_ids.clear()

            # Sync boundary from mouse callback when user completed 3 clicks
            bc = boundary_click_state
            if (
                not bc.get("awaiting_restricted")
                and "restricted_side" in bc
                and len(bc.get("pts", [])) == 2
            ):
                boundary_pts = list(bc["pts"])
                restricted_side = bc["restricted_side"]
                boundary_ready = True

            results = model_pose.track(
                source=frame,
                persist=True,
                tracker=tracker_cfg,
                imgsz=imgsz,
                conf=conf,
                classes=[0],
                device=device,
                verbose=False,
            )

            annotated = frame.copy()

            # Draw boundary line and zone labels when boundary is set
            if boundary_ready and len(boundary_pts) == 2:
                p1, p2 = boundary_pts[0], boundary_pts[1]
                cv2.line(annotated, p1, p2, (0, 0, 255), 3)
                mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                length = max(1e-6, (dx * dx + dy * dy) ** 0.5)
                # Perpendicular offset: (-dy, dx) gives side 0
                off = int(60 * (-dy) / length), int(60 * dx / length)
                pos_side0 = (mx + off[0], my + off[1])
                pos_side1 = (mx - off[0], my - off[1])
                cv2.putText(
                    annotated,
                    "RESTRICTED",
                    (pos_side0[0] - 50, pos_side0[1]) if restricted_side == 0 else (pos_side1[0] - 50, pos_side1[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    annotated,
                    "NON-RESTRICTED",
                    (pos_side1[0] - 60, pos_side1[1]) if restricted_side == 0 else (pos_side0[0] - 60, pos_side0[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            elif len(boundary_click_state.get("pts", [])) > 0:
                pts = boundary_click_state["pts"]
                for i, p in enumerate(pts):
                    cv2.circle(annotated, p, 8, (0, 255, 255), -1)
                if len(pts) == 2:
                    cv2.line(annotated, pts[0], pts[1], (0, 255, 255), 2)
                    cv2.putText(
                        annotated,
                        "Click on RESTRICTED side",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
            else:
                cv2.putText(
                    annotated,
                    "Press B to mark boundary line (2 pts, then click restricted side)",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    2,
                )

            r0 = results[0]

            if r0.boxes is not None and r0.boxes.xyxy is not None and len(r0.boxes) > 0:
                xyxy = r0.boxes.xyxy.cpu().numpy()
                ids = None
                if r0.boxes.id is not None:
                    ids = r0.boxes.id.cpu().numpy().astype(int)

                if ids is not None and len(ids) == len(xyxy):
                    for (x1f, y1f, x2f, y2f), track_id in zip(xyxy, ids):
                        x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)
                        clamped = clamp_bbox_xyxy(x1, y1, x2, y2, w, h)
                        if clamped is None:
                            continue
                        x1, y1, x2, y2 = clamped

                        last_seen_frame[track_id] = frame_idx

                        # Breach detection: Restricted -> Non-Restricted (only when boundary is marked)
                        is_breach = False
                        color = (0, 255, 0)
                        if boundary_ready and len(boundary_pts) == 2:
                            cx = (x1 + x2) / 2.0
                            cy = (y1 + y2) / 2.0
                            p1, p2 = boundary_pts[0], boundary_pts[1]
                            side = _point_side_of_line(
                                (float(p1[0]), float(p1[1])),
                                (float(p2[0]), float(p2[1])),
                                (cx, cy),
                            )
                            prev_side = person_area.get(track_id, side)
                            person_prev_area[track_id] = prev_side
                            person_area[track_id] = side
                            # Breach: was in Restricted, now in Non-Restricted
                            if prev_side == restricted_side and side != restricted_side:
                                _play_alarm_audio(alarm_audio_path)
                                breach_triggered_ids.add(track_id)
                                is_breach = True

                        run_pose = (frame_idx - last_pose_frame.get(track_id, -10**9)) >= pose_every_n_frames

                        if run_pose and pose is not None:
                            crop_bgr = frame[y1:y2, x1:x2]
                            if crop_bgr.size > 0:
                                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                                pose_res = pose.process(crop_rgb)
                                last_pose_frame[track_id] = frame_idx

                                if pose_res.pose_landmarks is not None:
                                    ankle_xy = ankle_point_from_pose(
                                        None,
                                        pose_res.pose_landmarks,
                                        crop_origin_xy=(x1, y1),
                                        crop_size_wh=(x2 - x1, y2 - y1),
                                        visibility_thresh=pose_visibility_thresh,
                                    )
                                    if ankle_xy is not None:
                                        ankle_history[track_id].append(ankle_xy)

                        if pose_available:
                            score = movement_score(ankle_history[track_id])
                            steps = max(1, len(ankle_history[track_id]) - 1)
                            avg_step = score / float(steps)
                            bbox_h = max(1.0, float(y2 - y1))
                            norm_score = avg_step / bbox_h

                            walking_high = 0.04
                            walking_low = 0.02

                            prev_status = last_status.get(track_id, "Standing")
                            if prev_status == "Walking":
                                status = "Walking" if norm_score >= walking_low else "Standing"
                            else:
                                status = "Walking" if norm_score >= walking_high else "Standing"

                            last_status[track_id] = status
                        else:
                            status = "Person"

                        if is_breach:
                            status = "BREACH!"
                            color = (0, 0, 255)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        label = f"Person | {status}"

                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        y_text = max(0, y1 - 10)
                        cv2.rectangle(annotated, (x1, y_text - th - 6), (x1 + tw + 6, y_text), color, -1)
                        cv2.putText(
                            annotated,
                            label,
                            (x1 + 3, y_text - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 0),
                            2,
                            cv2.LINE_AA,
                        )

            to_forget = [tid for tid, seen in last_seen_frame.items() if (frame_idx - seen) > forget_after_frames]
            for tid in to_forget:
                last_seen_frame.pop(tid, None)
                last_pose_frame.pop(tid, None)
                last_status.pop(tid, None)
                ankle_history.pop(tid, None)
                person_area.pop(tid, None)
                person_prev_area.pop(tid, None)
                breach_triggered_ids.discard(tid)

            cv2.imshow(window_name, annotated)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if pose is not None:
            pose.close()


def _get_authorized_paths() -> List[str]:
    """Resolve paths to authorized face images (iniya.jpeg, pranavi.jpeg)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    names = ["iniya", "pranavi"]
    exts = [".jpeg", ".jpg", ".png"]
    candidates = [
        os.path.join(script_dir, base, name + ext)
        for name in names
        for base in ("authorized", ".")
        for ext in exts
    ]
    return [p for p in candidates if os.path.isfile(p)]


def _to_uint8_image(img: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 [0..255] for DeepFace represent()."""
    if img is None:
        raise ValueError("Empty image.")
    if img.dtype == np.uint8:
        return img
    arr = np.asarray(img)
    mx = float(np.max(arr)) if arr.size else 0.0
    if mx <= 1.5:
        arr = np.clip(arr * 255.0, 0.0, 255.0)
    else:
        arr = np.clip(arr, 0.0, 255.0)
    return arr.astype(np.uint8)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance; lower is more similar."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(1.0 - (np.dot(a, b) / denom))


def _embed_aligned_face(DeepFace, face_img: np.ndarray, model_name: str) -> Optional[np.ndarray]:
    """Compute an embedding from an already-cropped/aligned face image."""
    if face_img is None:
        return None
    face_img_u8 = _to_uint8_image(np.asarray(face_img))
    reps = DeepFace.represent(
        img_path=face_img_u8,
        model_name=model_name,
        enforce_detection=False,
        detector_backend="skip",
    )
    if isinstance(reps, list) and len(reps) > 0 and "embedding" in reps[0]:
        return np.asarray(reps[0]["embedding"], dtype=np.float32)
    return None


def _iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    return float(inter / (a_area + b_area - inter + 1e-12))


@dataclass
class FaceTrack:
    bbox_xyxy: Tuple[int, int, int, int]
    dist_history: Deque[float]
    last_seen_frame: int
    authorized: bool = False

    def smoothed_authorized(self, enter_thresh: float, exit_thresh: float) -> bool:
        if not self.dist_history:
            self.authorized = False
            return self.authorized

        median_dist = float(np.median(np.asarray(list(self.dist_history), dtype=np.float32)))
        if self.authorized:
            self.authorized = median_dist <= exit_thresh
        else:
            self.authorized = median_dist <= enter_thresh
        return self.authorized


def face_mode() -> None:
    """Run webcam face auth (green for authorized images, red otherwise)."""
    from deepface import DeepFace

    model_name = "Facenet"
    enter_thresh = 0.62
    exit_thresh = 0.68
    recognize_every_n_frames = 2
    smooth_window = 9
    forget_after_frames = 20
    track_iou_thresh = 0.35

    authorized_paths = _get_authorized_paths()
    if not authorized_paths:
        raise FileNotFoundError(
            "No authorized images found. Place iniya/pranavi images in 'authorized/' or the project root."
        )

    authorized_embeddings: List[np.ndarray] = []
    for p in authorized_paths:
        try:
            auth_faces = DeepFace.extract_faces(
                img_path=p,
                enforce_detection=True,
                detector_backend="opencv",
                align=True,
            )
        except Exception:
            auth_faces = []
        if not auth_faces:
            continue
        emb = _embed_aligned_face(DeepFace, auth_faces[0].get("face"), model_name=model_name)
        if emb is not None:
            authorized_embeddings.append(emb)

    if not authorized_embeddings:
        raise RuntimeError("Could not compute embeddings for authorized images.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    tracks: Dict[int, FaceTrack] = {}
    next_track_id = 1
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            try:
                faces = DeepFace.extract_faces(
                    img_path=frame,
                    enforce_detection=False,
                    detector_backend="opencv",
                    align=True,
                )
            except Exception:
                faces = []

            dead = [tid for tid, t in tracks.items() if (frame_idx - t.last_seen_frame) > forget_after_frames]
            for tid in dead:
                tracks.pop(tid, None)

            do_recognition = (frame_idx % recognize_every_n_frames) == 0

            for face_info in faces:
                fa = face_info.get("facial_area") or {}
                x = int(fa.get("x", 0))
                y = int(fa.get("y", 0))
                w_box = int(fa.get("w", 0))
                h_box = int(fa.get("h", 0))
                if w_box <= 0 or h_box <= 0:
                    continue

                bbox_xyxy = (x, y, x + w_box, y + h_box)

                best_tid: Optional[int] = None
                best_iou = 0.0
                for tid, t in tracks.items():
                    iou = _iou_xyxy(t.bbox_xyxy, bbox_xyxy)
                    if iou > best_iou:
                        best_iou = iou
                        best_tid = tid

                if best_tid is None or best_iou < track_iou_thresh:
                    tid = next_track_id
                    next_track_id += 1
                    tracks[tid] = FaceTrack(
                        bbox_xyxy=bbox_xyxy,
                        dist_history=deque(maxlen=smooth_window),
                        last_seen_frame=frame_idx,
                    )
                else:
                    tid = best_tid
                    tracks[tid].bbox_xyxy = bbox_xyxy
                    tracks[tid].last_seen_frame = frame_idx

                if do_recognition:
                    emb = _embed_aligned_face(DeepFace, face_info.get("face"), model_name=model_name)
                    if emb is not None:
                        best_dist = min(_cosine_distance(emb, auth) for auth in authorized_embeddings)
                        tracks[tid].dist_history.append(best_dist)

                is_auth = tracks[tid].smoothed_authorized(enter_thresh=enter_thresh, exit_thresh=exit_thresh)
                label = "AUTHORIZED" if is_auth else "UNAUTHORIZED"
                color = (0, 255, 0) if is_auth else (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def combined_mode() -> None:
    """Run YOLO person tracking + DeepFace authorized/unauthorized together.

    Note: This variant does NOT use pose estimation because the installed
    mediapipe build only exposes the new Tasks API, not the classic
    `solutions.pose` API. It still tracks people and performs face auth
    per tracked person.
    """
    from deepface import DeepFace
    from ultralytics import YOLO as YOLOCombined

    model_path = "yolov8m.pt"
    imgsz = 320
    conf = 0.5
    device = "cpu"
    tracker_cfg = "bytetrack.yaml"
    forget_after_frames = 45

    model_name = "Facenet"
    enter_thresh = 0.62
    exit_thresh = 0.68
    recognize_every_n_frames = 2
    smooth_window = 9

    authorized_paths = _get_authorized_paths()
    if not authorized_paths:
        raise FileNotFoundError(
            "No authorized images found. Place iniya/pranavi images in 'authorized/' or the project root."
        )

    authorized_embeddings: List[np.ndarray] = []
    for p in authorized_paths:
        try:
            auth_faces = DeepFace.extract_faces(
                img_path=p,
                enforce_detection=True,
                detector_backend="opencv",
                align=True,
            )
        except Exception:
            auth_faces = []
        if not auth_faces:
            continue
        emb = _embed_aligned_face(DeepFace, auth_faces[0].get("face"), model_name=model_name)
        if emb is not None:
            authorized_embeddings.append(emb)

    if not authorized_embeddings:
        raise RuntimeError("Could not compute embeddings for authorized images.")

    model_combined = YOLOCombined(model_path)

    last_seen_frame: Dict[int, int] = {}
    auth_tracks: Dict[int, FaceTrack] = {}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    window_name = "Combined: Pose + Face Auth"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            h, w = frame.shape[:2]

            results = model_combined.track(
                source=frame,
                persist=True,
                tracker=tracker_cfg,
                imgsz=imgsz,
                conf=conf,
                classes=[0],
                device=device,
                verbose=False,
            )

            annotated = frame.copy()
            r0 = results[0]

            to_forget = [tid for tid, seen in last_seen_frame.items() if (frame_idx - seen) > forget_after_frames]
            for tid in to_forget:
                last_seen_frame.pop(tid, None)
                auth_tracks.pop(tid, None)

            do_recognition = (frame_idx % recognize_every_n_frames) == 0

            if r0.boxes is not None and r0.boxes.xyxy is not None and len(r0.boxes) > 0:
                xyxy = r0.boxes.xyxy.cpu().numpy()
                ids = None
                if r0.boxes.id is not None:
                    ids = r0.boxes.id.cpu().numpy().astype(int)

                if ids is not None and len(ids) == len(xyxy):
                    for (x1f, y1f, x2f, y2f), track_id in zip(xyxy, ids):
                        x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)
                        clamped = clamp_bbox_xyxy(x1, y1, x2, y2, w, h)
                        if clamped is None:
                            continue
                        x1, y1, x2, y2 = clamped

                        last_seen_frame[track_id] = frame_idx

                        # No pose-based walking/standing here; keep label simple.
                        status = "Person"

                        if track_id not in auth_tracks:
                            auth_tracks[track_id] = FaceTrack(
                                bbox_xyxy=(x1, y1, x2, y2),
                                dist_history=deque(maxlen=smooth_window),
                                last_seen_frame=frame_idx,
                            )
                        else:
                            auth_tracks[track_id].bbox_xyxy = (x1, y1, x2, y2)
                            auth_tracks[track_id].last_seen_frame = frame_idx

                        if do_recognition:
                            crop_bgr = frame[y1:y2, x1:x2]
                            try:
                                faces = DeepFace.extract_faces(
                                    img_path=crop_bgr,
                                    enforce_detection=False,
                                    detector_backend="opencv",
                                    align=True,
                                )
                            except Exception:
                                faces = []
                            if faces:
                                emb = _embed_aligned_face(DeepFace, faces[0].get("face"), model_name=model_name)
                                if emb is not None:
                                    best_dist = min(_cosine_distance(emb, auth) for auth in authorized_embeddings)
                                    auth_tracks[track_id].dist_history.append(best_dist)

                        is_auth = auth_tracks[track_id].smoothed_authorized(
                            enter_thresh=enter_thresh,
                            exit_thresh=exit_thresh,
                        )
                        auth_label = "AUTHORIZED" if is_auth else "UNAUTHORIZED"
                        color = (0, 255, 0) if is_auth else (0, 0, 255)

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        label = f"{status} | {auth_label}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        y_text = max(0, y1 - 10)
                        cv2.rectangle(annotated, (x1, y_text - th - 6), (x1 + tw + 6, y_text), color, -1)
                        cv2.putText(
                            annotated,
                            label,
                            (x1 + 3, y_text - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 0),
                            2,
                            cv2.LINE_AA,
                        )

            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def tflite_mode() -> None:
    """Run the TFLite webcam classifier."""
    import tensorflow as tf

    with open("labels.txt", "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]

    interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img = cv2.resize(frame, (224, 224))
            img = img.astype(np.float32) / 255.0
            input_data = np.expand_dims(img, axis=0)

            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]["index"])

            predicted_index = int(np.argmax(output_data))
            predicted_label = labels[predicted_index]
            confidence = float(output_data[0][predicted_index])

            text = f"{predicted_label} ({confidence:.2f})"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("TFLite Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    """Entry point. Runs the web app at http://127.0.0.1:5000 with full detection (pose, face, breach)."""
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()