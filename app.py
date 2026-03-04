import os
import cv2
import json
import time
import numpy as np
import base64
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, send_file
from ultralytics import YOLO
import torch
import google.generativeai as genai
import sqlite3

app = Flask(__name__)

# ==========================================
# 1. GEMINI AI SETUP
# ==========================================
GEMINI_API_KEY = "AIzaSyB6zPepWLHT5nx6GWcX-ZIwiUrHFTvwEkc"
genai.configure(api_key=GEMINI_API_KEY)

working_model_name = "models/gemini-1.5-flash"
try:
    print("Searching for available Google AI models...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            working_model_name = m.name
            break
    print(f"✅ Auto-selected AI model: {working_model_name}")
except Exception as e:
    print(f"⚠️ Could not fetch model list. Error: {e}")

ai_model = genai.GenerativeModel(working_model_name)
pest_memory_cache = {}


# ==========================================
# 2. DATABASE SETUP
# ==========================================
DB_PATH = "pest_history.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        pest_name TEXT,
        common_name TEXT,
        confidence REAL,
        count INTEGER,
        severity TEXT,
        image_path TEXT,
        crop TEXT DEFAULT 'Unknown',
        treatment TEXT
    )''')
    conn.commit()
    conn.close()
    print("✅ Database ready")


init_db()


def save_detection(pest_data, image_path, crop="Unknown"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO detections 
                 (timestamp, pest_name, common_name, confidence, count, severity, image_path, crop, treatment)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               pest_data["name"],
               pest_data.get("common_name", pest_data["name"]),
               pest_data["confidence"],
               pest_data["count"],
               pest_data.get("severity", "unknown"),
               image_path,
               crop,
               pest_data.get("chemical", "")))
    conn.commit()
    conn.close()


# ==========================================
# 3. AI FUNCTIONS
# ==========================================
def get_smart_pest_info(pest_name):
    if pest_name in pest_memory_cache:
        return pest_memory_cache[pest_name]

    print(f"✨ New pest detected: '{pest_name}'. Asking Gemini...")

    prompt = f"""
You are a practical agricultural advisor talking to a farmer. The pest "{pest_name}" was detected.
Be SHORT and DIRECT. No filler words. Respond ONLY with this JSON:

{{
  "common_name": "Simple common name of this pest",
  "damage": "In 1 short sentence, what damage does it do to the crop?",
  "identify": "In 1 short sentence, how to visually confirm it on the plant?",
  "chemical": "Best chemical pesticide name + dosage per liter of water",
  "organic": "Best organic remedy in 1 sentence",
  "quick_action": "The ONE most urgent thing the farmer should do RIGHT NOW in 1 sentence",
  "prevention": "1 sentence on how to prevent it next season",
  "severity": "low or medium or high",
  "crops_at_risk": "Top 3-4 crops this pest attacks, comma separated"
}}

Keep every value under 25 words. Be specific with chemical names and dosages.
"""

    try:
        response = ai_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )
        info = json.loads(response.text)
        pest_memory_cache[pest_name] = info
        print(f"✅ Got info for '{pest_name}'")
        return info
    except Exception as e:
        print(f"⚠️ AI Lookup Failed for {pest_name}: {e}")
        return _fallback_info(pest_name)


def _fallback_info(pest_name):
    return {
        "common_name": pest_name,
        "damage": "Could not retrieve info. Check internet connection.",
        "identify": "Consult an agricultural officer.",
        "chemical": "Consult local agri-store",
        "organic": "Neem oil spray (5ml per liter)",
        "quick_action": "Isolate affected plants immediately.",
        "prevention": "Practice crop rotation.",
        "severity": "unknown",
        "crops_at_risk": "Unknown"
    }


def verify_with_gemini(image, yolo_label):
    try:
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        prompt = f"""Look at this cropped image from a farm field. 
YOLO model detected this as "{yolo_label}".

Answer ONLY with this JSON:
{{
  "is_correct": true or false,
  "actual_pest": "What pest this actually is, or 'not a pest' if no pest visible",
  "confidence": "high" or "medium" or "low"
}}
"""
        response = ai_model.generate_content(
            [prompt, {"mime_type": "image/jpeg", "data": img_base64}],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"⚠️ Gemini verify failed: {e}")
        return {"is_correct": True, "actual_pest": yolo_label, "confidence": "unknown"}


def is_plant_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_percent = (np.sum(green_mask > 0) / green_mask.size) * 100
    lower_brown = np.array([8, 30, 30])
    upper_brown = np.array([25, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    brown_percent = (np.sum(brown_mask > 0) / brown_mask.size) * 100
    plant_score = green_percent + brown_percent
    return plant_score > 10, round(plant_score, 1)


# ==========================================
# 4. IMAGE ENHANCEMENT
# ==========================================
def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    return enhanced


def enhance_image_fast(image):
    """
    Lighter enhancement for live camera — faster than full enhance.
    Skips denoising (slow) and uses simpler contrast boost.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return enhanced


# ==========================================
# 5. MODEL LOADING
# ==========================================
os.makedirs(os.path.join("static", "results"), exist_ok=True)
os.makedirs(os.path.join("static", "uploads"), exist_ok=True)

print("Loading YOLO model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model = YOLO("best13C.pt")
    model.to(device)
    if device == "cuda":
        model.model.half()
    print(f"Model loaded successfully on {device}!")
except Exception as e:
    print(f"Error loading model: {e}")


# ==========================================
# 6. THREADED LIVE CAMERA SYSTEM
# ==========================================
class CameraDetector:
    """
    Separates camera reading from detection.
    Camera runs at full speed, detection runs in background thread.
    Result: smooth video feed + accurate detections.
    """

    def __init__(self):
        self.latest_frame = None
        self.latest_annotated = None
        self.latest_detections = []
        self.running = False
        self.lock = threading.Lock()
        self.det_lock = threading.Lock()
        self.cap = None
        self.confidence_threshold = 0.40  # Higher = only show confident detections

        # Stabilization: track detections over multiple frames
        self.detection_history = {}  # pest_name -> {count, total_conf, last_seen}
        self.stable_detections = []
        self.frame_number = 0
        self.CONFIRM_FRAMES = 3  # Pest must appear in 3+ detection cycles to show
        self.FORGET_FRAMES = 10  # Remove pest if not seen for 10 cycles

    def start(self):
        if self.running:
            return
        self.running = True
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Start detection thread
        self.det_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.det_thread.start()
        print("✅ Camera started")

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.detection_history = {}
        self.stable_detections = []
        self.frame_number = 0
        print("⏹ Camera stopped")

    def read_frame(self):
        """Read latest frame from camera (called by video feed generator)."""
        if not self.cap or not self.running:
            return None

        success, frame = self.cap.read()
        if not success:
            return None

        with self.lock:
            self.latest_frame = frame.copy()

        # Return annotated frame if available, otherwise raw frame
        with self.det_lock:
            if self.latest_annotated is not None:
                return self.latest_annotated
        return frame

    def _detection_loop(self):
        """Background thread: runs YOLO detection without blocking camera feed."""
        while self.running:
            frame = None
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()

            if frame is None:
                time.sleep(0.05)
                continue

            self.frame_number += 1

            # Fast enhancement
            enhanced = enhance_image_fast(frame)

            # Run YOLO
            results = model(
                enhanced,
                conf=self.confidence_threshold,
                verbose=False,
                imgsz=640,
                iou=0.5,
                agnostic_nms=True
            )

            current_frame_pests = {}
            annotated = frame.copy()

            for r in results:
                for box in r.boxes:
                    label = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])

                    if label not in current_frame_pests or conf > current_frame_pests[label]["conf"]:
                        current_frame_pests[label] = {
                            "conf": conf,
                            "box": box.xyxy[0].cpu().numpy()
                        }

            # ── STABILIZATION: Track pests across frames ──
            # Update seen pests
            for pest_name, data in current_frame_pests.items():
                if pest_name in self.detection_history:
                    h = self.detection_history[pest_name]
                    h["seen_count"] += 1
                    h["last_seen"] = self.frame_number
                    # Running average confidence
                    h["avg_conf"] = (h["avg_conf"] * 0.7) + (data["conf"] * 0.3)
                    h["best_conf"] = max(h["best_conf"], data["conf"])
                    h["latest_box"] = data["box"]
                    h["current_count"] = h.get("current_count", 0) + 1
                else:
                    self.detection_history[pest_name] = {
                        "seen_count": 1,
                        "last_seen": self.frame_number,
                        "avg_conf": data["conf"],
                        "best_conf": data["conf"],
                        "latest_box": data["box"],
                        "current_count": 1
                    }

            # Remove stale pests (not seen for FORGET_FRAMES cycles)
            stale = [name for name, h in self.detection_history.items()
                     if self.frame_number - h["last_seen"] > self.FORGET_FRAMES]
            for name in stale:
                del self.detection_history[name]

            # Build stable detections (only pests seen in CONFIRM_FRAMES+ cycles)
            confirmed_pests = []
            for pest_name, h in self.detection_history.items():
                if h["seen_count"] >= self.CONFIRM_FRAMES:
                    # Draw box on annotated frame
                    x1, y1, x2, y2 = map(int, h["latest_box"])
                    conf_pct = round(h["avg_conf"] * 100, 1)

                    # Color based on confidence
                    if conf_pct >= 70:
                        color = (0, 255, 0)      # Green = confident
                    elif conf_pct >= 50:
                        color = (0, 255, 255)    # Yellow = medium
                    else:
                        color = (0, 165, 255)    # Orange = lower

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                    # Label background
                    label_text = f"{pest_name} {conf_pct}%"
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(annotated, label_text, (x1 + 2, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    # Get pest info (cached, so no delay after first time)
                    info = get_smart_pest_info(pest_name)

                    confirmed_pests.append({
                        "name": pest_name,
                        "confidence": conf_pct,
                        "best_confidence": round(h["best_conf"] * 100, 1),
                        "count": h.get("current_count", 1),
                        "frames_seen": h["seen_count"],
                        "common_name": info.get("common_name", pest_name),
                        "damage": info.get("damage", ""),
                        "identify": info.get("identify", ""),
                        "chemical": info.get("chemical", "Consult expert"),
                        "organic": info.get("organic", ""),
                        "quick_action": info.get("quick_action", ""),
                        "prevention": info.get("prevention", ""),
                        "severity": info.get("severity", "unknown"),
                        "crops_at_risk": info.get("crops_at_risk", ""),
                        "verified": "stable",
                        "gemini_note": ""
                    })

            # Sort by confidence (highest first)
            confirmed_pests.sort(key=lambda x: x["confidence"], reverse=True)

            # Update shared state
            with self.det_lock:
                self.latest_annotated = annotated
                self.stable_detections = confirmed_pests

            # Reset current counts for next cycle
            for h in self.detection_history.values():
                h["current_count"] = 0

            # Small sleep to control detection rate (~5-8 detections per second)
            time.sleep(0.1)

    def get_detections(self):
        with self.det_lock:
            return self.stable_detections.copy()


# Global camera detector instance
camera = CameraDetector()


# ==========================================
# 7. ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    crop_type = request.form.get('crop', 'Unknown')

    original_img_path = os.path.join("static", "uploads", "temp_upload.jpg")
    file.save(original_img_path)

    image = cv2.imread(original_img_path)
    if image is None:
        return jsonify({"error": "Could not read image"}), 400

    is_plant, plant_score = is_plant_image(image)
    plant_warning = None
    if not is_plant:
        plant_warning = f"This image doesn't look like a plant/crop (plant score: {plant_score}%). Results may be unreliable."

    enhanced = enhance_image(image)
    enhanced_path = os.path.join("static", "uploads", "enhanced.jpg")
    cv2.imwrite(enhanced_path, enhanced)

    results = model.predict(
        source=enhanced_path,
        conf=0.30,
        iou=0.5,
        imgsz=640,
        agnostic_nms=True,
        augment=True,
        max_det=50
    )

    detections = []
    counts = {}
    timestamp = int(time.time())
    result_filename = f"annotated_{timestamp}.jpg"
    result_img_path = os.path.join("static", "results", result_filename)

    for r in results:
        cv2.imwrite(result_img_path, r.plot())

        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            conf = round(float(box.conf[0]) * 100, 1)
            counts[label] = counts.get(label, 0) + 1

            if not any(d['name'] == label for d in detections):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h, w = image.shape[:2]
                pad = 20
                x1c, y1c = max(0, x1 - pad), max(0, y1 - pad)
                x2c, y2c = min(w, x2 + pad), min(h, y2 + pad)
                cropped = image[y1c:y2c, x1c:x2c]

                verified = "unverified"
                gemini_note = ""

                if cropped.size > 0 and conf < 75:
                    verify_result = verify_with_gemini(cropped, label)
                    if verify_result.get("is_correct"):
                        verified = "verified"
                    else:
                        actual = verify_result.get("actual_pest", "unknown")
                        if actual.lower() == "not a pest":
                            verified = "rejected"
                            gemini_note = "Gemini says this is not a pest"
                        else:
                            verified = "corrected"
                            gemini_note = f"Gemini suggests: {actual}"
                elif conf >= 75:
                    verified = "high-confidence"

                if verified == "rejected":
                    continue

                info = get_smart_pest_info(label)

                detections.append({
                    "name": label,
                    "confidence": conf,
                    "common_name": info.get("common_name", label),
                    "damage": info.get("damage", ""),
                    "identify": info.get("identify", ""),
                    "chemical": info.get("chemical", "Consult expert"),
                    "organic": info.get("organic", ""),
                    "quick_action": info.get("quick_action", ""),
                    "prevention": info.get("prevention", ""),
                    "severity": info.get("severity", "unknown"),
                    "crops_at_risk": info.get("crops_at_risk", ""),
                    "verified": verified,
                    "gemini_note": gemini_note
                })

    for d in detections:
        d['count'] = counts.get(d['name'], 0)

    for d in detections:
        save_detection(d, f"/static/results/{result_filename}", crop_type)

    if os.path.exists(enhanced_path):
        os.remove(enhanced_path)

    response_data = {
        "result_image_url": f"/static/results/{result_filename}",
        "detections": detections,
        "plant_score": plant_score
    }
    if plant_warning:
        response_data["warning"] = plant_warning

    return jsonify(response_data)


@app.route('/detect_deep', methods=['POST'])
def detect_deep():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    crop_type = request.form.get('crop', 'Unknown')

    original_img_path = os.path.join("static", "uploads", "temp_upload.jpg")
    file.save(original_img_path)

    image = cv2.imread(original_img_path)
    if image is None:
        return jsonify({"error": "Could not read image"}), 400

    is_plant, plant_score = is_plant_image(image)
    plant_warning = None
    if not is_plant:
        plant_warning = f"This image doesn't look like a plant/crop (plant score: {plant_score}%). Results may be unreliable."

    enhanced = enhance_image(image)

    all_detections = {}
    all_counts = {}
    best_annotated = None

    for img_size in [480, 640, 832]:
        results = model.predict(
            source=enhanced,
            conf=0.25,
            iou=0.5,
            imgsz=img_size,
            agnostic_nms=True,
            augment=True,
            max_det=50,
            verbose=False
        )

        if img_size == 640:
            for r in results:
                best_annotated = r.plot()

        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                conf = round(float(box.conf[0]) * 100, 1)
                all_counts[label] = all_counts.get(label, 0) + 1

                if label not in all_detections or conf > all_detections[label]["confidence"]:
                    info = get_smart_pest_info(label)
                    all_detections[label] = {
                        "name": label,
                        "confidence": conf,
                        "common_name": info.get("common_name", label),
                        "damage": info.get("damage", ""),
                        "identify": info.get("identify", ""),
                        "chemical": info.get("chemical", "Consult expert"),
                        "organic": info.get("organic", ""),
                        "quick_action": info.get("quick_action", ""),
                        "prevention": info.get("prevention", ""),
                        "severity": info.get("severity", "unknown"),
                        "crops_at_risk": info.get("crops_at_risk", ""),
                        "verified": "deep-scan",
                        "gemini_note": ""
                    }

    detections = list(all_detections.values())
    for d in detections:
        d['count'] = all_counts.get(d['name'], 0)

    for d in detections:
        save_detection(d, "", crop_type)

    timestamp = int(time.time())
    result_filename = f"annotated_deep_{timestamp}.jpg"
    result_img_path = os.path.join("static", "results", result_filename)
    if best_annotated is not None:
        cv2.imwrite(result_img_path, best_annotated)

    response_data = {
        "result_image_url": f"/static/results/{result_filename}",
        "detections": detections,
        "plant_score": plant_score
    }
    if plant_warning:
        response_data["warning"] = plant_warning

    return jsonify(response_data)


# ==========================================
# 8. LIVE CAMERA ROUTES (SMOOTH)
# ==========================================
def generate_frames():
    """Generator that yields frames as fast as the camera can read them."""
    while camera.running:
        frame = camera.read_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        ret, buffer = cv2.imencode('.jpg', frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Send a blank frame when stopped
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    ret, buffer = cv2.imencode('.jpg', blank)
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_camera')
def start_camera():
    camera.start()
    return jsonify({"status": "started"})


@app.route('/stop_camera')
def stop_camera():
    camera.stop()
    return jsonify({"status": "stopped"})


@app.route('/detections')
def get_detections():
    return jsonify({"detections": camera.get_detections()})


@app.route('/set_confidence/<int:level>')
def set_confidence(level):
    """Change live camera confidence threshold. Level: 1=low, 2=medium, 3=high"""
    thresholds = {1: 0.25, 2: 0.40, 3: 0.60}
    camera.confidence_threshold = thresholds.get(level, 0.40)
    return jsonify({"threshold": camera.confidence_threshold})


# ==========================================
# 9. HISTORY ROUTES
# ==========================================
@app.route('/history')
def get_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT timestamp, pest_name, common_name, confidence, count, 
                 severity, crop, treatment 
                 FROM detections ORDER BY id DESC LIMIT 50''')
    rows = c.fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            "timestamp": row[0],
            "pest_name": row[1],
            "common_name": row[2],
            "confidence": row[3],
            "count": row[4],
            "severity": row[5],
            "crop": row[6],
            "treatment": row[7]
        })

    return jsonify({"history": history})


@app.route('/history/stats')
def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM detections")
    total = c.fetchone()[0]

    c.execute("""SELECT common_name, SUM(count) as total 
                 FROM detections GROUP BY pest_name 
                 ORDER BY total DESC LIMIT 5""")
    top_pests = [{"name": r[0], "count": r[1]} for r in c.fetchall()]

    c.execute("""SELECT DATE(timestamp) as day, COUNT(*) 
                 FROM detections 
                 WHERE timestamp >= DATE('now', '-7 days')
                 GROUP BY day ORDER BY day""")
    daily = [{"date": r[0], "count": r[1]} for r in c.fetchall()]

    c.execute("""SELECT severity, COUNT(*) FROM detections 
                 GROUP BY severity""")
    severity = {r[0]: r[1] for r in c.fetchall()}

    conn.close()

    return jsonify({
        "total_detections": total,
        "top_pests": top_pests,
        "daily_trend": daily,
        "severity_breakdown": severity
    })


@app.route('/history/clear', methods=['POST'])
def clear_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM detections")
    conn.commit()
    conn.close()
    return jsonify({"message": "History cleared"})


@app.route('/lookup/<pest_name>')
def lookup_pest(pest_name):
    info = get_smart_pest_info(pest_name)
    return jsonify(info)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)