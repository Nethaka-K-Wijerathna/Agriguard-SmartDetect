import os
import cv2
import json
import time
import numpy as np
import base64
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
GEMINI_API_KEY = ""
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
# 2. DATABASE SETUP (Pest History)
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
    """
    Send cropped pest image to Gemini Vision to verify YOLO's detection.
    Returns: verified (bool), gemini_label (str), confidence_note (str)
    """
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
            [
                prompt,
                {"mime_type": "image/jpeg", "data": img_base64}
            ],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )

        result = json.loads(response.text)
        return result

    except Exception as e:
        print(f"⚠️ Gemini verify failed: {e}")
        return {"is_correct": True, "actual_pest": yolo_label, "confidence": "unknown"}


def is_plant_image(image):
    """
    Quick check if the image looks like it's from a farm/plant.
    Prevents nonsense detections on random images.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Green range (plants/leaves)
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_percent = (np.sum(green_mask > 0) / green_mask.size) * 100

    # Brown range (soil, stems, dried leaves)
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


# ==========================================
# 5. MODEL LOADING
# ==========================================
os.makedirs(os.path.join("static", "results"), exist_ok=True)
os.makedirs(os.path.join("static", "uploads"), exist_ok=True)

print("Loading YOLO model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model = YOLO("last.pt")
    model.to(device)
    if device == "cuda":
        model.model.half()
    print(f"Model loaded successfully on {device}!")
except Exception as e:
    print(f"Error loading model: {e}")

latest_live_detections = []


# ==========================================
# 6. ROUTES
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

    # ── ACCURACY BOOST 1: Plant Check ──
    is_plant, plant_score = is_plant_image(image)
    plant_warning = None
    if not is_plant:
        plant_warning = f"This image doesn't look like a plant/crop (plant score: {plant_score}%). Results may be unreliable."

    # ── ACCURACY BOOST 2: Enhance Image ──
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
                # ── ACCURACY BOOST 3: Crop & Verify with Gemini ──
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Add padding around the crop
                h, w = image.shape[:2]
                pad = 20
                x1c = max(0, x1 - pad)
                y1c = max(0, y1 - pad)
                x2c = min(w, x2 + pad)
                y2c = min(h, y2 + pad)
                cropped = image[y1c:y2c, x1c:x2c]

                verified = "unverified"
                gemini_note = ""

                if cropped.size > 0 and conf < 75:
                    # Only verify low-confidence detections to save API calls
                    verify_result = verify_with_gemini(cropped, label)
                    if verify_result.get("is_correct"):
                        verified = "verified"
                    else:
                        actual = verify_result.get("actual_pest", "unknown")
                        if actual.lower() == "not a pest":
                            verified = "rejected"
                            gemini_note = "Gemini says this is not a pest — likely a false detection"
                        else:
                            verified = "corrected"
                            gemini_note = f"Gemini suggests this might be: {actual}"
                elif conf >= 75:
                    verified = "high-confidence"

                if verified == "rejected":
                    continue  # Skip false detections entirely

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

    # Save to history database
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
# 7. HISTORY ROUTES
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

    # Total scans
    c.execute("SELECT COUNT(*) FROM detections")
    total = c.fetchone()[0]

    # Most common pest
    c.execute("""SELECT common_name, SUM(count) as total 
                 FROM detections GROUP BY pest_name 
                 ORDER BY total DESC LIMIT 5""")
    top_pests = [{"name": r[0], "count": r[1]} for r in c.fetchall()]

    # Detections per day (last 7 days)
    c.execute("""SELECT DATE(timestamp) as day, COUNT(*) 
                 FROM detections 
                 WHERE timestamp >= DATE('now', '-7 days')
                 GROUP BY day ORDER BY day""")
    daily = [{"date": r[0], "count": r[1]} for r in c.fetchall()]

    # Severity breakdown
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


# ==========================================
# 8. LIVE CAMERA
# ==========================================
def generate_frames():
    global latest_live_detections
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        if frame_count % 3 == 0:
            enhanced = enhance_image(frame)

            results = model(
                enhanced,
                conf=0.30,
                verbose=False,
                imgsz=640,
                iou=0.5,
                agnostic_nms=True
            )

            current_dets = []
            temp_counts = {}

            for r in results:
                annotated_frame = r.plot()
                for box in r.boxes:
                    lbl = model.names[int(box.cls[0])]
                    temp_counts[lbl] = temp_counts.get(lbl, 0) + 1

                    if not any(d['name'] == lbl for d in current_dets):
                        info = get_smart_pest_info(lbl)
                        current_dets.append({
                            "name": lbl,
                            "confidence": round(float(box.conf[0]) * 100, 1),
                            "common_name": info.get("common_name", lbl),
                            "damage": info.get("damage", ""),
                            "identify": info.get("identify", ""),
                            "chemical": info.get("chemical", "Consult expert"),
                            "organic": info.get("organic", ""),
                            "quick_action": info.get("quick_action", ""),
                            "prevention": info.get("prevention", ""),
                            "severity": info.get("severity", "unknown"),
                            "crops_at_risk": info.get("crops_at_risk", ""),
                            "verified": "live",
                            "gemini_note": ""
                        })

            for d in current_dets:
                d['count'] = temp_counts[d['name']]

            latest_live_detections = current_dets
        else:
            annotated_frame = frame

        ret, buffer = cv2.imencode('.jpg', annotated_frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detections')
def get_detections():
    return jsonify({"detections": latest_live_detections})


@app.route('/lookup/<pest_name>')
def lookup_pest(pest_name):
    info = get_smart_pest_info(pest_name)
    return jsonify(info)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)