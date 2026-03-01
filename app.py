import os
import cv2
import json
import time
from flask import Flask, render_template, request, jsonify, Response
from ultralytics import YOLO
import torch
import google.generativeai as genai

app = Flask(__name__)

# ==========================================
# 1. GEMINI AI SETUP
# ==========================================
GEMINI_API_KEY = "AIzaSyDXD68jN8z1KRQuWm_iplZpNsfyblu-1Xw"  # Your API Key
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


def get_smart_pest_info(pest_name):
    """Asks Gemini for concise, farmer-friendly pest info."""
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

    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parse error for {pest_name}: {e}")
        return _fallback_info(pest_name)
    except Exception as e:
        print(f"⚠️ AI Lookup Failed for {pest_name}: {e}")
        return _fallback_info(pest_name)


def _fallback_info(pest_name):
    """Return a safe fallback if Gemini fails."""
    return {
        "common_name": pest_name,
        "damage": "Could not retrieve info. Check internet connection.",
        "identify": "Consult an agricultural officer for visual identification.",
        "chemical": "Consult local agri-store",
        "organic": "Neem oil spray (5ml per liter) as general remedy",
        "quick_action": "Isolate affected plants immediately.",
        "prevention": "Practice crop rotation and field hygiene.",
        "severity": "unknown",
        "crops_at_risk": "Unknown"
    }


# ==========================================
# 2. SYSTEM SETUP & YOLO MODEL LOADING
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
# 3. WEB ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    original_img_path = os.path.join("static", "uploads", "temp_upload.jpg")
    file.save(original_img_path)

    results = model.predict(source=original_img_path, conf=0.25)

    detections = []
    counts = {}
    timestamp = int(time.time())
    result_filename = f"annotated_upload_{timestamp}.jpg"
    result_img_path = os.path.join("static", "results", result_filename)

    for r in results:
        annotated_frame = r.plot()
        cv2.imwrite(result_img_path, annotated_frame)

        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            conf = round(float(box.conf[0]) * 100, 1)
            counts[label] = counts.get(label, 0) + 1

            if not any(d['name'] == label for d in detections):
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
                    "crops_at_risk": info.get("crops_at_risk", "")
                })

    for d in detections:
        d['count'] = counts[d['name']]

    return jsonify({
        "result_image_url": f"/static/results/{result_filename}",
        "detections": detections
    })


# ==========================================
# 4. LIVE CAMERA ROUTES
# ==========================================
def generate_frames():
    global latest_live_detections
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, conf=0.25, verbose=False)
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
                        "crops_at_risk": info.get("crops_at_risk", "")
                    })

        for d in current_dets:
            d['count'] = temp_counts[d['name']]

        latest_live_detections = current_dets

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
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


# ==========================================
# 5. STANDALONE AI LOOKUP (for testing)
# ==========================================
@app.route('/lookup/<pest_name>')
def lookup_pest(pest_name):
    """Hit /lookup/aphids in browser to test Gemini directly."""
    info = get_smart_pest_info(pest_name)
    return jsonify(info)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)