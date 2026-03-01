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
GEMINI_API_KEY = "AIzaSyARTvYcRppJQ6hczeAggZkg6CqsWm_e2pA"  # Replace with your key
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
    """Asks Gemini for detailed pest info using strict JSON mode."""
    if pest_name in pest_memory_cache:
        return pest_memory_cache[pest_name]

    print(f"✨ New pest detected: '{pest_name}'. Asking Gemini...")

    prompt = f"""
You are an expert agricultural entomologist. A farmer has detected the pest "{pest_name}" on their crop.
Respond ONLY with a valid JSON object using this exact schema (no extra keys, no markdown):

{{
  "pesticide": "The single best recommended chemical pesticide (brand or active ingredient name)",
  "organic_solution": "The best organic or biological control method",
  "description": "A detailed 3-4 sentence paragraph: what this pest is, what crops it attacks, its life cycle behavior, and the visible damage symptoms on the plant.",
  "action": "A numbered step-by-step action plan (as a single string) with at least 4 steps. Include: 1) Immediate action, 2) Chemical treatment with dosage if possible, 3) Organic alternative, 4) Long-term prevention.",
  "severity": "low, medium, or high",
  "crops_affected": "Comma-separated list of crops this pest commonly attacks"
}}
"""

    try:
        response = ai_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.3
            )
        )

        info = json.loads(response.text)
        pest_memory_cache[pest_name] = info
        print(f"✅ Got detailed guide for '{pest_name}'")
        return info

    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parse error for {pest_name}: {e}")
        print(f"Raw response: {response.text[:500]}")
        return _fallback_info(pest_name)
    except Exception as e:
        print(f"⚠️ AI Lookup Failed for {pest_name}: {e}")
        return _fallback_info(pest_name)


def _fallback_info(pest_name):
    """Return a safe fallback if Gemini fails."""
    return {
        "pesticide": "Consult local agricultural officer",
        "organic_solution": "Use neem oil spray as a general organic remedy",
        "description": f"The AI could not retrieve details for '{pest_name}'. Please verify your internet connection and API key.",
        "action": "1) Manually inspect the crop. 2) Isolate affected plants. 3) Consult a local agricultural extension officer. 4) Apply a broad-spectrum insecticide as a temporary measure.",
        "severity": "unknown",
        "crops_affected": "Unknown"
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
    # Use timestamp to avoid browser caching the old image
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
                smart_info = get_smart_pest_info(label)

                detections.append({
                    "name": label,
                    "confidence": conf,
                    "pesticide": smart_info.get("pesticide", "Unknown"),
                    "organic_solution": smart_info.get("organic_solution", ""),
                    "description": smart_info.get("description", ""),
                    "action": smart_info.get("action", ""),
                    "severity": smart_info.get("severity", "unknown"),
                    "crops_affected": smart_info.get("crops_affected", "")
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
                    smart_info = get_smart_pest_info(lbl)

                    current_dets.append({
                        "name": lbl,
                        "confidence": round(float(box.conf[0]) * 100, 1),
                        "pesticide": smart_info.get("pesticide", "Unknown"),
                        "organic_solution": smart_info.get("organic_solution", ""),
                        "description": smart_info.get("description", ""),
                        "action": smart_info.get("action", ""),
                        "severity": smart_info.get("severity", "unknown"),
                        "crops_affected": smart_info.get("crops_affected", "")
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
    """Hit /lookup/aphids in the browser to test Gemini directly."""
    info = get_smart_pest_info(pest_name)
    return jsonify(info)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)