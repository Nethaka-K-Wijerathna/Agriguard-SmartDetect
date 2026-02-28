import os
import cv2
import json
from flask import Flask, render_template, request, jsonify, Response
from ultralytics import YOLO
import torch
import google.generativeai as genai

app = Flask(__name__)

# ==========================================
# 1. GEMINI AI SETUP (Auto-Detecting + JSON Mode)
# ==========================================
GEMINI_API_KEY = "AIzaSyAOBX8qok1NJ9CN-Q5fjiNVPtlqZJC8b1s" # <--- PASTE YOUR API KEY HERE
genai.configure(api_key=GEMINI_API_KEY)

# Automatically find a working model
working_model_name = "models/gemini-1.5-flash" # Fallback
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

# Memory Cache
pest_memory_cache = {}

def get_smart_pest_info(pest_name):
    """Asks Gemini for detailed pest info using strict JSON mode."""
    if pest_name in pest_memory_cache:
        return pest_memory_cache[pest_name]
    
    print(f"✨ New pest detected: '{pest_name}'. Asking Gemini for detailed info...")
    
    prompt = f"""
    You are an expert agricultural scientist. Provide a comprehensive guide for the crop pest '{pest_name}'.
    Use this exact JSON schema:
    {{
      "pesticide": "Name of best chemical and organic pesticide",
      "description": "A detailed 4-sentence paragraph explaining what the pest is, its behavior, and exact visual symptoms on the plant.",
      "action": "A step-by-step guide on how to get rid of it, including organic methods and chemical application."
    }}
    """
    
    try:
        # THE MAGIC FIX: Force Gemini to output unbreakable JSON
        response = ai_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        
        # Parse the guaranteed-clean JSON
        info = json.loads(response.text)
        
        # Save it to our cache
        pest_memory_cache[pest_name] = info
        print(f"✅ Successfully downloaded detailed guide for {pest_name}!")
        return info
        
    except Exception as e:
        print(f"⚠️ AI Lookup Failed for {pest_name}: {e}")
        return {
            "pesticide": "Consult Expert",
            "description": "The AI encountered an error while writing the detailed description. Please check your internet connection.",
            "action": "Consult a local agricultural extension officer."
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
# 3. WEB ROUTES (Frontend & Upload)
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

    # Run YOLO inference
    results = model.predict(source=original_img_path, conf=0.25)
    
    detections = []
    counts = {}
    result_img_path = os.path.join("static", "results", "annotated_upload.jpg")
    
    for r in results:
        annotated_frame = r.plot()
        cv2.imwrite(result_img_path, annotated_frame)
        
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            conf = round(float(box.conf[0]) * 100, 1)
            counts[label] = counts.get(label, 0) + 1
            
            if not any(d['name'] == label for d in detections):
                # Ask AI for detailed info
                smart_info = get_smart_pest_info(label)
                
                detections.append({
                    "name": label,
                    "confidence": conf,
                    "pesticide": smart_info.get("pesticide", "Unknown"),
                    "description": smart_info.get("description", ""),
                    "action": smart_info.get("action", "")
                })

    for d in detections:
        d['count'] = counts[d['name']]

    return jsonify({
        "result_image_url": "/static/results/annotated_upload.jpg",
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
                    # Ask AI for detailed info
                    smart_info = get_smart_pest_info(lbl)
                    
                    current_dets.append({
                        "name": lbl,
                        "confidence": round(float(box.conf[0]) * 100, 1),
                        "pesticide": smart_info.get("pesticide", "Unknown"),
                        "description": smart_info.get("description", ""),
                        "action": smart_info.get("action", "")
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
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    return jsonify({"detections": latest_live_detections})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)