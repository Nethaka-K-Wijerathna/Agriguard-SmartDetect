import os
import cv2
import json
from flask import Flask, render_template, request, jsonify, Response
from ultralytics import YOLO
import torch
import google.generativeai as genai

app = Flask(__name__)

# --- 1. GEMINI AI SETUP ---
GEMINI_API_KEY = "AIzaSyDexhUetITC3LLS6i-ZwpyE-wz7AvGlVB0" # <--- PASTE YOUR API KEY HERE
genai.configure(api_key=GEMINI_API_KEY)
# We use gemini-1.5-flash because it is the fastest model for quick lookups
ai_model = genai.GenerativeModel('gemini-1.5-flash')

# Memory Cache to store AI responses so we don't ask twice and slow down the camera!
pest_memory_cache = {}

def get_smart_pest_info(pest_name):
    """Asks Gemini for pest info if we haven't seen it before."""
    # 1. Check if we already know this pest
    if pest_name in pest_memory_cache:
        return pest_memory_cache[pest_name]
    
    # 2. If it's new, ask Gemini!
    print(f"✨ New pest detected: '{pest_name}'. Asking Gemini for info...")
    prompt = f"""
    You are an expert agricultural AI. Provide info for the crop pest '{pest_name}'.
    Return ONLY a valid JSON object with exactly three keys:
    "pesticide": A short 1-3 word name of the best chemical or organic pesticide to use.
    "description": A 2-sentence description of the pest and the damage it causes to crops.
    "action": A 1-sentence instruction on how to apply the treatment.
    Do not include any markdown formatting like ```json. Just return the raw JSON.
    """
    
    try:
        response = ai_model.generate_content(prompt)
        # Parse the JSON response from Gemini
        info = json.loads(response.text.strip())
        
        # Save it to our cache so we never have to look it up again
        pest_memory_cache[pest_name] = info
        print(f"✅ Learned about {pest_name}!")
        return info
    except Exception as e:
        print(f"⚠️ AI Lookup Failed for {pest_name}: {e}")
        # Fallback if the AI fails or internet disconnects
        return {
            "pesticide": "Consult Expert",
            "description": "AI was unable to fetch information for this pest. Please check your internet connection.",
            "action": "Consult a local agricultural extension officer."
        }


# --- 2. SYSTEM SETUP & MODEL LOADING ---
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


# --- 3. WEB ROUTES ---

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
    result_img_path = os.path.join("static", "results", "annotated_upload.jpg")
    
    for r in results:
        annotated_frame = r.plot()
        cv2.imwrite(result_img_path, annotated_frame)
        
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            conf = round(float(box.conf[0]) * 100, 1)
            counts[label] = counts.get(label, 0) + 1
            
            if not any(d['name'] == label for d in detections):
                # >>> ASK THE AI FOR INFO <<<
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


# --- 4. LIVE CAMERA ROUTES ---

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
                    # >>> ASK THE AI FOR INFO <<<
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