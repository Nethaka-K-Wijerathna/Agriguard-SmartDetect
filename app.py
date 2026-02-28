import os
import cv2
from flask import Flask, render_template, request, jsonify, Response
from ultralytics import YOLO
import torch

app = Flask(__name__)

# --- 1. Bulletproof Folder Setup ---
# We force these folders to exist so we never get a path error
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

# Pesticide Dictionary
pesticide_dict = {
    "rice leaf roller": "Chlorantraniliprole",
    "rice leaf caterpillar": "Spinosad",
    "paddy stem maggot": "Chlorpyrifos",
    "asiatic rice borer": "Cartap hydrochloride",
    "yellow rice borer": "Cartap hydrochloride",
    "rice gall midge": "Imidacloprid",
    "Rice Stemfly": "Chlorpyrifos",
    "brown plant hopper": "Buprofezin",
    "white backed plant hopper": "Thiamethoxam",
    "small brown plant hopper": "Imidacloprid",
    "rice water weevil": "Lambda-cyhalothrin",
    "rice leafhopper": "Imidacloprid",
    "grain spreader thrips": "Spinosad",
    "rice shell pest": "Malathion"
}

latest_live_detections = []

# --- 2. Web Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['image']
    
    # Save original upload
    original_img_path = os.path.join("static", "uploads", "temp_upload.jpg")
    file.save(original_img_path)

    # Run YOLO (Lowered conf to 0.25 to make sure we don't miss pests!)
    results = model.predict(source=original_img_path, conf=0.25)
    
    detections = []
    counts = {}
    
    # We manually draw and save the image to avoid YOLO folder bugs
    result_img_path = os.path.join("static", "results", "annotated_upload.jpg")
    
    for r in results:
        # Draw bounding boxes
        annotated_frame = r.plot()
        # Save it directly where Flask expects it
        cv2.imwrite(result_img_path, annotated_frame)
        
        # Extract data for your UI cards
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            conf = round(float(box.conf[0]) * 100, 1)
            counts[label] = counts.get(label, 0) + 1
            
            if not any(d['name'] == label for d in detections):
                detections.append({
                    "name": label,
                    "confidence": conf,
                    "pesticide": pesticide_dict.get(label, "Consult Expert")
                })

    for d in detections:
        d['count'] = counts[d['name']]

    # Send the exact static URL to your Javascript
    return jsonify({
        "result_image_url": "/static/results/annotated_upload.jpg",
        "detections": detections
    })


# --- 3. Live Camera Routes ---

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
                    current_dets.append({
                        "name": lbl,
                        "confidence": round(float(box.conf[0]) * 100, 1),
                        "pesticide": pesticide_dict.get(lbl, "Check Guide")
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