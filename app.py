import os
import time
import threading
from threading import Lock
from flask import Flask, render_template, request, url_for, Response, jsonify
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# --- Load the Model ---
print("Loading YOLO model...")
try:
    model = YOLO("last.pt")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Pesticide mapping (Rice pests + others) ---
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
    "rice shell pest": "Malathion",
    "black cutworm": "Chlorpyrifos",
    "large cutworm": "Chlorpyrifos",
    "yellow cutworm": "Chlorantraniliprole",
    "army worm": "Emamectin benzoate",
    "corn borer": "Chlorantraniliprole",
    "aphids": "Imidacloprid",
    "english grain aphid": "Thiamethoxam",
    "green bug": "Acetamiprid",
    "bird cherry-oataphid": "Imidacloprid",
    "Aphis citricola Vander Goot": "Imidacloprid",
    "Toxoptera citricidus": "Thiamethoxam",
    "Toxoptera aurantii": "Acetamiprid",
    "Thrips": "Spinosad",
    "wheat phloeothrips": "Spinosad",
    "Scirtothrips dorsalis Hood": "Spinosad",
    "odontothrips loti": "Spinosad",
    "red spider": "Abamectin",
    "longlegged spider mite": "Abamectin",
    "Panonchus citri McGregor": "Abamectin",
    "Brevipoalpus lewisi McGregor": "Abamectin",
    "Polyphagotars onemus latus": "Abamectin",
    "flea beetle": "Carbaryl",
    "beet weevil": "Lambda-cyhalothrin",
    "alfalfa weevil": "Chlorpyrifos",
    "wireworm": "Fipronil",
    "white margined moth": "Chlorantraniliprole",
    "Lycorma delicatula": "Imidacloprid",
    "Bactrocera tsuneonis": "Spinosad bait",
    "Dacus dorsalis(Hendel)": "Spinosad bait",
    "Phyllocnistis citrella Stainton": "Imidacloprid",
    "Papilio xuthus": "Chlorantraniliprole",
    "Trialeurodes vaporariorum": "Imidacloprid",
    "Aleurocanthus spiniferus": "Thiamethoxam",
    "Icerya purchasi Maskell": "Imidacloprid",
    "Ceroplastes rubens": "Imidacloprid",
    "Unaspis yanonensis": "Imidacloprid",
    "Potosiabre vitarsis": "Lambda-cyhalothrin",
    "peach borer": "Chlorpyrifos",
    "sericaorient alismots chulsky": "Imidacloprid",
    "flax budworm": "Chlorantraniliprole",
    "alfalfa plant bug": "Thiamethoxam",
    "tarnished plant bug": "Imidacloprid",
    "Locustoidea": "Malathion",
    "lytta polita": "Carbaryl",
    "legume blister beetle": "Carbaryl",
    "blister beetle": "Carbaryl",
    "therioaphis maculata Buckton": "Imidacloprid",
    "alfalfa seed chalcid": "Chlorpyrifos",
    "Pieris canidia": "Chlorantraniliprole",
    "Apolygus lucorum": "Thiamethoxam",
    "Limacodidae": "Chlorantraniliprole",
    "Viteus vitifoliae": "Imidacloprid",
    "Colomerus vitis": "Abamectin",
    "oides decempunctata": "Lambda-cyhalothrin",
    "Pseudococcus comstocki Kuwana": "Imidacloprid",
    "parathrene regalis": "Fipronil",
    "Ampelophaga": "Fipronil",
    "Xylotrechus": "Fipronil",
    "Cicadella viridis": "Imidacloprid",
    "Miridae": "Thiamethoxam",
    "Erythroneura apicalis": "Imidacloprid",
    "Panonchus citri McGregor": "Oxamyl",
    "Phyllocoptes oleiverus ashmead": "Abamectin",
    "Chrysomphalus aonidum": "Imidacloprid",
    "Parlatoria zizyphus Lucus": "Imidacloprid",
    "Nipaecoccus vastalor": "Imidacloprid",
    "Tetradacus c Bactrocera minax": "Spinosad bait",
    "Prodenia litura": "Spinosad",
    "Adristyrannus": "Lambda-cyhalothrin",
    "Dasineura sp": "Chlorpyrifos",
    "Lawana imitata Melichar": "Imidacloprid",
    "Salurnis marginella Guerr": "Lambda-cyhalothrin",
    "Deporaus marginatus Pascoe": "Lambda-cyhalothrin",
    "Chlumetia transversa": "Chlorantraniliprole",
    "Mango flat beak leafhopper": "Imidacloprid",
    "Rhytidodera bowrinii white": "Lambda-cyhalothrin",
    "Sternochetus frigidus": "Fipronil",
    "Cicadellidae": "Imidacloprid"
}

# Shared state for latest detections
latest_info = {'detections': {}, 'suggestion': None, 'timestamp': 0}
info_lock = Lock()


@app.route('/')
def index():
    return render_template('index.html')


def gen_frames():
    """Generator that captures frames from the default camera, runs detection,
    annotates the frame and yields multipart JPEG frames for MJPEG streaming.
    It also updates `latest_info` so the client can poll `/detections`.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: camera not accessible")
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Resize for performance
            frame = cv2.resize(frame, (640, 480))

            # Run model inference
            results = model(frame, conf=0.5, imgsz=640, verbose=False)

            # Annotate frame
            annotated = results[0].plot()

            # Build detection summary
            detections = {}
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    class_name = model.names.get(cls_id, str(cls_id))
                    detections[class_name] = detections.get(class_name, 0) + 1

            # Top suggestion
            if detections:
                top = max(detections.items(), key=lambda x: x[1])[0]
                suggestion = pesticide_dict.get(top, "Consult Agri Officer")
            else:
                suggestion = None

            with info_lock:
                latest_info['detections'] = detections
                latest_info['suggestion'] = suggestion
                latest_info['timestamp'] = int(time.time())

            # Encode JPEG
            ret, buffer = cv2.imencode('.jpg', annotated)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detections')
def detections():
    with info_lock:
        # Return a small JSON summary the frontend can display
        return jsonify(latest_info)


@app.route('/detect', methods=['POST'])
def detect():
    # keep existing upload-based detection for single images
    if 'image' not in request.files:
        return "No image part", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    if file:
        filename = "temp_upload.jpg"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        print(f"Image saved to: {upload_path}")

        # Run inference and save result image into static results
        results = model.predict(source=upload_path, save=True, conf=0.5, project=app.config['RESULT_FOLDER'], name='latest_run', exist_ok=True)

        import glob, shutil
        matches = glob.glob(os.path.join('runs', '**', filename), recursive=True)
        if matches:
            saved_path = matches[-1]
            dest_dir = os.path.join(app.config['RESULT_FOLDER'], 'latest_run')
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, filename)
            try:
                shutil.copy(saved_path, dest_path)
            except Exception:
                dest_path = saved_path
        else:
            dest_path = os.path.join(app.config['RESULT_FOLDER'], 'latest_run', filename)

        try:
            rel = os.path.relpath(dest_path, 'static').replace('\\', '/')
            result_url = url_for('static', filename=rel)
        except Exception:
            result_url = url_for('static', filename=f'results/latest_run/{filename}')

        result_url += f"?t={int(time.time())}"
        return {'result_image_url': result_url}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)