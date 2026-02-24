import os
from flask import Flask, render_template, request, send_from_directory, url_for
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# --- Configuration ---
# Define paths for upload and result folders relative to the static directory
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')

# Ensure these directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# --- Load the Model ---
# Load your trained YOLOv8 model
print("Loading YOLO model...")
try:
    model = YOLO("last.pt")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """
    API endpoint that handles image upload, runs detection, 
    and returns the path to the processed image.
    """
    if 'image' not in request.files:
        return "No image part", 400
    
    file = request.files['image']
    
    if file.filename == '':
        return "No selected file", 400

    if file:
        # 1. Save the uploaded image temporarily
        filename = "temp_upload.jpg" # Using a fixed name for simplicity in this demo
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        print(f"Image saved to: {upload_path}")

        # 2. Run inference using your model (adapted from your image_detect.py)
        # conf=0.5 sets the confidence threshold to 50%
        results = model.predict(source=upload_path, save=True, conf=0.5, project=app.config['RESULT_FOLDER'], name='latest_run', exist_ok=True)
        
        # YOLO saves results in a subdirectory. We need to find the actual saved file.
        # The standard save path is usually static/results/latest_run/temp_upload.jpg
        predicted_image_path = os.path.join(app.config['RESULT_FOLDER'], 'latest_run', filename)

        print(f"Detection complete. Saved to: {predicted_image_path}")

        # 3. Construct the URL for the frontend to access the result image
        # We tell the browser to look in the static folder for the result
        result_url = url_for('static', filename=f'results/latest_run/{filename}')
        
        # append a timestamp to prevent browser caching of the image
        import time
        result_url += f"?t={int(time.time())}"

        return {'result_image_url': result_url}

if __name__ == '__main__':
    # Run the Flask app over local network. 
    # Change host to '127.0.0.1' if you only want access on this machine.
    app.run(host='0.0.0.0', port=5000, debug=True)