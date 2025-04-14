from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import base64
import re
import csv
import json
import os
import tempfile
from datetime import datetime
import uuid
import logging
from model_utils import get_model_names, load_model
import pandas as pd
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes and origins

# Set maximum content length for file uploads (100MB)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Global variable to store the MoveNet model
movenet = None

def load_movenet_model():
    global movenet
    try:
        # Load the MoveNet model
        logger.info("Loading MoveNet model...")
        
        # Try different model URLs
        model_urls = [
            "https://tfhub.dev/google/movenet/singlepose/lightning/4",  # Try this URL first
            "https://tfhub.dev/google/movenet/singlepose/lightning/3",
            "https://tfhub.dev/google/movenet/singlepose/lightning/2",
            "https://tfhub.dev/google/movenet/singlepose/lightning/1",
            "https://www.kaggle.com/models/google/movenet/TensorFlow2/singlepose-lightning/1"
        ]
        
        for url in model_urls:
            try:
                logger.info(f"Attempting to load model from: {url}")
                model = hub.load(url)
                movenet = model.signatures['serving_default']
                logger.info(f"MoveNet model loaded successfully from {url}")
                return
            except Exception as e:
                logger.warning(f"Failed to load from {url}: {e}")
                continue
        
        # If we get here, all URLs failed
        logger.error("All model URLs failed. MoveNet functionality will be disabled.")
        
    except Exception as e:
        logger.error(f"Error loading MoveNet model: {e}")
        # Don't raise, allow app to continue even if MoveNet fails to load

# Load the MoveNet model when the app starts
load_movenet_model()

# Keypoint names in the order they appear in the model's output
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Directories for saving output files
CSV_DIR = "pose_data/csv"
JSON_DIR = "pose_data/json"
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

# Ensure an upload folder exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def base64_to_image(base64_string):
    try:
        # Extract the base64 encoded binary data
        if 'base64,' in base64_string:
            # Remove data URL prefix if present
            base64_string = base64_string.split('base64,')[1]
        
        # Decode base64 string
        image_bytes = base64.b64decode(base64_string)
        
        # Convert to OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error("Failed to decode image from base64")
            return None
        
        logger.info(f"Image decoded successfully. Shape: {image.shape}")
        return image
    except Exception as e:
        logger.error(f"Error converting base64 to image: {e}")
        return None

def run_movenet(image):
    if movenet is None:
        logger.error("MoveNet model not loaded")
        return []
        
    try:
        # Convert to RGB (MoveNet expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize the image to the expected input dimensions
        input_size = 192  # MoveNet Lightning uses 192x192, Thunder uses 256x256
        input_image = tf.image.resize_with_pad(tf.expand_dims(image_rgb, axis=0), input_size, input_size)
        
        # Convert to float32
        input_image = tf.cast(input_image, dtype=tf.int32)
        
        # Run inference
        results = movenet(input_image)
        
        # Extract keypoints
        keypoints = results['output_0'].numpy().squeeze()
        
        # Format results
        formatted_keypoints = []
        for idx, kp in enumerate(keypoints):
            y, x, score = kp
            formatted_keypoints.append({
                'name': KEYPOINT_NAMES[idx],
                'y': float(y),
                'x': float(x),
                'score': float(score)
            })
        
        logger.info(f"Detected {len(formatted_keypoints)} keypoints")
        return formatted_keypoints
    except Exception as e:
        logger.error(f"Error running MoveNet: {e}")
        return []

def generate_mock_keypoints():
    """Generate mock keypoints for testing when the real model isn't available"""
    mock_keypoints = []
    for idx, name in enumerate(KEYPOINT_NAMES):
        # Generate random but reasonable positions
        x = random.uniform(0.3, 0.7)
        y = random.uniform(0.3, 0.7)
        score = random.uniform(0.7, 0.95)
        
        mock_keypoints.append({
            'name': name,
            'x': float(x),
            'y': float(y),
            'score': float(score)
        })
    return mock_keypoints

def save_to_csv(keypoints, filename=None, frame_number=None):
    try:
        if filename is None:
            # Generate a unique filename using timestamp and UUID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{CSV_DIR}/pose_data_{timestamp}_{unique_id}.csv"
        else:
            # Make sure the filename has the correct path and extension
            if not filename.startswith(CSV_DIR):
                filename = f"{CSV_DIR}/{filename}"
            if not filename.endswith('.csv'):
                filename += '.csv'
        
        # Prepare data for CSV
        headers = ['keypoint', 'x', 'y', 'score']
        if frame_number is not None:
            headers.insert(0, 'frame')
        
        rows = []
        
        # If it's a single frame
        if frame_number is None:
            for kp in keypoints:
                rows.append([kp['name'], kp['x'], kp['y'], kp['score']])
        # If it's part of a video
        else:
            for kp in keypoints:
                rows.append([frame_number, kp['name'], kp['x'], kp['y'], kp['score']])
        
        # Check if file exists to determine if we should write headers
        file_exists = os.path.isfile(filename)
        
        # Write to CSV
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(headers)
            writer.writerows(rows)
        
        logger.info(f"Saved keypoints to CSV: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")
        return None

def save_to_json(keypoints_by_frame, filename=None):
    try:
        if filename is None:
            # Generate a unique filename using timestamp and UUID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{JSON_DIR}/pose_data_{timestamp}_{unique_id}.json"
        else:
            # Make sure the filename has the correct path and extension
            if not filename.startswith(JSON_DIR):
                filename = f"{JSON_DIR}/{filename}"
            if not filename.endswith('.json'):
                filename += '.json'
        
        # Write to JSON
        with open(filename, 'w') as jsonfile:
            json.dump(keypoints_by_frame, jsonfile, indent=2)
        
        logger.info(f"Saved keypoints to JSON: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving to JSON: {e}")
        return None

@app.route('/detect_pose', methods=['POST'])
def detect_pose():
    try:
        # Check if MoveNet model is loaded
        if movenet is None:
            logger.error("MoveNet model not loaded")
            return jsonify({'error': 'MoveNet model not loaded. Please check server logs.'}), 503
            
        # Check if request has JSON data
        if not request.is_json:
            logger.error("Invalid request: Not JSON")
            return jsonify({'error': 'Request must be JSON'}), 400
        
        # Get the base64 image data
        data = request.get_json()
        if 'image' not in data:
            logger.error("Invalid request: No image data")
            return jsonify({'error': 'No image data provided'}), 400
        
        base64_image = data['image']
        logger.info(f"Received image data, length: {len(base64_image)}")
        
        # Convert base64 to image
        image = base64_to_image(base64_image)
        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Run MoveNet on the image
        keypoints = run_movenet(image)
        if not keypoints:
            return jsonify({'error': 'Failed to detect keypoints'}), 500
        
        # For real-time tracking, optionally skip saving to CSV to improve performance
        save_csv = data.get('save_csv', True)  # Default to true
        csv_filename = None
        
        if save_csv:
            # Save keypoints to CSV
            csv_filename = save_to_csv(keypoints)
            if csv_filename is None:
                return jsonify({'error': 'Failed to save keypoints to CSV'}), 500
        
        # Return results
        return jsonify({
            'keypoints': keypoints,
            'csv_filename': os.path.basename(csv_filename) if csv_filename else None
        })
    except Exception as e:
        logger.error(f"Error in /detect_pose: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect_pose_simple', methods=['POST'])
def detect_pose_simple():
    try:
        # Check if request has JSON data
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        # Get the base64 image data
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        base64_image = data['image']
        
        # Convert base64 to image
        image = base64_to_image(base64_image)
        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # For testing: Generate mock keypoints instead of using the model
        keypoints = generate_mock_keypoints()
        
        # Check if we should save to CSV (for real-time tracking, we might skip)
        save_csv = data.get('save_csv', True)  # Default to true
        csv_filename = None
        
        if save_csv:
            # Save keypoints to CSV
            csv_filename = save_to_csv(keypoints)
        
        # Return results
        return jsonify({
            'keypoints': keypoints,
            'csv_filename': os.path.basename(csv_filename) if csv_filename else "mock_data.csv"
        })
        
    except Exception as e:
        logger.error(f"Error in /detect_pose_simple: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/process_video', methods=['POST'])
def process_video():
    try:
        if 'video' not in request.files:
            logger.error("No video file provided")
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        logger.info(f"Processing video: {video_file.filename}")
        
        # Generate timestamp for file identification
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        file_id = f"{timestamp}_{unique_id}"
        
        # Save the uploaded file to a temporary location
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"temp_video_{file_id}.mp4")
        video_file.save(temp_path)
        logger.info(f"Saved video to temporary file: {temp_path}")
        
        # Open the video file
        cap = cv2.VideoCapture(temp_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            logger.error("Error opening video file")
            return jsonify({'error': 'Error opening video file'}), 500
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video info: FPS={fps}, Total frames={frame_count}")
        
        # Create filenames for CSV and JSON output
        csv_filename = f"video_pose_data_{file_id}.csv"
        json_filename = f"video_pose_data_{file_id}.json"
        
        # Container for all keypoints by frame
        all_keypoints = {}
        
        # Process video frames at a reasonable sampling rate
        # For long videos, we may want to sample at a lower rate
        frame_interval = max(1, int(fps / 5))  # Process at most 5 frames per second
        
        frame_idx = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every nth frame
            if frame_idx % frame_interval == 0:
                # Run pose detection (use movenet if available, otherwise use mock data)
                if movenet is not None:
                    keypoints = run_movenet(frame)
                else:
                    keypoints = generate_mock_keypoints()
                
                # Save to CSV
                save_to_csv(keypoints, csv_filename, frame_idx)
                
                # Add to JSON data
                all_keypoints[str(frame_idx)] = keypoints
                
                processed_frames += 1
            
            frame_idx += 1
        
        # Save all keypoints to JSON
        json_path = save_to_json(all_keypoints, json_filename)
        
        # Release the video file
        cap.release()
        
        # Remove temporary file
        try:
            os.remove(temp_path)
            logger.info(f"Removed temporary file: {temp_path}")
        except:
            logger.warning(f"Failed to remove temporary file: {temp_path}")
        
        # Return results
        return jsonify({
            'csv_filename': os.path.basename(csv_filename),
            'json_filename': os.path.basename(json_filename),
            'frames_processed': processed_frames,
            'total_frames': frame_count,
            'fps': fps
        })
    except Exception as e:
        logger.error(f"Error in /process_video: {e}")
        return jsonify({'error': str(e)}), 500

# Add a simple health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model_loaded': movenet is not None
    })

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello from Flask API!'})

@app.route("/api/get-models", methods=["GET"])
def get_models_route():
    """API endpoint to fetch model names."""
    models_data = get_model_names()
    return jsonify(models_data)

@app.route('/api/models', methods=['GET'])
def get_models():
    models = [
        {"label": "Mode,l A", "value": "model_a"},
        {"label": "Model B", "value": "model_b"},
        {"label": "Model C", "value": "model_c"},
    ]
    return jsonify(models)

# Endpoint for Tab 2 dropdown options
@app.route('/api/categorizingModels', methods=['GET'])
def get_categorizing_models():
    categorizing_models = [
        {"label": "Category X", "value": "category_x"},
        {"label": "Category Y", "value": "category_y"},
        {"label": "Category Z", "value": "category_z"},
    ]
    return jsonify(categorizing_models)

@app.route('/api/predict/<model_name>', methods=['POST'])
def predict(model_name):
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Load the model correctly
    loaded_tuple = load_model(model_name,"regression")

    if not loaded_tuple:
        return jsonify({"error": "Model not found"}), 404
    
    try:
        # Unpack the tuple correctly
        model, scaler, feature_names = loaded_tuple
        
        # Read the file
        contents = file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))

        # Check if this model version requires transformed features
        if "multiplied" in model_name:  # If model name contains 'multiplied', we apply transformations
            symmetrical_columns = [
                ("No_3_Angle_Deviation", "No_5_Angle_Deviation"),
                ("No_4_Angle_Deviation", "No_6_Angle_Deviation"),
                ("No_7_Angle_Deviation", "No_10_Angle_Deviation"),
                ("No_8_Angle_Deviation", "No_11_Angle_Deviation"),
                ("No_9_Angle_Deviation", "No_12_Angle_Deviation"),
                ("No_13_NASM_Deviation", "No_14_NASM_Deviation"),
                ("No_16_NASM_Deviation", "No_17_NASM_Deviation"),
                ("No_20_NASM_Deviation", "No_21_NASM_Deviation"),
                ("No_23_NASM_Deviation", "No_24_NASM_Deviation")
            ]
            
            # Add multiplied columns
            for col1, col2 in symmetrical_columns:
                if col1 in df.columns and col2 in df.columns:
                    df[f"{col1}_multiplied"] = df[col1] * df[col2]

        # Ensure test data has the same columns as training
        missing_columns = [col for col in feature_names if col not in df.columns]
        if missing_columns:
            return jsonify({"error": f"Missing columns in test data: {missing_columns}"}), 400

        df = df[feature_names]  # Match feature order

        # Standardize test data
        df = scaler.transform(df)

        # Make predictions
        prediction = model.predict(df)
        score = float(np.clip(prediction[0], 0, 1))

        return jsonify({
            "model_name": model_name,
            "score": score
        })
    
    except FileNotFoundError:
        return jsonify({"error": f"Model '{model_name}' not found"}), 404
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# Route to classify weakest link using a string input
@app.route("/api/classify-weakest-link/<model_name>", methods=["POST"])
def classify_weakest_link(model_name):
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Load the model correctly
    loaded_tuple = load_model(model_name,"weaksetLink")

    if not loaded_tuple:
        return jsonify({"error": "Model not found"}), 404
    
    try:
        # Unpack the tuple correctly
        model = loaded_tuple
        
        # Read the file
        contents = file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))

        # Make predictions
        predicted_class = model.predict(df)[0]

        return jsonify({
            "model_name": model_name,
            "weakest_link": predicted_class
        })
    
    except FileNotFoundError:
        return jsonify({"error": f"Model '{model_name}' not found"}), 404
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route('/save_realtime_frames', methods=['POST'])
def save_realtime_frames():
    try:
        # Check if request has JSON data
        if not request.is_json:
            logger.error("Invalid request: Not JSON")
            return jsonify({'error': 'Request must be JSON'}), 400
        
        # Get the frames data
        data = request.get_json()
        if 'frames' not in data or not data['frames']:
            logger.error("Invalid request: No frames data or empty frames")
            return jsonify({'error': 'No frames data provided'}), 400
        
        frames = data['frames']
        logger.info(f"Received {len(frames)} frames for saving")
        
        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        csv_filename = f"realtime_tracking_{timestamp}_{unique_id}.csv"
        json_filename = f"realtime_tracking_{timestamp}_{unique_id}.json"
        
        # Save all frames to a single CSV file
        for frame_idx, keypoints in enumerate(frames):
            save_to_csv(keypoints, csv_filename, frame_idx)
        
        # Also save as JSON for easier processing
        all_keypoints = {str(idx): keypoints for idx, keypoints in enumerate(frames)}
        json_path = save_to_json(all_keypoints, json_filename)
        
        return jsonify({
            'csv_filename': os.path.basename(csv_filename),
            'json_filename': os.path.basename(json_filename),
            'frames_processed': len(frames)
        })
        
    except Exception as e:
        logger.error(f"Error in /save_realtime_frames: {e}")
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)