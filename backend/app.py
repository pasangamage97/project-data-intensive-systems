from flask import Flask, request, jsonify, send_from_directory
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
from werkzeug.utils import secure_filename
import traceback
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FFMpegWriter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes and origins

# Set maximum content length for file uploads (2GB)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024

# Global variable to store the MoveNet model
movenet = None

# Global variable to store the GRU depth estimation model
gru_depth_model = None

def load_gru_depth_model():
    """Load the GRU model for Z value estimation"""
    global gru_depth_model
    try:
        model_path = "models/gru_depth_model"
        logger.info(f"Loading Z value estimation model from {model_path}")
        gru_depth_model = tf.keras.models.load_model(model_path)
        logger.info("Successfully loaded GRU depth model")
        return True
    except Exception as e:
        logger.error(f"Error loading GRU depth model: {e}")
        logger.warning("Will fall back to proportional Z estimation")
        return False

def load_movenet_model():
    global movenet
    try:
        # Print TensorFlow version for debugging
        logger.info(f"TensorFlow version: {tf.__version__}")
        
        # Check if TensorFlow GPU is available
        logger.info(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
        
        # Load the MoveNet model
        logger.info("Loading MoveNet model...")
        
        # Try different model URLs
        model_urls = [
            "https://tfhub.dev/google/movenet/singlepose/lightning/4",  # Try this URL first
            "https://tfhub.dev/google/movenet/singlepose/lightning/3",
            "https://tfhub.dev/google/movenet/singlepose/lightning/2",
            "https://tfhub.dev/google/movenet/singlepose/lightning/1",
            "https://www.kaggle.com/models/google/movenet/TensorFlow2/singlepose-lightning/1",
            "./models/movenet_singlepose_lightning_4"  # Try local path as last resort
        ]
        
        for url in model_urls:
            try:
                logger.info(f"Attempting to load model from: {url}")
                model = hub.load(url)
                movenet = model.signatures['serving_default']
                logger.info(f"MoveNet model loaded successfully from {url}")
                
                # Test the model with dummy input to verify it works
                dummy_input = tf.zeros((1, 192, 192, 3), dtype=tf.int32)
                try:
                    test_result = movenet(dummy_input)
                    logger.info(f"Model test successful, output shape: {test_result['output_0'].shape}")
                except Exception as test_err:
                    logger.warning(f"Model test failed: {test_err}")
                    continue
                    
                return True
            except Exception as e:
                logger.warning(f"Failed to load from {url}: {str(e)}")
                continue
        
        # If we get here, all URLs failed
        logger.error("All model URLs failed. MoveNet functionality will be disabled.")
        return False
        
    except Exception as e:
        logger.error(f"Error loading MoveNet model: {e}")
        return False

# Load the models when the app starts
model_loaded = load_movenet_model()
gru_model_loaded = load_gru_depth_model()

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
PROCESSED_VIDEO_DIR = "static/processed_videos"
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(PROCESSED_VIDEO_DIR, exist_ok=True)

# Ensure an upload folder exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def estimate_z_values_with_gru(keypoints_2d):
    """
    Estimate Z values for 2D keypoints using the GRU model
    
    Args:
        keypoints_2d: List of dictionaries with keypoints from MoveNet
    
    Returns:
        List of same keypoints with 'z' values added
    """
    if not gru_model_loaded or gru_depth_model is None:
        logger.warning("GRU model not loaded, using proportional estimation")
        return estimate_z_values_proportional(keypoints_2d)
    
    try:
        # Extract x, y coordinates in the order expected by the model
        keypoint_map = {kp['name']: kp for kp in keypoints_2d}
        
        # Prepare the input sequence for the GRU model
        input_sequence = []
        
        # Use the KEYPOINT_NAMES order
        for name in KEYPOINT_NAMES:
            if name in keypoint_map and keypoint_map[name]['score'] > 0.3:
                x = keypoint_map[name]['x']
                y = keypoint_map[name]['y']
                input_sequence.extend([x, y])
            else:
                # Use zeros for missing keypoints
                input_sequence.extend([0.0, 0.0])
        
        # Reshape for the model: [batch_size, sequence_length, features]
        # Adjust shape based on your GRU model architecture
        model_input = np.array([input_sequence])
        model_input = model_input.reshape(1, -1, 2)  # Adjust shape as needed
        
        # Run inference
        z_values = gru_depth_model.predict(model_input)[0]
        
        # Add Z values to the original keypoints
        for i, name in enumerate(KEYPOINT_NAMES):
            if name in keypoint_map:
                keypoint_map[name]['z'] = float(z_values[i])
        
        return keypoints_2d
    
    except Exception as e:
        logger.error(f"Error in GRU Z estimation: {e}")
        # Fall back to proportional estimation
        return estimate_z_values_proportional(keypoints_2d)

def estimate_z_values_proportional(keypoints):
    """Fallback method using anatomical proportions for depth estimation"""
    # Create a mapping for easier access
    keypoint_map = {kp['name']: kp for kp in keypoints}
    
    # Set initial depth values to 0
    for kp in keypoints:
        kp['z'] = 0.0
    
    # Use shoulder width as reference for scaling
    left_shoulder = keypoint_map.get('left_shoulder')
    right_shoulder = keypoint_map.get('right_shoulder')
    
    if left_shoulder and right_shoulder and left_shoulder['score'] > 0.3 and right_shoulder['score'] > 0.3:
        # Calculate shoulder width in x-coordinate space
        shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
        
        # Set scale factor based on shoulder width
        scale = shoulder_width * 0.5
        
        # Face depth - nose is in front
        if 'nose' in keypoint_map and keypoint_map['nose']['score'] > 0.3:
            keypoint_map['nose']['z'] = 0.3 * scale
        
        # Eyes slightly behind nose
        for eye in ['left_eye', 'right_eye']:
            if eye in keypoint_map and keypoint_map[eye]['score'] > 0.3:
                keypoint_map[eye]['z'] = 0.25 * scale
        
        # Ears behind eyes
        for ear in ['left_ear', 'right_ear']:
            if ear in keypoint_map and keypoint_map[ear]['score'] > 0.3:
                keypoint_map[ear]['z'] = 0.15 * scale
        
        # Shoulders at reference plane
        keypoint_map['left_shoulder']['z'] = 0
        keypoint_map['right_shoulder']['z'] = 0
        
        # Elbows slightly forward
        for elbow in ['left_elbow', 'right_elbow']:
            if elbow in keypoint_map and keypoint_map[elbow]['score'] > 0.3:
                # Calculate x-distance from shoulder to determine if arm is in front or behind
                side = 'left' if 'left' in elbow else 'right'
                shoulder_x = keypoint_map[f'{side}_shoulder']['x']
                elbow_x = keypoint_map[elbow]['x']
                
                # If elbow is more inward than the shoulder, it's likely in front
                # For left side: if elbow_x > shoulder_x, it's inward
                # For right side: if elbow_x < shoulder_x, it's inward
                if (side == 'left' and elbow_x > shoulder_x) or (side == 'right' and elbow_x < shoulder_x):
                    keypoint_map[elbow]['z'] = 0.1 * scale
                else:
                    keypoint_map[elbow]['z'] = -0.1 * scale
        
        # Wrists can extend further in z based on elbow positions
        for wrist in ['left_wrist', 'right_wrist']:
            if wrist in keypoint_map and keypoint_map[wrist]['score'] > 0.3:
                side = 'left' if 'left' in wrist else 'right'
                elbow = f'{side}_elbow'
                
                if elbow in keypoint_map and keypoint_map[elbow]['score'] > 0.3:
                    # Extend the elbow z position further
                    elbow_z = keypoint_map[elbow]['z']
                    keypoint_map[wrist]['z'] = elbow_z * 1.5
        
        # Hips slightly behind shoulders
        for hip in ['left_hip', 'right_hip']:
            if hip in keypoint_map and keypoint_map[hip]['score'] > 0.3:
                keypoint_map[hip]['z'] = -0.05 * scale
        
        # Knees can be in front or behind based on pose
        for knee in ['left_knee', 'right_knee']:
            if knee in keypoint_map and keypoint_map[knee]['score'] > 0.3:
                # Default slightly behind
                keypoint_map[knee]['z'] = -0.1 * scale
        
        # Ankles typically aligned with knees in z
        for ankle in ['left_ankle', 'right_ankle']:
            if ankle in keypoint_map and keypoint_map[ankle]['score'] > 0.3:
                side = 'left' if 'left' in ankle else 'right'
                knee = f'{side}_knee'
                
                if knee in keypoint_map and keypoint_map[knee]['score'] > 0.3:
                    # Roughly same z as knee
                    keypoint_map[ankle]['z'] = keypoint_map[knee]['z']
    
    return keypoints

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
        
        # Add Z values using our GRU model or proportional estimation
        formatted_keypoints = estimate_z_values_with_gru(formatted_keypoints)
        
        logger.info(f"Detected {len(formatted_keypoints)} keypoints with Z values")
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
    
    # Add Z values using our estimation methods
    mock_keypoints = estimate_z_values_proportional(mock_keypoints)
    return mock_keypoints

def save_to_csv(keypoints, filename=None, frame_number=0):
    logger.info(f"[save_to_csv] Saving: {filename}, Frame: {frame_number}")

    try:
        if filename is None:
            # Generate a unique filename using timestamp and UUID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{CSV_DIR}/pose_data_{timestamp}_{unique_id}.csv"
        else:
            # Make sure the filename has the correct path and extension
            # Ensure parent directories exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            if not filename.endswith('.csv'):
                filename += '.csv'

        
        # Check if file exists to determine if we should write headers
        file_exists = os.path.isfile(filename)
        
        # Define the column names (including z-coordinate)
        fieldnames = ['FrameNo']
        joint_names = ['head', 'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow',
                      'left_hand', 'right_hand', 'left_hip', 'right_hip',
                      'left_knee', 'right_knee', 'left_foot', 'right_foot']
        
        for name in joint_names:
            fieldnames.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
        
        # Map MoveNet keypoints to our joint names
        keypoint_mapping = {
            'nose': 'head',
            'left_shoulder': 'left_shoulder',
            'right_shoulder': 'right_shoulder',
            'left_elbow': 'left_elbow',
            'right_elbow': 'right_elbow',
            'left_wrist': 'left_hand',
            'right_wrist': 'right_hand',
            'left_hip': 'left_hip',
            'right_hip': 'right_hip',
            'left_knee': 'left_knee',
            'right_knee': 'right_knee',
            'left_ankle': 'left_foot',
            'right_ankle': 'right_foot'
        }
        
        # Convert keypoints to a dictionary matching our format
        keypoint_dict = {keypoint_mapping.get(kp['name']): kp for kp in keypoints 
                          if kp['name'] in keypoint_mapping}
        
        # Prepare the row data
        row_data = {'FrameNo': frame_number}
        
        # Add each mapped keypoint's x, y, and z values
        for joint in joint_names:
            if joint in keypoint_dict:
                kp = keypoint_dict[joint]
                row_data[f"{joint}_x"] = kp['x']
                row_data[f"{joint}_y"] = kp['y']
                row_data[f"{joint}_z"] = kp.get('z', 0.0)  # Default to 0 if z not present
            else:
                # If we don't have data for this joint, use None or 0
                row_data[f"{joint}_x"] = None
                row_data[f"{joint}_y"] = None
                row_data[f"{joint}_z"] = None
                
        # Write to CSV
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header only if the file is new
            if not file_exists:
                writer.writeheader()
            
            # Write the row
            writer.writerow(row_data)
        
        logger.info(f"Saved frame {frame_number} to CSV: {filename}")
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
        
        # Log image dimensions for debugging
        logger.info(f"Image dimensions: {image.shape}")
        
        # Run MoveNet on the image
        if model_loaded and movenet is not None:
            keypoints = run_movenet(image)
            if not keypoints:
                logger.warning("MoveNet returned no keypoints, falling back to mock data")
                keypoints = generate_mock_keypoints()
        else:
            # Use mock data if model isn't loaded
            logger.info("Using mock keypoints as MoveNet is not loaded")
            keypoints = generate_mock_keypoints()
        
        # For real-time tracking, optionally skip saving to CSV to improve performance
        save_csv = data.get('save_csv', True)  # Default to true
        csv_filename = None
        
        if save_csv:
            # Save keypoints to CSV
            csv_filename = save_to_csv(keypoints)
            if csv_filename is None:
                return jsonify({'error': 'Failed to save keypoints to CSV'}), 500
        
        # Return results with a flag indicating if mock data was used
        return jsonify({
            'keypoints': keypoints,
            'csv_filename': os.path.basename(csv_filename) if csv_filename else None,
            'using_mock_data': not model_loaded or movenet is None
        })
    except Exception as e:
        logger.error(f"Error in /detect_pose: {e}")
        return jsonify({'error': str(e)}), 500
    

@app.route('/process_bulk_videos', methods=['POST'])
def process_bulk_videos():
    try:
        if 'videos' not in request.files:
            return jsonify({'error': 'No videos uploaded'}), 400

        uploaded_files = request.files.getlist('videos')
        if not uploaded_files:
            return jsonify({'error': 'Empty file list'}), 400

        # Create a unique output folder for this batch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join("pose_data", "generated", f"batch_{timestamp}")
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"Created output folder: {output_folder}")

        # 13 PoseNet joints used
        pose_joints = [
            'head', 'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow',
            'left_hand', 'right_hand', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_foot', 'right_foot'
        ]

        results = []

        for file in uploaded_files:
            filename = secure_filename(file.filename)
            base_name = os.path.splitext(filename)[0]
            temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}_{filename}")
            file.save(temp_path)
            logger.info(f"[{filename}] Saved to temp path: {temp_path}")

            try:
                cap = cv2.VideoCapture(temp_path)
                if not cap.isOpened():
                    logger.warning(f"[{filename}] Could not open video.")
                    continue

                frame_interval = 5
                saved_frames = 0
                frame_idx = 0
                combined_rows = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % frame_interval == 0:
                        keypoints = run_movenet(frame) if model_loaded and movenet else generate_mock_keypoints()
                        kp_map = {kp['name']: kp for kp in keypoints}

                        row = []
                        for joint in pose_joints:
                            kp = kp_map.get(joint)
                            row.extend([kp['x'], kp['y']] if kp else [None, None])

                        combined_rows.append(row)
                        logger.info(f"[{filename}] Frame {frame_idx}: Keypoints processed.")
                        saved_frames += 1

                    frame_idx += 1

                cap.release()
                os.remove(temp_path)

                # Save CSV
                pose_cols = []
                for joint in pose_joints:
                    pose_cols.extend([f"{joint}_x", f"{joint}_y"])
                df = pd.DataFrame(combined_rows, columns=pose_cols)

                csv_path = os.path.join(output_folder, f"{base_name}_pose.csv")
                df.to_csv(csv_path, index=False)
                logger.info(f"[{filename}] CSV saved to {csv_path}")

                results.append({
                    "video": filename,
                    "frames_processed": frame_idx,
                    "frames_saved": saved_frames,
                    "csv_file": os.path.basename(csv_path),
                    "csv_path": csv_path
                })

            except Exception as e:
                logger.error(f"[{filename}] Error while processing video: {e}")

        return jsonify({
            "status": "completed",
            "total_videos": len(results),
            "output_folder": output_folder,
            "results": results
        })

    except Exception as e:
        logger.error(f"Bulk video processing error: {e}")
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
        
        # Skip image decoding if it's a placeholder or too short
        if len(base64_image) < 50:  # Arbitrary threshold for a minimal valid image
            logger.warning("Image data too short, using mock data without decoding")
            keypoints = generate_mock_keypoints()
        else:
            # Try to convert base64 to image
            image = base64_to_image(base64_image)
            if image is None:
                logger.warning("Failed to decode image, using mock data")
                keypoints = generate_mock_keypoints()
            else:
                # For testing: Generate mock keypoints
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
        # Always return mock data on any error
        keypoints = generate_mock_keypoints()
        return jsonify({
            'keypoints': keypoints,
            'csv_filename': "error_mock_data.csv",
            'error_details': str(e)
        })
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
        
        # Generate mock keypoints
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
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        file_id = f"{timestamp}_{unique_id}"

        temp_path = os.path.join(tempfile.gettempdir(), f"temp_video_{file_id}.mp4")
        video_file.save(temp_path)

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return jsonify({'error': 'Error opening video file'}), 500

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps / 5))
        all_frames = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                keypoints = run_movenet(frame) if model_loaded and movenet else generate_mock_keypoints()
                all_frames.append(keypoints)
            frame_idx += 1

        cap.release()
        os.remove(temp_path)

        # Setup headless matplotlib
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 6))
        ax_front = fig.add_subplot(121, projection='3d')
        ax_side = fig.add_subplot(122, projection='3d')

        output_video_path = os.path.join(PROCESSED_VIDEO_DIR, f"processed_skeleton_{file_id}.mp4")
        writer = FFMpegWriter(fps=5)

        def update_plot(keypoints):
            ax_front.cla()
            ax_side.cla()

            for ax in [ax_front, ax_side]:
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)

            ax_front.view_init(elev=15, azim=-60)  # default front view
            ax_side.view_init(elev=15, azim=0)     # side view facing person

            keypoint_map = {kp['name']: kp for kp in keypoints}

            # Draw joints
            for kp in keypoints:
                if kp['score'] < 0.3:
                    continue
                x = kp['x'] * 2 - 1
                y = kp['z'] * 0.8
                z = -(kp['y'] * 2 - 1)

                ax_front.scatter(x, y, z, c='red', s=30)
                ax_side.scatter(z, y, -x, c='red', s=30)  # Rotated

            # Connect bones
            skeleton = [
                ('nose', 'left_eye'), ('nose', 'right_eye'),
                ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
                ('left_shoulder', 'right_shoulder'),
                ('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow'),
                ('left_elbow', 'left_wrist'), ('right_elbow', 'right_wrist'),
                ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
                ('left_hip', 'right_hip'),
                ('left_hip', 'left_knee'), ('right_hip', 'right_knee'),
                ('left_knee', 'left_ankle'), ('right_knee', 'right_ankle')
            ]

            for start, end in skeleton:
                if start in keypoint_map and end in keypoint_map:
                    p1, p2 = keypoint_map[start], keypoint_map[end]
                    if p1['score'] < 0.3 or p2['score'] < 0.3:
                        continue
                    x1, y1, z1 = p1['x'] * 2 - 1, p1['z'] * 0.8, -(p1['y'] * 2 - 1)
                    x2, y2, z2 = p2['x'] * 2 - 1, p2['z'] * 0.8, -(p2['y'] * 2 - 1)

                    # Front view
                    ax_front.plot([x1, x2], [y1, y2], [z1, z2], c='blue', linewidth=2)

                    # Side view (rotated around Y-axis)
                    ax_side.plot([z1, z2], [y1, y2], [-x1, -x2], c='blue', linewidth=2)

        with writer.saving(fig, output_video_path, dpi=100):
            for frame_kps in all_frames:
                update_plot(frame_kps)
                writer.grab_frame()

        plt.close(fig)

        return jsonify({
            'processed_video_url': f"/static/processed_videos/processed_skeleton_{file_id}.mp4",
            'frames_processed': len(all_frames),
            'note': 'Dual 3D skeleton video generated: front and true side view'
        })

    except Exception as e:
        logger.error(f"Error in process_video: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# Add a simple health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded,
        'gru_model_loaded': gru_model_loaded
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
        {"label": "Model A", "value": "model_a"},
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