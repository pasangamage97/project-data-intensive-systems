from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import base64
import re
import csv
import os
from datetime import datetime
import uuid
from model_utils import get_model_names,load_model

app = Flask(__name__)
CORS(app)


# Load the MoveNet model (You can choose either "lightning" or "thunder" variant)
model_name = "movenet_lightning"  # or "movenet_thunder" for higher accuracy but slower processing
model = hub.load(f"https://tfhub.dev/google/movenet/{model_name}/1")
movenet = model.signatures['serving_default']

# Keypoint names in the order they appear in the model's output
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Directory for saving CSV files
CSV_DIR = "pose_data"
os.makedirs(CSV_DIR, exist_ok=True)

def base64_to_image(base64_string):
    # Extract the base64 encoded binary data
    image_data = re.sub('^data:image/.+;base64,', '', base64_string)
    image_bytes = base64.b64decode(image_data)
    
    # Convert to OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return image

def run_movenet(image):
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
    
    return formatted_keypoints

def save_to_csv(keypoints, filename=None):
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
    rows = []
    
    for kp in keypoints:
        rows.append([kp['name'], kp['x'], kp['y'], kp['score']])
    
    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)
    
    return filename

@app.route('/detect_pose', methods=['POST'])
def detect_pose():
    if 'image' not in request.json:
        return jsonify({'error': 'No image data provided'}), 400
    
    # Get the base64 image data
    base64_image = request.json['image']
    
    # Convert base64 to image
    image = base64_to_image(base64_image)
    
    # Run MoveNet on the image
    keypoints = run_movenet(image)
    
    # Save keypoints to CSV
    csv_filename = save_to_csv(keypoints)
    
    # Return results
    return jsonify({
        'keypoints': keypoints,
        'csv_filename': os.path.basename(csv_filename)
    })


# Ensure an upload folder exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

