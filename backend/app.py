from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
from flask_cors import CORS
from model_utils import get_model_names,load_model


app = Flask(__name__)
CORS(app)


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

# Route to handle CSV file upload for prediction
@app.route('/api/predict/<model_name>', methods=['POST'])
def predict(model_name):
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    model = load_model(model_name)

    if not model:
        return jsonify({"error": "Model not found"}), 404
    
    try:
        # Process file
        contents = file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))

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
@app.route("/api/classify-weakest-link", methods=["POST"])
def classify_weakest_link():
    """Handle weakest link classification."""
    data = request.json
    selected_model = data.get("model")
    input_text = data.get("input_text")

    if not selected_model or not input_text:
        return jsonify({"error": "Model and input text required"}), 400

    # Process classification logic
    return jsonify({"message": "Classification received successfully"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

