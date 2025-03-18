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
# @app.route('/api/predict/<model_name>', methods=['POST'])
# def predict(model_name):
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400

#     file = request.files['file']

#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
    
#     model = load_model(model_name)

#     if not model:
#         return jsonify({"error": "Model not found"}), 404
    
#     try:
#         # Process file
#         contents = file.read()
#         df = pd.read_csv(pd.io.common.BytesIO(contents))

#         prediction = model.predict(df)
#         score = float(np.clip(prediction[0], 0, 1))

#         return jsonify({
#             "model_name": model_name,
#             "score": score
#         })
#     except FileNotFoundError:
#         return jsonify({"error": f"Model '{model_name}' not found"}), 404
#     except ValueError as ve:
#         return jsonify({"error": str(ve)}), 400
#     except Exception as e:
#         return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# @app.route('/api/predict/<model_name>', methods=['POST'])
# def predict(model_name):
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400

#     file = request.files['file']

#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
    
#     # Load the model correctly
#     loaded_tuple = load_model(model_name)

#     if not loaded_tuple:
#         return jsonify({"error": "Model not found"}), 404
    
#     try:
#         # Unpack the tuple correctly
#         model, scaler, feature_names = loaded_tuple
        
#         # Read the file
#         contents = file.read()
#         df = pd.read_csv(pd.io.common.BytesIO(contents))

#         # Ensure test data has the same columns as training
#         df = df[feature_names]

#         # Standardize test data
#         df = scaler.transform(df)

#         # Make predictions
#         prediction = model.predict(df)
#         score = float(np.clip(prediction[0], 0, 1))

#         return jsonify({
#             "model_name": model_name,
#             "score": score
#         })
    
#     except FileNotFoundError:
#         return jsonify({"error": f"Model '{model_name}' not found"}), 404
#     except ValueError as ve:
#         return jsonify({"error": str(ve)}), 400
#     except Exception as e:
#         return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

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

