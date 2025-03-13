from flask import Flask, request, jsonify
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Ensure an upload folder exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello from Flask API!'})

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
@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Invalid file format. Please upload a CSV file."}), 400

    # Save file (optional, if processing immediately you may not need to save)
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # TODO: Process the CSV file for prediction
    # Example: data = pd.read_csv(file_path), then use ML model for prediction

    return jsonify({"message": "File received successfully!", "filename": file.filename})


# Route to classify weakest link using a string input
@app.route("/api/classify-weakest-link", methods=["POST"])
def classify_weakest_link():
    data = request.json

    if not data or "input_string" not in data:
        return jsonify({"error": "Missing input string"}), 400

    input_string = data["input_string"]

    # TODO: Process the input_string and classify weakest link
    # Example: result = model.classify(input_string)

    return jsonify({"message": "Weakest link classification successful!", "input": input_string})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

