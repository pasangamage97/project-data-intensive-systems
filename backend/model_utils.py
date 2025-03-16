import os
import joblib

# Get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # Define paths for model directories
MODELS_DIR = os.path.join(BASE_DIR, "models")
CATEGORIZING_MODELS_DIR = os.path.join(BASE_DIR, "categorizing_models")


def get_models(directory):
    """Returns a list of .pkl file names (without extensions) from a given directory."""
    try:
        return [os.path.splitext(f)[0] for f in os.listdir(directory) if f.endswith(".pkl")]
    except FileNotFoundError:
        return []

def get_model_names():
    """Returns models and categorizing models."""
    return {
        "models": get_models(MODELS_DIR),
        "categorizingModels": get_models(CATEGORIZING_MODELS_DIR),
    }

def load_model(model_name):
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    
    if not os.path.exists(model_path):  # Check if the file exists
        raise FileNotFoundError(f"Model file '{model_name}.pkl' not found!")

    # Load the model correctly using joblib
    model = joblib.load(model_path)
    
    # Ensure that the loaded model has the 'predict' method
    # if not hasattr(model, 'predict'):
    #     raise ValueError(f"The loaded model '{model_name}' does not have a 'predict' method.")
    
    return model




# def get_models(directory):
#     """Returns a list of .pkl file names (without extensions) from a given directory."""
#     try:
#         return [os.path.splitext(f)[0] for f in os.listdir(directory) if f.endswith(".pkl")]
#     except FileNotFoundError:
#         return []

# def get_model_names():
#     """Returns models and categorizing models."""
#     return {
#         "models": get_models(MODELS_DIR),
#         "categorizingModels": get_models(CATEGORIZING_MODELS_DIR),
#     }



