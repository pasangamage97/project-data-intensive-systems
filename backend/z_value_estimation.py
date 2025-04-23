# z_value_estimation.py
import tensorflow as tf
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class ZValueEstimator:
    def __init__(self, model_dir="models"):
        self.model = None
        self.model_path = os.path.join(model_dir, "gru_depth_model")
        self.initialized = False
        self._load_model()
    
    def _load_model(self):
        """Load the GRU model for Z value estimation"""
        try:
            logger.info(f"Loading Z value estimation model from {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Successfully loaded model: {self.model.summary()}")
            self.initialized = True
        except Exception as e:
            logger.error(f"Error loading Z estimation model: {e}")
            logger.warning("Will fall back to proportional Z estimation")
            self.initialized = False
    
    def estimate_z_values(self, keypoints_2d):
        """
        Estimate Z values for 2D keypoints using the GRU model
        
        Args:
            keypoints_2d: List of dictionaries with keypoints from MoveNet
                         (each with 'name', 'x', 'y', 'score')
        
        Returns:
            List of same keypoints with 'z' values added
        """
        if not self.initialized or not self.model:
            logger.warning("Model not initialized, using proportional estimation")
            return self._proportional_estimation(keypoints_2d)
        
        try:
            # Extract x, y coordinates in the order expected by the model
            keypoint_map = {kp['name']: kp for kp in keypoints_2d}
            
            # Prepare the input sequence for the GRU model
            # Assuming model expects a specific order of keypoints
            input_sequence = []
            
            # Use the KEYPOINT_NAMES order (from your Flask app)
            keypoint_order = [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ]
            
            # Gather the normalized coordinates
            for name in keypoint_order:
                if name in keypoint_map and keypoint_map[name]['score'] > 0.3:
                    x = keypoint_map[name]['x']
                    y = keypoint_map[name]['y']
                    input_sequence.extend([x, y])
                else:
                    # Use zeros for missing keypoints
                    input_sequence.extend([0.0, 0.0])
            
            # Reshape for the model: [batch_size, sequence_length, features]
            # The exact shape depends on your GRU model architecture
            model_input = np.array([input_sequence])
            model_input = model_input.reshape(1, -1, 2)  # Adjust shape as needed
            
            # Run inference
            z_values = self.model.predict(model_input)[0]
            
            # Add Z values to the original keypoints
            for i, name in enumerate(keypoint_order):
                if name in keypoint_map:
                    keypoint_map[name]['z'] = float(z_values[i])
            
            return keypoints_2d
        
        except Exception as e:
            logger.error(f"Error in GRU Z estimation: {e}")
            # Fall back to proportional estimation
            return self._proportional_estimation(keypoints_2d)
    
    def _proportional_estimation(self, keypoints):
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
                        elbow_z = keypoint_map[elbow]['z']
                        keypoint_map[wrist]['z'] = elbow_z * 1.5
            
            # Hips slightly behind shoulders
            for hip in ['left_hip', 'right_hip']:
                if hip in keypoint_map and keypoint_map[hip]['score'] > 0.3:
                    keypoint_map[hip]['z'] = -0.05 * scale
            
            # Knees can be in front or behind based on pose
            for knee in ['left_knee', 'right_knee']:
                if knee in keypoint_map and keypoint_map[knee]['score'] > 0.3:
                    keypoint_map[knee]['z'] = -0.1 * scale
            
            # Ankles typically aligned with knees in z
            for ankle in ['left_ankle', 'right_ankle']:
                if ankle in keypoint_map and keypoint_map[ankle]['score'] > 0.3:
                    side = 'left' if 'left' in ankle else 'right'
                    knee = f'{side}_knee'
                    
                    if knee in keypoint_map and keypoint_map[knee]['score'] > 0.3:
                        keypoint_map[ankle]['z'] = keypoint_map[knee]['z']
        
        return keypoints

# Create a singleton instance
z_estimator = ZValueEstimator()

def estimate_keypoint_z_values(keypoints):
    """
    Public function to estimate Z values for keypoints
    """
    return z_estimator.estimate_z_values(keypoints)