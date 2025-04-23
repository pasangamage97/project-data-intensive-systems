import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
import joblib

# === CONFIGURATION ===
POSE_CSV_DIR = "C:/Users/Ashidow/project-data-intensive-systems/backend/pose_data/generated/batch_20250423_044632"
KINECT_CSV_DIR = "C:/Users/Ashidow/Documents/DL A2/kinect_all_videos/all_videos/kinect_good_preprocessed"
MODEL_SAVE_DIR = "C:/Users/Ashidow/project-data-intensive-systems/backend/models/p2dk2d"

# === JOINTS & COLUMNS ===
joints = [
    'head', 'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow',
    'left_hand', 'right_hand', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_foot', 'right_foot'
]
pose_cols = [f"{j}_x" for j in joints] + [f"{j}_y" for j in joints]
kinect_cols = [f"{j}_x" for j in joints] + [f"{j}_y" for j in joints]

# === LOAD AND PAIR FUNCTION ===
def load_csv_pairs(pose_dir, kinect_dir):
    input_data, target_data = [], []
    skipped = []

    for pose_file in os.listdir(pose_dir):
        if not pose_file.endswith("_pose.csv"):
            continue
        base_name = os.path.splitext(pose_file)[0].replace("_pose", "")
        pose_path = os.path.join(pose_dir, pose_file)
        kinect_path = os.path.join(kinect_dir, f"{base_name}_kinect.csv")

        if not os.path.exists(kinect_path):
            skipped.append((base_name, "Missing Kinect file"))
            continue

        try:
            pose_df = pd.read_csv(pose_path)
            kinect_df = pd.read_csv(kinect_path)
            kinect_df.columns = kinect_df.columns.str.strip()

            # Clean Kinect: Drop FrameNo & _z
            kinect_df = kinect_df.drop(columns=['FrameNo'], errors='ignore')
            kinect_df = kinect_df[[col for col in kinect_df.columns if not col.endswith('_z')]]

            # Validate columns
            if not set(pose_cols).issubset(pose_df.columns):
                skipped.append((base_name, "Missing PoseNet columns"))
                continue
            if not set(kinect_cols).issubset(kinect_df.columns):
                skipped.append((base_name, "Missing Kinect columns"))
                continue

            # Match frame count
            min_len = min(len(pose_df), len(kinect_df))
            pose_df = pose_df.iloc[:min_len].reset_index(drop=True)
            kinect_df = kinect_df.iloc[:min_len].reset_index(drop=True)

            # Fill NaNs with 0.0
            pose_df = pose_df.fillna(0.0)
            kinect_df = kinect_df.fillna(0.0)

            input_data.append(pose_df[pose_cols].values)
            target_data.append(kinect_df[kinect_cols].values)
        except Exception as e:
            skipped.append((base_name, f"Error: {str(e)}"))

    if not input_data:
        raise ValueError("No matching CSV pairs found.")

    print("\n=== Skipped Files Summary ===")
    for base_name, reason in skipped:
        print(f"{base_name}: {reason}")

    return np.vstack(input_data), np.vstack(target_data)

# === LOAD DATA ===
X_raw, y_raw = load_csv_pairs(POSE_CSV_DIR, KINECT_CSV_DIR)
print(f"\n✅ Loaded {X_raw.shape[0]} pose–kinect frame pairs")

# === SCALE ===
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X_raw)
y = scaler_y.fit_transform(y_raw)

# === SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === BUILD MODEL ===
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(y.shape[1])
])
mse_loss = MeanSquaredError()
model.compile(optimizer='adam', loss=mse_loss, metrics=['mae'])

# === TRAIN ===
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# === EVALUATE ===
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
print(f"\n📊 Final Evaluation:\nMSE: {mse:.4f}\nMAE: {mae:.4f}")

# === LOSS PLOT ===
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.show()

# === PREDICTED vs ACTUAL PLOT ===
plt.figure(figsize=(7, 7))
plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5)
plt.plot([min(y_true[:, 0]), max(y_true[:, 0])], [min(y_true[:, 0]), max(y_true[:, 0])], 'r--')
plt.title("Predicted vs Actual (First Joint X)")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.grid(True)
plt.show()

# === SAVE MODEL AND SCALERS ===
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
model_path = os.path.join(MODEL_SAVE_DIR, "pose_to_kinect_model.h5")
scaler_x_path = os.path.join(MODEL_SAVE_DIR, "scaler_X.pkl")
scaler_y_path = os.path.join(MODEL_SAVE_DIR, "scaler_y.pkl")

model.save(model_path)
joblib.dump(scaler_X, scaler_x_path)
joblib.dump(scaler_y, scaler_y_path)

print(f"\n✅ Model saved to: {model_path}")
print(f"✅ Scaler_X saved to: {scaler_x_path}")
print(f"✅ Scaler_y saved to: {scaler_y_path}")
