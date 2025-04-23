import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === CONFIGURATION ===
kinect_folder = r"C:\Users\Ashidow\Documents\DL A2\kinect_all_videos\all_videos\kinect_good_preprocessed"
output_model_path = "kinect2d_to_z_model.h5"

# Ordered joint names
joints = [
    'head', 'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow',
    'left_hand', 'right_hand', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_foot', 'right_foot'
]

def load_kinect_data():
    input_data = []
    target_data = []

    for file in os.listdir(kinect_folder):
        if not file.endswith(".csv"):
            continue

        path = os.path.join(kinect_folder, file)
        df = pd.read_csv(path)

        # 🧹 Normalize column names for consistent access
        df.columns = [col.lower().strip() for col in df.columns]

        for _, row in df.iterrows():
            input_row = []
            target_row = []

            for joint in joints:
                x_col = f"{joint}_x"
                y_col = f"{joint}_y"
                z_col = f"{joint}_z"

                # Use .get to safely fallback to 0.0 if missing
                input_row.append(row.get(x_col, 0.0))
                input_row.append(row.get(y_col, 0.0))
                target_row.append(row.get(z_col, 0.0))

            input_data.append(input_row)
            target_data.append(target_row)

    return np.array(input_data), np.array(target_data)

# === LOAD DATA ===
print("📦 Loading Kinect (xk, yk → zk) training data...")
X, y = load_kinect_data()
print(f"✅ Loaded {len(X)} samples with shape {X.shape} input and {y.shape} output.")

# === TRAIN/VAL SPLIT ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === MODEL DEFINITION ===
model = Sequential([
    Dense(128, input_shape=(26,), activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(13, activation='linear')  # 13 Z values
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()

# === TRAINING ===
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# === SAVE MODEL ===
model.save(output_model_path)
print(f"✅ Model saved to: {output_model_path}")
