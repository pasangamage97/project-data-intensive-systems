import os
import glob
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import LSTM, GRU, Reshape, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

DATA_PATH = r"C:\Users\Ashidow\Downloads\kinect_good_preprocessed\kinect_good_preprocessed"
OUTPUT_DIR = "models"  # Directory to save model and scalers
BATCH_SIZE = 32  # Batch size for training
EPOCHS = 100  # Maximum number of epochs
PATIENCE = 20  # Patience for early stopping
LEARNING_RATE = 0.001  # Initial learning rate

#'dense', 'conv1d', 'lstm', 'gru', or 'all'
MODEL_TYPE = 'all'

def load_and_preprocess_data(data_path):
    """
    Load and preprocess data from CSV file(s)
    
    Args:
        data_path: Path to either a single CSV file or a directory containing CSV files
        
    Returns:
        Processed input and output data
    """
    all_data = []
    
    if os.path.isdir(data_path):
        print(f"Loading all CSV files from directory: {data_path}")
        csv_files = glob.glob(os.path.join(data_path, "*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_path}")
        print(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                print(f"Processing {os.path.basename(csv_file)}...")
                df = pd.read_csv(csv_file)
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue
        
        # Combine all dataframes
        data = pd.concat(all_data, ignore_index=True)
        print(f"Combined data shape: {data.shape}")
        
    else:
        #single CSV file
        print(f"Loading data from file: {data_path}")
        try:
            data = pd.read_csv(data_path)
            print(f"Data shape: {data.shape}")
        except Exception as e:
            raise ValueError(f"Error loading file {data_path}: {e}")
    
    all_columns = data.columns
    x_columns = [col for col in all_columns if col.endswith('_x')]
    y_columns = [col for col in all_columns if col.endswith('_y')]
    z_columns = [col for col in all_columns if col.endswith('_z')]
    
    print(f"Found {len(x_columns)} x-columns, {len(y_columns)} y-columns, {len(z_columns)} z-columns")
    
    input_features = []
    for i in range(len(x_columns)):
        input_features.append(x_columns[i])
        input_features.append(y_columns[i])
    
    output_features = z_columns
    
    print(f"Input features ({len(input_features)}): {input_features}")
    print(f"Output features ({len(output_features)}): {output_features}")
    
    data = data.dropna(subset=input_features + output_features)
    print(f"Data shape after removing NaN values: {data.shape}")
    
    X = data[input_features].values
    y = data[output_features].values
    
    return X, y, len(x_columns)



#dense
def build_dense_model(input_dim, output_dim):
    """Build a dense neural network (MLP)"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        
        Dense(output_dim)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    return model


#conv1d
def build_conv1d_model(input_dim, output_dim, num_joints):
    """Build a 1D convolutional neural network"""
    
    model = Sequential([
        Reshape((num_joints, 2), input_shape=(input_dim,)),
        
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        
        Flatten(),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(output_dim)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_lstm_model(input_dim, output_dim, num_joints):
    model = Sequential([
        Reshape((num_joints, 2), input_shape=(input_dim,)),
        
        # LSTM layers
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(128),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        
        Dense(output_dim)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_gru_model(input_dim, output_dim, num_joints):
    model = Sequential([
        Reshape((num_joints, 2), input_shape=(input_dim,)),
        
        GRU(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        GRU(128),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        
        Dense(output_dim)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, model_name):

    # Ensure output directory exists
    model_dir = os.path.join(OUTPUT_DIR, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    checkpoint_path = os.path.join(model_dir, f'{model_name}_model.h5')
    callbacks = [
        # Early stopping 
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history, model_dir

def evaluate_model(model, X_test, y_test, scaler_y):

    # Get predictions
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_orig = scaler_y.inverse_transform(y_test)
    
    # Calculate MSE and MAE
    mse = np.mean(np.square(y_pred - y_test_orig))
    mae = np.mean(np.abs(y_pred - y_test_orig))
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    # Calculate per-joint error
    joint_mae = np.mean(np.abs(y_pred - y_test_orig), axis=0)
    
    print("\nMean Absolute Error per joint:")
    for i, error in enumerate(joint_mae):
        print(f"Joint {i+1}: {error:.4f}")
        
    return mse, mae

def plot_training_history(history, model_name):
    model_dir = os.path.join(OUTPUT_DIR, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # Plot training & validation loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Model Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title(f'{model_name} Model MAE')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f'{model_name}_training_history.png'))
    plt.close()

def save_model_and_scalers(model, scaler_X, scaler_y, model_dir):
    # Save model
    model_path = os.path.join(model_dir, 'depth_prediction_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save scalers
    scaler_X_path = os.path.join(model_dir, 'scaler_X.pkl')
    with open(scaler_X_path, 'wb') as f:
        pickle.dump(scaler_X, f)
    print(f"Input scaler saved to {scaler_X_path}")
    
    scaler_y_path = os.path.join(model_dir, 'scaler_y.pkl')
    with open(scaler_y_path, 'wb') as f:
        pickle.dump(scaler_y, f)
    print(f"Output scaler saved to {scaler_y_path}")

def compare_models(model_results):
    if len(model_results) <= 1:
        print("Not enough models to compare.")
        return
        
    # Create comparison directory
    compare_dir = os.path.join(OUTPUT_DIR, 'comparison')
    if not os.path.exists(compare_dir):
        os.makedirs(compare_dir)
    
    # Extract model names and metrics
    model_names = list(model_results.keys())
    mse_values = [results['mse'] for results in model_results.values()]
    mae_values = [results['mae'] for results in model_results.values()]
    
    # Plot MSE comparison
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, mse_values, color='skyblue')
    plt.title('MSE Comparison Between Models')
    plt.ylabel('Mean Squared Error')
    plt.xticks(rotation=45)
    for i, v in enumerate(mse_values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, 'mse_comparison.png'))
    plt.close()
    
    # Plot MAE comparison
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, mae_values, color='lightgreen')
    plt.title('MAE Comparison Between Models')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(rotation=45)
    for i, v in enumerate(mae_values):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, 'mae_comparison.png'))
    plt.close()
    
    # Print comparison results
    print("\n--- Model Comparison Results ---")
    print(f"{'Model Type':<10} {'MSE':<10} {'MAE':<10}")
    print("-" * 30)
    for model_name in model_names:
        print(f"{model_name:<10} {model_results[model_name]['mse']:<10.4f} {model_results[model_name]['mae']:<10.4f}")
    
    # Find the best model based on MAE
    best_model = model_names[mae_values.index(min(mae_values))]
    print(f"\nBest model based on MAE: {best_model}")
    
    # Create a copy of the best model to the main output directory
    best_model_dir = os.path.join(OUTPUT_DIR, best_model)
    best_model_path = os.path.join(best_model_dir, 'depth_prediction_model.h5')
    scaler_X_path = os.path.join(best_model_dir, 'scaler_X.pkl')
    scaler_y_path = os.path.join(best_model_dir, 'scaler_y.pkl')
    
    # Copy the best model to the main output directory
    import shutil
    shutil.copy(best_model_path, os.path.join(OUTPUT_DIR, 'depth_prediction_model.h5'))
    shutil.copy(scaler_X_path, os.path.join(OUTPUT_DIR, 'scaler_X.pkl'))
    shutil.copy(scaler_y_path, os.path.join(OUTPUT_DIR, 'scaler_y.pkl'))
    
    print(f"Copied the best model ({best_model}) to the main output directory.")
    
    # Save comparison results to a file
    with open(os.path.join(compare_dir, 'comparison_results.txt'), 'w') as f:
        f.write("--- Model Comparison Results ---\n")
        f.write(f"{'Model Type':<10} {'MSE':<10} {'MAE':<10}\n")
        f.write("-" * 30 + "\n")
        for model_name in model_names:
            f.write(f"{model_name:<10} {model_results[model_name]['mse']:<10.4f} {model_results[model_name]['mae']:<10.4f}\n")
        f.write(f"\nBest model based on MAE: {best_model}\n")

def main():
    """Main function to train and evaluate different model architectures"""
    print("Starting training process...")
    print(f"Using data from: {DATA_PATH}")
    print(f"Model type: {MODEL_TYPE}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, num_joints = load_and_preprocess_data(DATA_PATH)
    
    # Split data into training, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    
    print(f"Training set shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
    print(f"Test set shape: {X_test.shape}, {y_test.shape}")
    
    # Normalize data
    print("Normalizing data...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)
    
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Define input and output dimensions
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    # Dictionary to store model results for comparison
    model_results = {}
    
    # Train models based on the selected model type
    if MODEL_TYPE == 'all' or MODEL_TYPE == 'dense':
        print("\n=== Training Dense (MLP) Model ===")
        model = build_dense_model(input_dim, output_dim)
        model.summary()
        
        trained_model, history, model_dir = train_model(
            model, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, 'dense')
        
        print("Evaluating Dense model...")
        mse, mae = evaluate_model(trained_model, X_test_scaled, y_test_scaled, scaler_y)
        
        plot_training_history(history, 'dense')
        save_model_and_scalers(trained_model, scaler_X, scaler_y, model_dir)
        
        model_results['dense'] = {'mse': mse, 'mae': mae}
    
    if MODEL_TYPE == 'all' or MODEL_TYPE == 'conv1d':
        print("\n=== Training Conv1D Model ===")
        model = build_conv1d_model(input_dim, output_dim, num_joints)
        model.summary()
        
        trained_model, history, model_dir = train_model(
            model, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, 'conv1d')
        
        print("Evaluating Conv1D model...")
        mse, mae = evaluate_model(trained_model, X_test_scaled, y_test_scaled, scaler_y)
        
        plot_training_history(history, 'conv1d')
        save_model_and_scalers(trained_model, scaler_X, scaler_y, model_dir)
        
        model_results['conv1d'] = {'mse': mse, 'mae': mae}
    
    if MODEL_TYPE == 'all' or MODEL_TYPE == 'lstm':
        print("\n=== Training LSTM Model ===")
        model = build_lstm_model(input_dim, output_dim, num_joints)
        model.summary()
        
        trained_model, history, model_dir = train_model(
            model, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, 'lstm')
        
        print("Evaluating LSTM model...")
        mse, mae = evaluate_model(trained_model, X_test_scaled, y_test_scaled, scaler_y)
        
        plot_training_history(history, 'lstm')
        save_model_and_scalers(trained_model, scaler_X, scaler_y, model_dir)
        
        model_results['lstm'] = {'mse': mse, 'mae': mae}
    
    if MODEL_TYPE == 'all' or MODEL_TYPE == 'gru':
        print("\n=== Training GRU Model ===")
        model = build_gru_model(input_dim, output_dim, num_joints)
        model.summary()
        
        trained_model, history, model_dir = train_model(
            model, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, 'gru')
        
        print("Evaluating GRU model...")
        mse, mae = evaluate_model(trained_model, X_test_scaled, y_test_scaled, scaler_y)
        
        plot_training_history(history, 'gru')
        save_model_and_scalers(trained_model, scaler_X, scaler_y, model_dir)
        
        model_results['gru'] = {'mse': mse, 'mae': mae}
    
    # Compare models if multiple models were trained
    if len(model_results) > 1:
        compare_models(model_results)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred during training: {e}")
        import traceback
        traceback.print_exc()