import os
import glob
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import LSTM, GRU, Reshape, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

DATA_PATH = r"C:\Users\Ashidow\Downloads\kinect_good_preprocessed\kinect_good_preprocessed"
OUTPUT_DIR = "models"  # Directory to save model and scalers
BATCH_SIZE = 32  # Batch size for training
EPOCHS = 100  # Maximum number of epochs
PATIENCE = 20  # Patience for early stopping
LEARNING_RATE = 0.001  # Initial learning rate

# Choose one of: 'dense', 'conv1d', 'lstm', 'gru', or 'all'
MODEL_TYPE = 'gru'

# Primary loss function - choose one of: 'mse', 'mae', 'rss'
PRIMARY_LOSS = 'rss'

# Number of random sequences for testing
NUM_TEST_SEQUENCES = 10

# Define custom loss functions
def rss_loss(y_true, y_pred):
    """Residual Sum of Squares (RSS) loss function"""
    return tf.reduce_sum(tf.square(y_true - y_pred))

# Get the loss function based on the selected PRIMARY_LOSS
def get_loss_function(loss_type):
    if loss_type == 'mse':
        return 'mse'
    elif loss_type == 'mae':
        return 'mae'
    elif loss_type == 'rss':
        return rss_loss
    else:
        print(f"Warning: Unknown loss function '{loss_type}', using default 'mse'")
        return 'mse'

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

# Model building functions
def build_dense_model(input_dim, output_dim):
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
        loss=get_loss_function(PRIMARY_LOSS),
        metrics=['mae', 'mse']
    )
    
    return model


def build_conv1d_model(input_dim, output_dim, num_joints):
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
        loss=get_loss_function(PRIMARY_LOSS),
        metrics=['mae', 'mse']
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
        loss=get_loss_function(PRIMARY_LOSS),
        metrics=['mae', 'mse']
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
        
        Dense(64, activation='sigmoid'),
        BatchNormalization(),
        
        Dense(output_dim)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=get_loss_function(PRIMARY_LOSS),
        metrics=['mae', 'mse']
    )
    
    return model

def calculate_metrics(y_true, y_pred):
    """Calculate all loss metrics for evaluation"""
    # Standard metrics
    mse = np.mean(np.square(y_true - y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    rss = np.sum(np.square(y_true - y_pred))
    
    return {
        'mse': mse,
        'mae': mae,
        'rss': rss
    }

def train_model(model, X_train, y_train, X_val, y_val, model_name):
    # Ensure output directory exists
    model_dir = os.path.join(OUTPUT_DIR, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    checkpoint_path = os.path.join(model_dir, f'{model_name}_model.h5')
    callbacks = [
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

def test_model_on_random_sequences(model, X, y, scaler_X, scaler_y, num_sequences=10, test_size=0.2):
    """
    Test the model on multiple random sequences of the data
    
    Args:
        model: Trained model
        X: Input features
        y: Target values
        scaler_X: Feature scaler
        scaler_y: Target scaler
        num_sequences: Number of random sequences to test on
        test_size: Size of each test set (fraction of total data)
        
    Returns:
        List of metric dictionaries, one for each test sequence
    """
    results = []
    
    print(f"\nTesting model on {num_sequences} random sequences...")
    
    for i in range(num_sequences):
        # Create a different random split each time
        random_seed = 42 + i  # Use different seeds
        _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
        
        # Scale test data
        X_test_scaled = scaler_X.transform(X_test)
        y_test_scaled = scaler_y.transform(y_test)
        
        # Get predictions
        y_pred_scaled = model.predict(X_test_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_test_orig = scaler_y.inverse_transform(y_test_scaled)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test_orig, y_pred)
        
        # Calculate per-joint error (just for the first sequence)
        if i == 0:
            joint_mae = np.mean(np.abs(y_pred - y_test_orig), axis=0)
            print("\nMean Absolute Error per joint (first random sequence):")
            for j, error in enumerate(joint_mae):
                print(f"Joint {j+1}: {error:.4f}")
        
        print(f"Sequence {i+1}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, RSS={metrics['rss']:.4f}")
        results.append(metrics)
    
    # Calculate average metrics across all sequences
    avg_mse = np.mean([result['mse'] for result in results])
    avg_mae = np.mean([result['mae'] for result in results])
    avg_rss = np.mean([result['rss'] for result in results])
    
    std_mse = np.std([result['mse'] for result in results])
    std_mae = np.std([result['mae'] for result in results])
    std_rss = np.std([result['rss'] for result in results])
    
    print("\n--- Average Metrics Across All Random Sequences ---")
    print(f"MSE: {avg_mse:.4f} ± {std_mse:.4f}")
    print(f"MAE: {avg_mae:.4f} ± {std_mae:.4f}")
    print(f"RSS: {avg_rss:.4f} ± {std_rss:.4f}")
    
    return results, {
        'avg_mse': avg_mse, 'std_mse': std_mse,
        'avg_mae': avg_mae, 'std_mae': std_mae,
        'avg_rss': avg_rss, 'std_rss': std_rss
    }

def plot_training_history(history, model_name):
    model_dir = os.path.join(OUTPUT_DIR, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Create a more comprehensive plot with multiple metrics
    plt.figure(figsize=(15, 10))
    
    # Plot training & validation loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Model Loss ({PRIMARY_LOSS})')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot MAE
    plt.subplot(2, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title(f'{model_name} Mean Absolute Error')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot MSE
    plt.subplot(2, 2, 3)
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title(f'{model_name} Mean Squared Error')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot learning rate if it was changed during training
    if 'lr' in history.history:
        plt.subplot(2, 2, 4)
        plt.plot(history.history['lr'])
        plt.title('Learning Rate')
        plt.ylabel('Learning Rate')
        plt.xlabel('Epoch')
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f'{model_name}_training_history.png'))
    plt.close()

def plot_random_sequence_results(all_results, model_name):
    """Plot the distribution of metrics across random test sequences"""
    model_dir = os.path.join(OUTPUT_DIR, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    metrics = ['mse', 'mae', 'rss']
    
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        
        # Extract values for this metric from all sequences
        values = [result[metric] for result in all_results]
        
        # Box plot
        plt.boxplot(values)
        plt.title(f'Distribution of {metric.upper()} Across Random Sequences')
        plt.ylabel(metric.upper())
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add individual points
        plt.plot(np.ones(len(values)) + np.random.normal(0, 0.05, len(values)), 
                 values, 'ko', alpha=0.3)
        
        # Add mean line
        plt.axhline(y=np.mean(values), color='r', linestyle='-', 
                    label=f'Mean: {np.mean(values):.4f}')
        
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f'{model_name}_random_sequences_results.png'))
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
    metrics_to_plot = ['avg_mse', 'avg_mae', 'avg_rss']
    
    plt.figure(figsize=(18, 12))
    
    for idx, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 2, idx+1)
        values = [results[metric] for results in model_results.values()]
        errors = [results[metric.replace('avg_', 'std_')] for results in model_results.values()]
        
        # For RSS, use log scale as values can be very large
        if 'rss' in metric:
            plt.yscale('log')
        
        # Bar plot with error bars
        bars = plt.bar(model_names, values, yerr=errors, alpha=0.7, 
                        capsize=10, error_kw={'elinewidth': 2, 'capthick': 2})
                        
        plt.title(f'{metric.replace("avg_", "").upper()} Comparison')
        plt.ylabel(metric.replace("avg_", "").upper())
        plt.xticks(rotation=45)
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{values[i]:.4f}',
                    ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, 'metrics_comparison.png'))
    plt.close()
    
    # Print comparison results in a table
    print("\n--- Model Comparison Results (Averaged Across Random Sequences) ---")
    header = f"{'Model':<10}"
    for metric in metrics_to_plot:
        header += f" | {metric.replace('avg_', '').upper():<15}"
    print(header)
    print("-" * len(header))
    
    for model_name in model_names:
        row = f"{model_name:<10}"
        for metric in metrics_to_plot:
            std_metric = metric.replace('avg_', 'std_')
            row += f" | {model_results[model_name][metric]:<8.4f} ± {model_results[model_name][std_metric]:<5.4f}"
        print(row)
    
    # Find the best model based on the primary loss function
    best_metric = f"avg_{PRIMARY_LOSS}"
    best_model_idx = np.argmin([results[best_metric] for results in model_results.values()])
    best_model = model_names[best_model_idx]
    print(f"\nBest model based on {PRIMARY_LOSS}: {best_model}")
    
    # Save comparison results to a file
    with open(os.path.join(compare_dir, 'comparison_results.txt'), 'w') as f:
        f.write("--- Model Comparison Results (Averaged Across Random Sequences) ---\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for model_name in model_names:
            row = f"{model_name:<10}"
            for metric in metrics_to_plot:
                std_metric = metric.replace('avg_', 'std_')
                row += f" | {model_results[model_name][metric]:<8.4f} ± {model_results[model_name][std_metric]:<5.4f}"
            f.write(row + "\n")
        f.write(f"\nBest model based on {PRIMARY_LOSS}: {best_model}\n")

def main():
    """Main function to train and evaluate different model architectures"""
    print("Starting training process...")
    print(f"Using data from: {DATA_PATH}")
    print(f"Model type: {MODEL_TYPE}")
    print(f"Primary loss function: {PRIMARY_LOSS}")
    print(f"Number of random test sequences: {NUM_TEST_SEQUENCES}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, num_joints = load_and_preprocess_data(DATA_PATH)
    
    # Split data into training, validation
    # We'll use 60% for training, 20% for validation, and reserve 20% for random sequence testing
    X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, _, y_val, _ = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=42)
    
    print(f"Training set shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
    
    # Normalize data
    print("Normalizing data...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)
    
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
        
        print("Testing Dense model on random sequences...")
        all_metrics, avg_metrics = test_model_on_random_sequences(
            trained_model, X, y, scaler_X, scaler_y, NUM_TEST_SEQUENCES)
        
        plot_training_history(history, 'dense')
        plot_random_sequence_results(all_metrics, 'dense')
        save_model_and_scalers(trained_model, scaler_X, scaler_y, model_dir)
        
        model_results['dense'] = avg_metrics
    
    if MODEL_TYPE == 'all' or MODEL_TYPE == 'conv1d':
        print("\n=== Training Conv1D Model ===")
        model = build_conv1d_model(input_dim, output_dim, num_joints)
        model.summary()
        
        trained_model, history, model_dir = train_model(
            model, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, 'conv1d')
        
        print("Testing Conv1D model on random sequences...")
        all_metrics, avg_metrics = test_model_on_random_sequences(
            trained_model, X, y, scaler_X, scaler_y, NUM_TEST_SEQUENCES)
        
        plot_training_history(history, 'conv1d')
        plot_random_sequence_results(all_metrics, 'conv1d')
        save_model_and_scalers(trained_model, scaler_X, scaler_y, model_dir)
        
        model_results['conv1d'] = avg_metrics
    
    if MODEL_TYPE == 'all' or MODEL_TYPE == 'lstm':
        print("\n=== Training LSTM Model ===")
        model = build_lstm_model(input_dim, output_dim, num_joints)
        model.summary()
        
        trained_model, history, model_dir = train_model(
            model, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, 'lstm')
        
        print("Testing LSTM model on random sequences...")
        all_metrics, avg_metrics = test_model_on_random_sequences(
            trained_model, X, y, scaler_X, scaler_y, NUM_TEST_SEQUENCES)
        
        plot_training_history(history, 'lstm')
        plot_random_sequence_results(all_metrics, 'lstm')
        save_model_and_scalers(trained_model, scaler_X, scaler_y, model_dir)
        
        model_results['lstm'] = avg_metrics
    
    if MODEL_TYPE == 'all' or MODEL_TYPE == 'gru':
        print("\n=== Training GRU Model ===")
        model = build_gru_model(input_dim, output_dim, num_joints)
        model.summary()
        
        trained_model, history, model_dir = train_model(
            model, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, 'gru')
        
        print("Testing GRU model on random sequences...")
        all_metrics, avg_metrics = test_model_on_random_sequences(
            trained_model, X, y, scaler_X, scaler_y, NUM_TEST_SEQUENCES)
        
        plot_training_history(history, 'gru')
        plot_random_sequence_results(all_metrics, 'gru')
        save_model_and_scalers(trained_model, scaler_X, scaler_y, model_dir)
        
        model_results['gru'] = avg_metrics
    
    # Compare models if multiple models were trained
    if len(model_results) > 1:
        compare_models(model_results)
    
    print("\nTraining and testing on random sequences completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred during training: {e}")
        import traceback
        traceback.print_exc()