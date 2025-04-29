import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, gaussian_kde
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from .drift_monitor import DriftMonitor

class NeuralDriftMonitor(DriftMonitor):
    """
    Neural network-based drift monitoring class that extends the base DriftMonitor.
    
    This class implements drift detection using neural network models:
    - Multilayer Perceptron (MLP)
    - Long Short-Term Memory (LSTM)
    - Convolutional Neural Network (CNN)
    
    It inherits KS test and PSI calculation methods from the base class.
    """
    
    def __init__(self, window_size=500, threshold=0.05, psi_threshold=0.2, n_bins=10):
        """
        Initialize the NeuralDriftMonitor with parameters.
        
        Args:
            window_size (int): Size of data windows to use for drift detection
            threshold (float): p-value threshold for KS test (lower values indicate drift)
            psi_threshold (float): Threshold for PSI (higher values indicate drift)
            n_bins (int): Number of bins to use for PSI calculation
        """
        super().__init__(window_size, threshold, psi_threshold, n_bins)

        # Neural network models dictionary
        self.nn_models = {}

    def create_mlp(self, input_shape, epochs=20, batch_size=64):
        """
        Create and initialize an MLP model
        
        Args:
            input_shape (int): Number of input features
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            keras.Model: Initialized MLP model
        """
        print("Creating MLP model...")
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

        return model

    def create_lstm(self, input_shape, epochs=20, batch_size=64):
        """
        Create and initialize an LSTM model
        
        Args:
            input_shape (int): Number of input features
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            keras.Model: Initialized LSTM model
        """
        print("Creating LSTM model...")
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(1, input_shape)),
            Dropout(0.2),
            LSTM(25),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

        return model

    def create_cnn(self, input_shape, epochs=20, batch_size=64):
        """
        Create and initialize a 1D CNN model
        
        Args:
            input_shape (int): Number of input features
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            keras.Model: Initialized 1D CNN model
        """
        print("Creating CNN model...")
        model = Sequential([
            Conv1D(32, kernel_size=3, activation='relu', input_shape=(input_shape, 1)),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            Conv1D(64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

        return model

    def detect_drift_with_nn(self, X_train, y_train, X_test, y_test):
        """
        Detect drift using neural network models with fixed reference window
        
        Args:
            X_train (array-like): Training features
            y_train (array-like): Training labels
            X_test (array-like): Test features to check for drift
            y_test (array-like): Test labels
            
        Returns:
            dict: Results for each neural network model
        """
        X_train = self._ensure_numpy(X_train)
        X_test = self._ensure_numpy(X_test)
        y_train = self._ensure_numpy(y_train)
        y_test = self._ensure_numpy(y_test)

        # Make sure y is in correct shape
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)

        # Create and train neural network models
        # MLP
        mlp_model = self.create_mlp(X_train.shape[1])
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        mlp_model.fit(X_train, y_train, epochs=15, batch_size=128,
                    validation_split=0.2, callbacks=[early_stopping], verbose=0)
        self.nn_models['MLP'] = mlp_model

        # LSTM (reshape input)
        lstm_model = self.create_lstm(X_train.shape[1])
        X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        lstm_model.fit(X_train_lstm, y_train, epochs=15, batch_size=128,
                      validation_split=0.2, callbacks=[early_stopping], verbose=0)
        self.nn_models['LSTM'] = lstm_model

        # CNN (reshape input)
        cnn_model = self.create_cnn(X_train.shape[1])
        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        cnn_model.fit(X_train_cnn, y_train, epochs=15, batch_size=128,
                    validation_split=0.2, callbacks=[early_stopping], verbose=0)
        self.nn_models['CNN'] = cnn_model

        # Initialize storage for results
        results = {
            'MLP': {'drift_scores': [], 'psi_scores': [], 'accuracies': []},
            'LSTM': {'drift_scores': [], 'psi_scores': [], 'accuracies': []},
            'CNN': {'drift_scores': [], 'psi_scores': [], 'accuracies': []}
        }

        # Define fixed reference window
        fixed_window_size = min(self.window_size, len(X_train))
        fixed_reference_window, _, y_ref, _ = train_test_split(
            X_train, y_train, train_size=fixed_window_size/len(X_train),
            stratify=y_train, random_state=42
        )

        # Process windows for each model
        for i in range(0, len(X_test), self.window_size):
            if i + self.window_size <= len(X_test):
                current_window = X_test[i:i + self.window_size]
                y_window = y_test[i:i + self.window_size]

                # Process with each neural network model
                for model_name in ['MLP', 'LSTM', 'CNN']:
                    model = self.nn_models[model_name]

                    # Get predictions according to model type
                    if model_name == 'MLP':
                        pred1 = model.predict(fixed_reference_window, verbose=0).ravel()
                        pred2 = model.predict(current_window, verbose=0).ravel()
                    elif model_name == 'LSTM':
                        pred1 = model.predict(fixed_reference_window.reshape(fixed_reference_window.shape[0], 1, fixed_reference_window.shape[1]), verbose=0).ravel()
                        pred2 = model.predict(current_window.reshape(current_window.shape[0], 1, current_window.shape[1]), verbose=0).ravel()
                    elif model_name == 'CNN':
                        pred1 = model.predict(fixed_reference_window.reshape(fixed_reference_window.shape[0], fixed_reference_window.shape[1], 1), verbose=0).ravel()
                        pred2 = model.predict(current_window.reshape(current_window.shape[0], current_window.shape[1], 1), verbose=0).ravel()

                    # KS test
                    _, ks_p_value = ks_2samp(pred1, pred2)
                    is_ks_drift = ks_p_value < self.threshold

                    # PSI calculation
                    psi_value = self._calculate_psi(pred1, pred2)
                    is_psi_drift = psi_value > self.psi_threshold

                    # Visualize first window or when drift is detected
                    if i == 0 or (is_ks_drift or is_psi_drift):
                        plt.figure(figsize=(15, 15))

                        # Plot histogram
                        plt.subplot(2, 1, 1)
                        plt.hist(pred1, bins=20, alpha=0.7, label='Fixed Reference Window', color='blue', density=True)
                        plt.hist(pred2, bins=20, alpha=0.7, label='Current Window', color='orange', density=True)
                        drift_status = "DRIFT DETECTED" if (is_ks_drift or is_psi_drift) else "NO DRIFT"
                        plt.title(f'{model_name} - {drift_status} (KS p-value: {ks_p_value:.4f}, PSI: {psi_value:.4f})')
                        plt.legend()
                        plt.grid(alpha=0.3)

                        # Plot KDE
                        plt.subplot(2, 1, 2)
                        x_range = np.linspace(0, 1, 1000)

                        if len(pred1) > 1:
                            kde1 = gaussian_kde(pred1)
                            plt.plot(x_range, kde1(x_range), color='blue', linewidth=2, label='Fixed Reference Window')

                        if len(pred2) > 1:
                            kde2 = gaussian_kde(pred2)
                            plt.plot(x_range, kde2(x_range), color='orange', linewidth=2, label='Current Window')

                        plt.title(f'{model_name} Probability Density Function')
                        plt.legend()
                        plt.grid(alpha=0.3)

                        plt.tight_layout()
                        plt.show()

                    # Calculate accuracy
                    if model_name == 'MLP':
                        y_pred = (model.predict(current_window, verbose=0) > 0.5).astype(int).ravel()
                    elif model_name == 'LSTM':
                        current_window_lstm = current_window.reshape(current_window.shape[0], 1, current_window.shape[1])
                        y_pred = (model.predict(current_window_lstm, verbose=0) > 0.5).astype(int).ravel()
                    elif model_name == 'CNN':
                        current_window_cnn = current_window.reshape(current_window.shape[0], current_window.shape[1], 1)
                        y_pred = (model.predict(current_window_cnn, verbose=0) > 0.5).astype(int).ravel()

                    accuracy = accuracy_score(y_window, y_pred)

                    # Store results
                    results[model_name]['drift_scores'].append(ks_p_value)
                    results[model_name]['psi_scores'].append(psi_value)
                    results[model_name]['accuracies'].append(accuracy)

                # No need to update the reference window as we're using a fixed reference window

        # Plot summary results
        self.plot_neural_network_results(results)

        return results

    def plot_neural_network_results(self, results):
        """
        Plot comparative results for neural network models
        
        Args:
            results (dict): Results for each neural network model
        """
        # Set up colors for each model
        colors = {
            'MLP': 'blue',
            'LSTM': 'green',
            'CNN': 'purple'
        }

        plt.figure(figsize=(15, 12))

        # Calculate x-axis points
        num_windows = len(list(results.values())[0]['drift_scores'])
        x_points = np.arange(num_windows)

        # Plot 1: KS Drift Scores
        plt.subplot(3, 1, 1)
        for name, res in results.items():
            plt.plot(x_points, res['drift_scores'], label=f'{name}',
                    color=colors[name], linewidth=2)

        plt.axhline(y=self.threshold, color='r', linestyle='--', label='KS Threshold', linewidth=2)
        plt.title(f'KS Test Drift Scores Comparison (Neural Networks)', fontsize=16)
        plt.xlabel('Window Index', fontsize=14)
        plt.ylabel('KS Score (p-value)', fontsize=14)
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: PSI Scores
        plt.subplot(3, 1, 2)
        for name, res in results.items():
            plt.plot(x_points, res['psi_scores'], label=f'{name}',
                    color=colors[name], linewidth=2)

        plt.axhline(y=self.psi_threshold, color='r', linestyle='--', label='PSI Threshold', linewidth=2)
        plt.title(f'PSI Scores Comparison', fontsize=16)
        plt.xlabel('Window Index', fontsize=14)
        plt.ylabel('PSI Value', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 3: Accuracy
        plt.subplot(3, 1, 3)
        for name, res in results.items():
            plt.plot(x_points, res['accuracies'], label=f'{name}',
                    color=colors[name], linewidth=2)

        plt.title(f'Model Accuracy Comparison', fontsize=16)
        plt.xlabel('Window Index', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'neural_networks_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()