import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import copy
import io
import base64
import os
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.stats import ks_2samp

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Some functionality will be limited.")

from .data_utils import split_into_windows, inject_distorting_noise

class SimpleModel(nn.Module):
    """
    Simple PyTorch neural network model for drift detection.
    """
    def __init__(self, input_dim, hidden_dim=64):
        """
        Initialize the model architecture.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Size of hidden layers
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Output predictions
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

def functional_drift_score(model_a, model_b, test_data, num_samples=1000):
    """
    Calculate functional drift by comparing predictions from two models.
    
    Args:
        model_a (nn.Module): First model
        model_b (nn.Module): Second model
        test_data (DataFrame): Test data for predictions
        num_samples (int): Maximum number of samples to use
        
    Returns:
        tuple: (prediction_difference, disagreement_rate)
    """
    if len(test_data) > num_samples:
        test_subset = test_data.sample(num_samples, random_state=42)
    else:
        test_subset = test_data

    # Get predictions for reference data
    X_ref = ref_subset.drop('label', axis=1).values.astype(np.float32)
    scaler_ref = StandardScaler()
    X_ref_scaled = scaler_ref.fit_transform(X_ref)
    X_ref_tensor = torch.tensor(X_ref_scaled)

    # Get predictions for test data
    X_test = test_subset.drop('label', axis=1).values.astype(np.float32)
    scaler_test = StandardScaler()
    X_test_scaled = scaler_test.fit_transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled)

    model.eval()
    with torch.no_grad():
        preds_ref = model(X_ref_tensor).numpy().flatten()
        preds_test = model(X_test_tensor).numpy().flatten()

    # Calculate PSI
    psi_value = calculate_psi(preds_ref, preds_test, n_bins=n_bins)

    return psi_value

def train_model(model, df, max_epochs=50, patience=5, min_delta=0.001, lr=0.001):
    """
    Train a PyTorch model on the given data with early stopping.
    
    Args:
        model (nn.Module): Model to train
        df (DataFrame): Training data
        max_epochs (int): Maximum number of epochs
        patience (int): Early stopping patience
        min_delta (float): Minimum improvement for early stopping
        lr (float): Learning rate
        
    Returns:
        tuple: (training_losses, validation_losses)
    """
    model.train()
    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values.astype(np.float32).reshape(-1, 1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    val_size = int(0.2 * len(X))
    train_size = len(X) - val_size
    indices = np.random.permutation(len(X))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=32, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    best_model = None
    counter = 0
    losses = []
    val_losses = []

    for epoch in range(max_epochs):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.tensor(X_val))
            val_loss = criterion(val_pred, torch.tensor(y_val)).item()
            val_losses.append(val_loss)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                if best_model is not None:
                    model.load_state_dict(best_model)
                break

    if best_model is not None and counter < patience:
        model.load_state_dict(best_model)

    return losses, val_losses

def evaluate_model(model, df):
    """
    Evaluate a PyTorch model on the given data.
    
    Args:
        model (nn.Module): Model to evaluate
        df (DataFrame): Evaluation data
        
    Returns:
        float: Accuracy
    """
    model.eval()
    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values.astype(np.float32).reshape(-1, 1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    with torch.no_grad():
        preds = model(torch.tensor(X))
        pred_probs = preds.numpy()
        pred_labels = (pred_probs > 0.5).astype(int)
    accuracy = accuracy_score(y, pred_labels)
    return accuracy

def aggregate_models(client_models):
    """
    Average the weights of client models to create a global model (federated learning).
    
    Args:
        client_models (list): List of client models
        
    Returns:
        dict: Aggregated model state dictionary
    """
    global_state = {}
    for key in client_models[0].state_dict().keys():
        params = torch.stack([client.state_dict()[key] for client in client_models])
        global_state[key] = torch.mean(params, dim=0)
    return global_state

class DriftDetectionSimulator:
    """
    Interactive simulator for drift detection with noise injection capabilities.
    """
    def __init__(self):
        """Initialize the simulator."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for the drift detection simulator.")
        
        self.setup_data()
        self.setup_models()
        self.setup_ui()

        # Storage for noise injection parameters
        self.client_noise_settings = {}

    def setup_data(self):
        """Load and prepare data for the simulation."""
        print("Loading data...")
        try:
            self.df1 = pd.read_csv("data/NSL_KDD_binary_train.csv")
            self.df2 = pd.read_csv("data/NSL_KDD_binary_test.csv")
        except FileNotFoundError:
            # Try alternate location
            try:
                self.df1 = pd.read_csv("NSL_KDD_binary_train.csv")
                self.df2 = pd.read_csv("NSL_KDD_binary_test.csv")
            except FileNotFoundError:
                raise FileNotFoundError("Could not find NSL-KDD dataset files. Make sure they exist in the data directory or current directory.")

        # Split train data
        X_full_train = self.df1.drop(['label'], axis=1)
        y_full_train = self.df1['label']
        X_train, X_val, y_train, y_val = train_test_split(X_full_train, y_full_train, test_size=0.3, stratify=y_full_train, random_state=42)

        X_val_front, X_val_end, y_val_front, y_val_end = train_test_split(X_val, y_val, test_size=0.5, stratify=y_val, random_state=42)
        self.df_val_front = pd.DataFrame(X_val_front)
        self.df_val_front['label'] = y_val_front.values
        self.df_val_end = pd.DataFrame(X_val_end)
        self.df_val_end['label'] = y_val_end.values

        # Store clean data for later injection
        self.df_val_end_clean = self.df_val_end.copy()

        # Create initial training data
        self.stream1_data = pd.DataFrame(X_train)
        self.stream1_data['label'] = y_train.values

        # Prepare client data splits
        train_size = len(self.stream1_data)
        client_1_end = train_size // 3
        client_2_end = 2 * (train_size // 3)

        self.client_train_dfs = [
            self.stream1_data.iloc[:client_1_end].reset_index(drop=True),
            self.stream1_data.iloc[client_1_end:client_2_end].reset_index(drop=True),
            self.stream1_data.iloc[client_2_end:].reset_index(drop=True)
        ]

        self.input_dim = X_full_train.shape[1]

    def setup_models(self):
        """Initialize models for the simulation."""
        print("Setting up models...")
        self.global_model = SimpleModel(self.input_dim)

        # Check if model exists, if not train and save
        MODEL_PATH = 'models/global_model.pth'
        if os.path.exists(MODEL_PATH):
            print("Loading existing global model...")
            self.global_model.load_state_dict(torch.load(MODEL_PATH))
        else:
            # Check if models directory exists, if not create it
            if not os.path.exists('models'):
                os.makedirs('models')
                
            print("Training new global model...")
            client_models = [SimpleModel(self.input_dim) for _ in range(3)]
            for i, (client_model, client_train_df) in enumerate(zip(client_models, self.client_train_dfs)):
                print(f"Initial training for Client {i+1}")
                train_model(client_model, client_train_df, max_epochs=50, patience=5, min_delta=0.001)

            global_state = aggregate_models(client_models)
            self.global_model.load_state_dict(global_state)
            torch.save(self.global_model.state_dict(), MODEL_PATH)
            print(f"Global model saved to {MODEL_PATH}")

    def setup_ui(self):
        """Create the interactive UI for the simulator."""
        # Initialize windows button
        self.init_windows_button = widgets.Button(
            description='Initialize Windows',
            button_style='info'
        )
        self.init_windows_button.on_click(self.initialize_windows)

        # Client noise configuration buttons
        self.client_config_buttons = []
        for i in range(3):
            button = widgets.Button(
                description=f'Configure Client {i+1}',
                button_style='primary',
                disabled=True
            )
            button.on_click(lambda b, client=i: self.configure_client_noise(client))
            self.client_config_buttons.append(button)

        # Run button for noise configuration
        self.run_button = widgets.Button(
            description='Run with Noise',
            button_style='success',
            disabled=True
        )
        self.run_button.on_click(self.run_simulation)

        # Run button without noise
        self.run_without_noise_button = widgets.Button(
            description='Run Without Noise',
            button_style='warning',
            disabled=True
        )
        self.run_without_noise_button.on_click(self.run_simulation_without_noise)

        # Output area
        self.output_area = widgets.Output()

        # Layout
        client_buttons_box = widgets.HBox(self.client_config_buttons)
        run_buttons_box = widgets.HBox([self.run_button, self.run_without_noise_button])

        self.ui = widgets.VBox([
            self.init_windows_button,
            widgets.HTML('<h3>Configure Noise for Each Client</h3>'),
            client_buttons_box,
            run_buttons_box,
            self.output_area
        ])

    def initialize_windows(self, _):
        """Initialize windows for selection."""
        # Create the test data stream
        self.stream2_data = pd.concat([self.df_val_front, self.df2, self.df_val_front, self.df_val_end_clean], ignore_index=True)

        # Split into windows
        self.stream2_windows = split_into_windows(self.stream2_data, window_size=500)
        self.total_windows = len(self.stream2_windows)

        # Calculate window ranges for each client
        client_1_windows_end = self.total_windows // 3
        client_2_windows_end = 2 * (self.total_windows // 3)

        self.client_window_ranges = [
            (0, client_1_windows_end),
            (client_1_windows_end, client_2_windows_end),
            (client_2_windows_end, self.total_windows)
        ]

        # Enable configuration buttons
        for button in self.client_config_buttons:
            button.disabled = False

        # Enable both run buttons
        self.run_button.disabled = False
        self.run_without_noise_button.disabled = False

        # Initialize noise settings
        self.client_noise_settings = {i: {'enable': False, 'windows': [], 'noise_scale': 1.0, 'scaling_range': (0.5, 2.0)} for i in range(3)}

        with self.output_area:
            clear_output(wait=True)
            print(f"Initialized {self.total_windows} windows.")
            print(f"Client 1 range: {self.client_window_ranges[0][0]}-{self.client_window_ranges[0][1]-1}")
            print(f"Client 2 range: {self.client_window_ranges[1][0]}-{self.client_window_ranges[1][1]-1}")
            print(f"Client 3 range: {self.client_window_ranges[2][0]}-{self.client_window_ranges[2][1]-1}")
            print("\nYou can now:")
            print("1. Click 'Run Without Noise' to run simulation with clean data")
            print("2. Configure noise for clients and click 'Run with Noise'")

    def configure_client_noise(self, client_idx):
        """
        Configure noise settings for a specific client.
        
        Args:
            client_idx (int): Index of client to configure
        """
        with self.output_area:
            clear_output(wait=True)
            print(f"Configuring noise for Client {client_idx + 1}")

            # Create configuration UI
            enable_noise = widgets.Checkbox(
                value=self.client_noise_settings[client_idx]['enable'],
                description='Enable Noise Injection'
            )

            noise_scale = widgets.FloatSlider(
                value=self.client_noise_settings[client_idx]['noise_scale'],
                min=0.0, max=5.0, step=0.1,
                description='Noise Scale:'
            )

            scaling_min = widgets.FloatSlider(
                value=self.client_noise_settings[client_idx]['scaling_range'][0],
                min=0.1, max=1.0, step=0.1,
                description='Scale Min:'
            )

            scaling_max = widgets.FloatSlider(
                value=self.client_noise_settings[client_idx]['scaling_range'][1],
                min=1.0, max=5.0, step=0.1,
                description='Scale Max:'
            )

            # Window selection (range within client's windows)
            start_window = widgets.IntSlider(
                value=self.client_window_ranges[client_idx][0],
                min=self.client_window_ranges[client_idx][0],
                max=self.client_window_ranges[client_idx][1] - 1,
                description='Start Window:'
            )

            end_window = widgets.IntSlider(
                value=self.client_window_ranges[client_idx][0],
                min=self.client_window_ranges[client_idx][0],
                max=self.client_window_ranges[client_idx][1] - 1,
                description='End Window:'
            )

            # Update end_window constraints when start_window changes
            def update_end_min(change):
                end_window.min = change.new
                if end_window.value < change.new:
                    end_window.value = change.new

            start_window.observe(update_end_min, names='value')

            save_button = widgets.Button(
                description='Save Configuration',
                button_style='success'
            )

            def save_config(_):
                self.client_noise_settings[client_idx] = {
                    'enable': enable_noise.value,
                    'noise_scale': noise_scale.value,
                    'scaling_range': (scaling_min.value, scaling_max.value),
                    'windows': list(range(start_window.value, end_window.value + 1))
                }

                with self.output_area:
                    clear_output(wait=True)
                    print(f"Configuration saved for Client {client_idx + 1}")
                    if enable_noise.value:
                        print(f"Noise will be injected in windows {start_window.value} to {end_window.value}")
                    else:
                        print("Noise injection disabled for this client")

                    # Show all configurations
                    print("\n--- Current Configurations ---")
                    for i in range(3):
                        settings = self.client_noise_settings[i]
                        print(f"Client {i+1}: {'Enabled' if settings['enable'] else 'Disabled'}")
                        if settings['enable']:
                            print(f"  Windows: {settings['windows'][0]} to {settings['windows'][-1]}")
                            print(f"  Noise Scale: {settings['noise_scale']}")
                            print(f"  Scaling Range: {settings['scaling_range']}")

            save_button.on_click(save_config)

            config_ui = widgets.VBox([
                enable_noise,
                noise_scale,
                scaling_min,
                scaling_max,
                widgets.HTML('<b>Window Range:</b>'),
                start_window,
                end_window,
                save_button
            ])

            display(config_ui)

    def prepare_injection_data(self):
        """Prepare data with noise injection based on UI settings."""
        total_windows = len(self.stream2_windows)

        # Split windows among clients
        client_1_windows_end = total_windows // 3
        client_2_windows_end = 2 * (total_windows // 3)

        self.client_test_windows = [
            self.stream2_windows[:client_1_windows_end],
            self.stream2_windows[client_1_windows_end:client_2_windows_end],
            self.stream2_windows[client_2_windows_end:],
        ]

        # Apply noise according to settings
        for client_idx in range(3):
            if self.client_noise_settings[client_idx]['enable']:
                settings = self.client_noise_settings[client_idx]
                for window_idx in settings['windows']:
                    if window_idx < len(self.stream2_windows):
                        # Convert global window index to client-specific window index
                        local_window_idx = window_idx - self.client_window_ranges[client_idx][0]
                        if 0 <= local_window_idx < len(self.client_test_windows[client_idx]):
                            original_window = self.client_test_windows[client_idx][local_window_idx]
                            noisy_window = inject_distorting_noise(
                                original_window,
                                noise_scale=settings['noise_scale'],
                                scaling_range=settings['scaling_range'],
                                exclude_columns=['label']
                            )
                            self.client_test_windows[client_idx][local_window_idx] = noisy_window

    def plot_to_html(self, fig):
        """
        Convert matplotlib figure to HTML image.
        
        Args:
            fig (Figure): Matplotlib figure
            
        Returns:
            str: HTML img tag with base64-encoded figure
        """
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        img_buffer.seek(0)
        img_data = base64.b64encode(img_buffer.read()).decode()
        return f'<img src="data:image/png;base64,{img_data}" />'

    def run_simulation_without_noise(self, _):
        """Run the simulation without any noise."""
        # Reset all noise settings to disabled
        for client_idx in range(3):
            self.client_noise_settings[client_idx]['enable'] = False

        # Call the main simulation function
        self.run_simulation(None)

    def run_simulation(self, _):
        """Run the simulation with selected parameters."""
        with self.output_area:
            clear_output(wait=True)

            # Create persistent configuration display
            config_text = "Noise Injection Configuration:\n"
            has_any_noise = False
            for client_idx in range(3):
                settings = self.client_noise_settings[client_idx]
                config_text += f"Client {client_idx + 1}: {'Enabled' if settings['enable'] else 'Disabled'}\n"
                if settings['enable']:
                    has_any_noise = True
                    config_text += f"  Windows: {settings['windows'][0]} to {settings['windows'][-1]}\n"
                    config_text += f"  Noise Scale: {settings['noise_scale']}\n"
                    config_text += f"  Scaling Range: {settings['scaling_range']}\n"

            if not has_any_noise:
                config_text += "\nNo noise configured - Running with clean data for all clients\n"

            # Display initial configuration
            print(config_text)
            print("-" * 50)

            self.prepare_injection_data()

            # Initialize models
            client_models = [SimpleModel(self.input_dim) for _ in range(3)]
            for client_model in client_models:
                client_model.load_state_dict(copy.deepcopy(self.global_model.state_dict()))

            reference_models = [SimpleModel(self.input_dim) for _ in range(3)]
            for i in range(3):
                reference_models[i].load_state_dict(copy.deepcopy(client_models[i].state_dict()))

            prev_global_model = SimpleModel(self.input_dim)
            prev_global_model.load_state_dict(copy.deepcopy(self.global_model.state_dict()))

            val_for_drift = self.stream1_data.sample(min(1000, len(self.stream1_data)), random_state=42)

            # Initialize tracking
            client_accuracies = [[] for _ in range(3)]
            client_func_diffs = [[] for _ in range(3)]
            client_disagreements = [[] for _ in range(3)]
            client_ks_avg_p = [[] for _ in range(3)]
            client_ks_min_p = [[] for _ in range(3)]
            client_pred_ks_stat = [[] for _ in range(3)]
            client_pred_ks_p = [[] for _ in range(3)]
            client_pred_psi = [[] for _ in range(3)]  # PSI tracking
            global_func_diffs = []
            global_disagreements = []

            # Track global model performance
            global_model_accuracy_all = []  # Global model with all clients
            global_model_accuracy_clean = []  # Global model excluding noisy clients

            # Track which clients have noise injection
            client_has_noise = [settings['enable'] for settings in self.client_noise_settings.values()]

            # Create figure with smaller plots (4 columns)
            fig = plt.figure(figsize=(16, 15))
            axes = []
            for i in range(3):
                axes.append([
                    fig.add_subplot(4, 4, i*4 + 1),
                    fig.add_subplot(4, 4, i*4 + 2),
                    fig.add_subplot(4, 4, i*4 + 3),
                    fig.add_subplot(4, 4, i*4 + 4)  # Column for PSI
                ])

            # Add global performance plot
            global_ax = fig.add_subplot(4, 1, 4)  # Bottom row, full width

            # Run simulation round by round
            num_rounds = min(len(self.client_test_windows[0]), len(self.client_test_windows[1]), len(self.client_test_windows[2]))

            for round_idx in range(num_rounds):
                for client_idx in range(3):
                    if round_idx < len(self.client_test_windows[client_idx]):
                        window_df = self.client_test_windows[client_idx][round_idx]

                        # Evaluate and collect metrics
                        acc = evaluate_model(reference_models[client_idx], window_df)
                        client_accuracies[client_idx].append(acc)

                        avg_p_value, min_p_value = ks_test_drift(self.stream1_data, window_df)
                        client_ks_avg_p[client_idx].append(avg_p_value)
                        client_ks_min_p[client_idx].append(min_p_value)

                        pred_ks_stat, pred_ks_p = ks_test_predictions(reference_models[client_idx], self.stream1_data, window_df)
                        client_pred_ks_stat[client_idx].append(pred_ks_stat)
                        client_pred_ks_p[client_idx].append(pred_ks_p)

                        # Calculate PSI
                        pred_psi = calculate_psi_predictions(reference_models[client_idx], self.stream1_data, window_df)
                        client_pred_psi[client_idx].append(pred_psi)

                        train_model(client_models[client_idx], window_df, max_epochs=30, patience=5, min_delta=0.001)

                        func_diff, disagree_rate = functional_drift_score(
                            prev_global_model, client_models[client_idx], val_for_drift
                        )
                        client_func_diffs[client_idx].append(func_diff)
                        client_disagreements[client_idx].append(disagree_rate)

                # Aggregate models
                global_state = aggregate_models(client_models)
                new_global_model = SimpleModel(self.input_dim)
                new_global_model.load_state_dict(global_state)

                # Aggregate models excluding noisy clients
                clean_client_models = [model for idx, model in enumerate(client_models) if not client_has_noise[idx]]
                if clean_client_models:
                    clean_global_state = aggregate_models(clean_client_models)
                    clean_global_model = SimpleModel(self.input_dim)
                    clean_global_model.load_state_dict(clean_global_state)
                else:
                    # If all clients have noise, use the full global model
                    clean_global_model = new_global_model

                # Evaluate global models
                global_acc_all = evaluate_model(new_global_model, val_for_drift)
                global_model_accuracy_all.append(global_acc_all)

                global_acc_clean = evaluate_model(clean_global_model, val_for_drift)
                global_model_accuracy_clean.append(global_acc_clean)

                global_func_diff, global_disagree_rate = functional_drift_score(
                    prev_global_model, new_global_model, val_for_drift
                )
                global_func_diffs.append(global_func_diff)
                global_disagreements.append(global_disagree_rate)

                # Update plots
                for i in range(3):
                    rounds_so_far = list(range(len(client_accuracies[i])))

                    # Accuracy plot
                    axes[i][0].clear()
                    axes[i][0].plot(rounds_so_far, client_accuracies[i], marker='o', linewidth=1.5, markersize=4)
                    axes[i][0].set_title(f"Client {i+1} - Accuracy", fontsize=8, pad=3)
                    axes[i][0].set_xlabel("Round", fontsize=6)
                    axes[i][0].set_ylabel("Accuracy", fontsize=6)
                    axes[i][0].grid(True, alpha=0.3)
                    axes[i][0].tick_params(axis='both', which='major', labelsize=6)

                    # KS p-value plot (with log scale)
                    axes[i][1].clear()
                    axes[i][1].plot(rounds_so_far, client_ks_avg_p[i], marker='o', label='Avg p-value', linewidth=1.5, markersize=4)
                    axes[i][1].plot(rounds_so_far, client_ks_min_p[i], marker='s', label='Min p-value', linewidth=1.5, markersize=4)
                    axes[i][1].axhline(y=0.05, color='r', linestyle='--', label='p=0.05', linewidth=1)
                    axes[i][1].set_title(f"Client {i+1} - KS p-values", fontsize=8, pad=3)
                    axes[i][1].set_xlabel("Round", fontsize=6)
                    axes[i][1].set_ylabel("p-value (log scale)", fontsize=6)
                    axes[i][1].set_yscale('log')

                    # Automatically adjust y-axis limits based on data
                    all_p_values = client_ks_avg_p[i] + client_ks_min_p[i]
                    if all_p_values:  # Check if list is not empty
                        min_p = min(filter(lambda x: x > 0, all_p_values)) if any(x > 0 for x in all_p_values) else 1e-50
                        max_p = max(all_p_values)
                        # Add some padding for better visualization
                        axes[i][1].set_ylim(bottom=min_p/10, top=max_p*10)
                    else:
                        axes[i][1].set_ylim(bottom=1e-50, top=1e2)  # Default limits

                    axes[i][1].legend(fontsize=5, loc='upper right')
                    axes[i][1].grid(True, alpha=0.3, which='both')
                    axes[i][1].tick_params(axis='both', which='major', labelsize=6)

                    # Prediction KS plot (with log scale)
                    axes[i][2].clear()
                    axes[i][2].plot(rounds_so_far, client_pred_ks_p[i], marker='s', color='orange', linewidth=1.5, markersize=4)
                    axes[i][2].axhline(y=0.05, color='r', linestyle='--', label='p=0.05', linewidth=1)
                    axes[i][2].set_title(f"Client {i+1} - Pred. KS p-value", fontsize=8, pad=3)
                    axes[i][2].set_xlabel("Round", fontsize=6)
                    axes[i][2].set_ylabel("p-value (log scale)", fontsize=6)
                    axes[i][2].set_yscale('log')

                    # Automatically adjust y-axis limits based on data
                    if client_pred_ks_p[i]:  # Check if list is not empty
                        pred_p_values = client_pred_ks_p[i]
                        min_pred_p = min(filter(lambda x: x > 0, pred_p_values)) if any(x > 0 for x in pred_p_values) else 1e-50
                        max_pred_p = max(pred_p_values)
                        # Add some padding for better visualization
                        axes[i][2].set_ylim(bottom=min_pred_p/10, top=max_pred_p*10)
                    else:
                        axes[i][2].set_ylim(bottom=1e-50, top=1e2)  # Default limits

                    axes[i][2].legend(fontsize=5, loc='upper right')
                    axes[i][2].grid(True, alpha=0.3, which='both')
                    axes[i][2].tick_params(axis='both', which='major', labelsize=6)

                    # PSI plot
                    axes[i][3].clear()
                    axes[i][3].plot(rounds_so_far, client_pred_psi[i], marker='D', color='purple', linewidth=1.5, markersize=4)
                    axes[i][3].axhline(y=0.1, color='y', linestyle='--', label='PSI=0.1 (warning)', linewidth=1)
                    axes[i][3].axhline(y=0.2, color='r', linestyle='--', label='PSI=0.2 (critical)', linewidth=1)
                    axes[i][3].set_title(f"Client {i+1} - Pred. PSI", fontsize=8, pad=3)
                    axes[i][3].set_xlabel("Round", fontsize=6)
                    axes[i][3].set_ylabel("PSI", fontsize=6)
                    axes[i][3].legend(fontsize=5, loc='upper right')
                    axes[i][3].grid(True, alpha=0.3)
                    axes[i][3].tick_params(axis='both', which='major', labelsize=6)

                    # Highlight affected windows (only for noisy clients)
                    if client_has_noise[i]:
                        noise_settings = self.client_noise_settings[i]
                        for window_idx in noise_settings['windows']:
                            # Convert global window index to client-specific round index
                            client_round_idx = window_idx - self.client_window_ranges[i][0]
                            if 0 <= client_round_idx < len(rounds_so_far):
                                for ax in axes[i]:
                                    ax.axvspan(client_round_idx - 0.5, client_round_idx + 0.5,
                                             alpha=0.2, color='red')

                # Plot global model performance (both models on same chart)
                rounds_so_far = list(range(len(global_model_accuracy_all)))

                # Combined global model plot
                global_ax.clear()
                global_ax.plot(rounds_so_far, global_model_accuracy_all, marker='o', color='blue',
                              linewidth=1.5, markersize=4, label='All Clients')
                global_ax.plot(rounds_so_far, global_model_accuracy_clean, marker='s', color='green',
                              linewidth=1.5, markersize=4, label='Clean Clients Only')
                global_ax.set_title("Global Model Performance", fontsize=10, pad=5)
                global_ax.set_xlabel("Round", fontsize=8)
                global_ax.set_ylabel("Accuracy", fontsize=8)
                global_ax.legend(loc='best', fontsize=8)
                global_ax.grid(True, alpha=0.3)
                global_ax.tick_params(axis='both', which='major', labelsize=7)

                # Add noise indication
                noisy_clients = [i+1 for i, has_noise in enumerate(client_has_noise) if has_noise]
                if noisy_clients:
                    noise_text = f"Noise on Client(s): {', '.join(map(str, noisy_clients))}"
                    global_ax.text(0.02, 0.02, noise_text, transform=global_ax.transAxes,
                                  fontsize=8, color='red', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

                fig.tight_layout(pad=1.0)

                # Create output containers for persistent display
                if round_idx == 0:
                    self.config_display = widgets.HTML(value=f"<pre>{config_text}</pre>")
                    self.round_display = widgets.HTML(value=f"<b>Running Round 1/{num_rounds}</b>")
                    self.plot_display = widgets.Output()
                    display(widgets.VBox([self.config_display, self.round_display, self.plot_display]))
                else:
                    self.round_display.value = f"<b>Running Round {round_idx + 1}/{num_rounds}</b>"

                # Update plot only
                with self.plot_display:
                    clear_output(wait=True)
                    display(HTML(self.plot_to_html(fig)))

                time.sleep(0.01)  # Small delay to make updates visible

                # Update clients with new global model
                for client_idx in range(3):
                    client_models[client_idx].load_state_dict(copy.deepcopy(global_state))

                prev_global_model.load_state_dict(copy.deepcopy(global_state))

            # Display final plot and completion message
            with self.plot_display:
                clear_output(wait=True)
                display(HTML(self.plot_to_html(fig)))

            self.round_display.value = f"<b>Simulation Complete - {num_rounds} rounds</b>"

            # Re-enable the run button for next run
            self.run_button.disabled = False

    def display(self):
        """Display the UI"""
        display(self.ui)
, random_state=42)
    else:
        test_subset = test_data

    X = test_subset.drop('label', axis=1).values.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled)

    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        preds_a = model_a(X_tensor).numpy()
        preds_b = model_b(X_tensor).numpy()

    pred_diff = np.mean(np.abs(preds_a - preds_b))
    binary_a = (preds_a > 0.5).astype(int)
    binary_b = (preds_b > 0.5).astype(int)
    disagree_rate = np.mean(binary_a != binary_b)

    return pred_diff, disagree_rate

def ks_test_drift(reference_data, test_data, num_samples=1000):
    """
    Perform KS test between reference and test data on each feature.
    
    Args:
        reference_data (DataFrame): Reference data
        test_data (DataFrame): Test data to check for drift
        num_samples (int): Maximum number of samples to use
        
    Returns:
        tuple: (average_p_value, minimum_p_value)
    """
    if len(reference_data) > num_samples:
        ref_subset = reference_data.sample(num_samples, random_state=42)
    else:
        ref_subset = reference_data

    if len(test_data) > num_samples:
        test_subset = test_data.sample(num_samples, random_state=42)
    else:
        test_subset = test_data

    # Drop label column if present
    if 'label' in ref_subset.columns:
        ref_features = ref_subset.drop('label', axis=1)
    else:
        ref_features = ref_subset

    if 'label' in test_subset.columns:
        test_features = test_subset.drop('label', axis=1)
    else:
        test_features = test_subset

    ks_p_values = []

    # Perform KS test for each feature
    for col in ref_features.columns:
        ks_stat, p_value = ks_2samp(ref_features[col], test_features[col])
        ks_p_values.append(p_value)

    avg_p_value = np.mean(ks_p_values)
    min_p_value = np.min(ks_p_values)

    return avg_p_value, min_p_value

def ks_test_predictions(model, reference_data, test_data, num_samples=1000):
    """
    Perform KS test on predicted probabilities between reference and test data.
    
    Args:
        model (nn.Module): Model to use for predictions
        reference_data (DataFrame): Reference data
        test_data (DataFrame): Test data to check for drift
        num_samples (int): Maximum number of samples to use
        
    Returns:
        tuple: (ks_statistic, p_value)
    """
    # Sample data if needed
    if len(reference_data) > num_samples:
        ref_subset = reference_data.sample(num_samples, random_state=42)
    else:
        ref_subset = reference_data

    if len(test_data) > num_samples:
        test_subset = test_data.sample(num_samples, random_state=42)
    else:
        test_subset = test_data

    # Get predictions for reference data
    X_ref = ref_subset.drop('label', axis=1).values.astype(np.float32)
    scaler_ref = StandardScaler()
    X_ref_scaled = scaler_ref.fit_transform(X_ref)
    X_ref_tensor = torch.tensor(X_ref_scaled)

    # Get predictions for test data
    X_test = test_subset.drop('label', axis=1).values.astype(np.float32)
    scaler_test = StandardScaler()
    X_test_scaled = scaler_test.fit_transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled)

    model.eval()
    with torch.no_grad():
        preds_ref = model(X_ref_tensor).numpy().flatten()
        preds_test = model(X_test_tensor).numpy().flatten()

    # Perform KS test on predicted probabilities
    ks_stat, p_value = ks_2samp(preds_ref, preds_test)

    return ks_stat, p_value

def calculate_psi(expected, actual, n_bins=10):
    """
    Calculate Population Stability Index between two arrays.
    
    Args:
        expected (array): The reference/expected distribution
        actual (array): The distribution to compare against the reference
        n_bins (int): Number of bins for histogram
        
    Returns:
        float: PSI value (higher values indicate more drift)
    """
    from sklearn.preprocessing import KBinsDiscretizer
    
    # Create bins based on the expected distribution
    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')

    # Reshape to 2D array for KBinsDiscretizer
    expected_reshaped = expected.reshape(-1, 1)
    kbd.fit(expected_reshaped)

    # Get bin counts for both distributions
    expected_counts = np.histogram(expected, bins=kbd.bin_edges_[0])[0]
    actual_counts = np.histogram(actual, bins=kbd.bin_edges_[0])[0]

    # Convert to percentages and add small epsilon to avoid division by zero
    epsilon = 1e-10
    expected_percents = expected_counts / float(sum(expected_counts)) + epsilon
    actual_percents = actual_counts / float(sum(actual_counts)) + epsilon

    # Calculate PSI
    psi_value = sum((actual_percents - expected_percents) * 
                   np.log(actual_percents / expected_percents))

    return psi_value

def calculate_psi_predictions(model, reference_data, test_data, num_samples=1000, n_bins=10):
    """
    Calculate PSI on predicted probabilities between reference and test data.
    
    Args:
        model (nn.Module): Model to use for predictions
        reference_data (DataFrame): Reference data
        test_data (DataFrame): Test data to check for drift
        num_samples (int): Maximum number of samples to use
        n_bins (int): Number of bins for PSI calculation
        
    Returns:
        float: PSI value (higher values indicate more drift)
    """
    # Sample data if needed
    if len(reference_data) > num_samples:
        ref_subset = reference_data.sample(num_samples, random_state=42)
    else:
        ref_subset = reference_data

    if len(test_data) > num_samples:
        test_subset = test_data.sample(num_samples