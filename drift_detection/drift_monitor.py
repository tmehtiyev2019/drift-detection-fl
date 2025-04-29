import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier

class DriftMonitor:
    """
    A class for monitoring data drift using various classifiers and metrics.
    
    This class implements multiple drift detection methods:
    - Kolmogorov-Smirnov (KS) test on model predictions
    - Population Stability Index (PSI) on model predictions
    - Feature-level drift detection using both KS and PSI
    
    It supports multiple classifiers for drift detection:
    - XGBoost
    - Random Forest
    - Gradient Boosting
    - Logistic Regression
    - Decision Tree
    """
    
    def __init__(self, window_size=500, threshold=0.05, psi_threshold=0.2, n_bins=10):
        """
        Initialize the DriftMonitor with parameters.
        
        Args:
            window_size (int): Size of data windows to use for drift detection
            threshold (float): p-value threshold for KS test (lower values indicate drift)
            psi_threshold (float): Threshold for PSI (higher values indicate drift)
            n_bins (int): Number of bins to use for PSI calculation
        """
        self.window_size = window_size
        self.threshold = threshold
        self.psi_threshold = psi_threshold
        self.n_bins = n_bins
        self.classifiers = {
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42)
        }
        # Store results by classifier
        self.results_by_classifier = {name: {} for name in self.classifiers.keys()}

    def _ensure_numpy(self, X):
        """Convert input to numpy array if it's a DataFrame"""
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.array(X)

    def _calculate_psi(self, expected, actual):
        """
        Calculate Population Stability Index between two arrays
        
        Args:
            expected (array): The reference/expected distribution
            actual (array): The distribution to compare against the reference
            
        Returns:
            float: PSI value (higher values indicate more drift)
        """
        # Create bins based on the expected distribution
        kbd = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='quantile')
        kbd.fit(expected.reshape(-1, 1))

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

    def detect_drift(self, X_train, y_train, X_test, y_test):
        """
        Main drift detection function using multiple classifiers
        
        Args:
            X_train (array-like): Training features
            y_train (array-like): Training labels
            X_test (array-like): Test features to check for drift
            y_test (array-like): Test labels
            
        Returns:
            dict: Results organized by classifier, containing drift scores, PSI values, and accuracies
        """
        X_train = self._ensure_numpy(X_train)
        X_test = self._ensure_numpy(X_test)
        y_train = self._ensure_numpy(y_train)
        y_test = self._ensure_numpy(y_test)

        # Define reference window using stratified sampling
        stratified_size = min(self.window_size, len(X_train))
        window1, _, y_ref, _ = train_test_split(
            X_train,
            y_train,
            train_size=stratified_size/len(X_train),
            stratify=y_train,
            random_state=42
        )

        # Train each classifier and run drift detection
        for name, classifier in self.classifiers.items():
            print(f"Training {name} classifier and detecting drift...")

            # Initialize classifier and fit
            classifier.fit(X_train, y_train)

            # Initialize results storage for this classifier
            self.results_by_classifier[name] = {
                'drift_scores': [],
                'psi_scores': [],
                'accuracies': [],
                'drift_results': []
            }

            # Process windows
            for i in range(0, len(X_test), self.window_size):
                # Ensure we have enough data for a full window
                if i + self.window_size <= len(X_test):

                    window2 = X_test[i:i + self.window_size]
                    y_window = y_test[i:i + self.window_size]

                    result = self._analyze_window(
                        window1, window2,
                        y_window,
                        classifier
                    )

                    self.results_by_classifier[name]['drift_results'].append(result)
                    self.results_by_classifier[name]['drift_scores'].append(result['ks_p_value'])
                    self.results_by_classifier[name]['psi_scores'].append(result['psi_value'])
                    self.results_by_classifier[name]['accuracies'].append(result['accuracy'])
                    # Update reference window for next comparison (sliding window approach)
                    # window1 = current_window

        return self.results_by_classifier

    def _analyze_window(self, window1, window2, y_true, classifier):
        """
        Analyze drift in a single window using both KS test and PSI
        
        Args:
            window1 (array): Reference window features
            window2 (array): Current window features to check for drift
            y_true (array): True labels for current window
            classifier: Trained classifier to use for predictions
            
        Returns:
            dict: Drift analysis results including KS test p-value, PSI value, and accuracy
        """
        # Get predictions
        pred1 = self._get_probabilities(classifier, window1)
        pred2 = self._get_probabilities(classifier, window2)

        # KS test
        _, ks_p_value = ks_2samp(pred1, pred2)
        is_ks_drift = ks_p_value < self.threshold

        # PSI calculation
        psi_value = self._calculate_psi(pred1, pred2)
        is_psi_drift = psi_value > self.psi_threshold

        # Feature-level drift detection
        feature_drifts_ks = []
        feature_drifts_psi = []
        for j in range(window1.shape[1]):
            # KS test for each feature
            _, feat_ks_p_value = ks_2samp(window1[:, j], window2[:, j])
            feature_drifts_ks.append(feat_ks_p_value < self.threshold)

            # PSI for each feature
            feat_psi = self._calculate_psi(window1[:, j], window2[:, j])
            feature_drifts_psi.append(feat_psi > self.psi_threshold)

        is_feature_drift_ks = np.mean(feature_drifts_ks) > 0.3
        is_feature_drift_psi = np.mean(feature_drifts_psi) > 0.3

        # Calculate accuracy
        y_pred = classifier.predict(window2)
        accuracy = accuracy_score(y_true, y_pred)

        return {
            'is_ks_drift': any([is_ks_drift, is_feature_drift_ks]),
            'is_psi_drift': any([is_psi_drift, is_feature_drift_psi]),
            'ks_p_value': ks_p_value,
            'psi_value': psi_value,
            'accuracy': accuracy
        }

    def _get_probabilities(self, classifier, X):
        """
        Get probability predictions with appropriate handling for different classifiers
        
        Args:
            classifier: The classifier to use for predictions
            X (array): Feature matrix to get predictions for
            
        Returns:
            array: Predicted probabilities
        """
        try:
            # For classifiers with predict_proba method returning multiple classes
            probs = classifier.predict_proba(X)
            # Return probability for positive class (index 1)
            if probs.shape[1] > 1:
                return probs[:, 1]
            return probs.ravel()
        except (AttributeError, IndexError):
            # Fallback for models without predict_proba
            return classifier.decision_function(X)


def run_drift_detection_comparison(X_train, y_train, X_test, y_test, window_sizes=[100, 300, 500, 1000],
                                  front_size=None, original_size=None):
    """
    Run drift detection with multiple window sizes and compare multiple classifiers
    
    Args:
        X_train (array-like): Training features
        y_train (array-like): Training labels
        X_test (array-like): Test features to check for drift
        y_test (array-like): Test labels
        window_sizes (list): List of window sizes to try
        front_size (int, optional): Size of front section in test data
        original_size (int, optional): Size of original test set
        
    Returns:
        dict: Results organized by window size and classifier
    """
    results_by_window = {}

    # Set global font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    for window_size in window_sizes:
        print(f"\nRunning with window size: {window_size}")

        # Initialize drift monitor with current window size
        monitor = DriftMonitor(window_size=window_size, threshold=0.01, psi_threshold=0.2)

        # Run drift detection with all classifiers
        results = monitor.detect_drift(X_train, y_train, X_test, y_test)
        results_by_window[window_size] = results

        # Plot results for each classifier
        plot_multi_classifier_results(monitor, window_size, front_size, original_size)

        # Also create a comparative view
        plot_classifier_comparison(monitor, window_size, front_size, original_size)

    return results_by_window


def plot_multi_classifier_results(monitor, window_size, front_size=None, original_size=None):
    """
    Plot detailed results for each classifier
    
    Args:
        monitor (DriftMonitor): Drift monitor instance with results
        window_size (int): Window size used for drift detection
        front_size (int, optional): Size of front section in test data
        original_size (int, optional): Size of original test set
    """
    # Set up colors for each classifier
    colors = {
        'XGBoost': 'blue',
        'RandomForest': 'green',
        'GradientBoosting': 'purple',
        'LogisticRegression': 'orange',
        'DecisionTree': 'red'
    }

    for clf_name, results in monitor.results_by_classifier.items():
        if not results['drift_scores']:  # Skip if no results
            continue

        # Create figure for each classifier
        plt.figure(figsize=(15, 12))

        # Calculate x-axis points for windows
        num_windows = len(results['drift_scores'])
        x_points = np.arange(num_windows)

        # Calculate window indices for boundaries if provided
        if front_size is not None and original_size is not None:
            front_end_idx = front_size // window_size
            original_end_idx = (front_size + original_size) // window_size
            has_segments = True
        else:
            has_segments = False

        # Plot 1: KS Drift Scores with segments
        plt.subplot(3, 1, 1)
        if has_segments:
            plt.axvspan(0, front_end_idx, alpha=0.2, color='lightblue', label='Known Distribution')
            plt.axvspan(front_end_idx, original_end_idx, alpha=0.2, color='salmon', label='Drift Distribution')
            plt.axvspan(original_end_idx, num_windows, alpha=0.2, color='lightblue')

        plt.plot(x_points, results['drift_scores'], label=f'{clf_name} KS Score',
                color=colors[clf_name], linewidth=2)
        plt.axhline(y=0.01, color='r', linestyle='--', label='KS Threshold', linewidth=2)
        plt.title(f'KS Test Drift Scores - {clf_name} (Window Size: {window_size})', fontsize=18, pad=20)
        plt.xlabel('Window Index', fontsize=16, labelpad=10)
        plt.ylabel('KS Score (p-value)', fontsize=16, labelpad=10)
        plt.yscale('log')
        plt.legend(loc='upper right', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=14)

        # Plot 2: PSI Scores with segments
        plt.subplot(3, 1, 2)
        if has_segments:
            plt.axvspan(0, front_end_idx, alpha=0.2, color='lightblue')
            plt.axvspan(front_end_idx, original_end_idx, alpha=0.2, color='salmon')
            plt.axvspan(original_end_idx, num_windows, alpha=0.2, color='lightblue')

        plt.plot(x_points, results['psi_scores'], label=f'{clf_name} PSI Score',
                color=colors[clf_name], linewidth=2)
        plt.axhline(y=0.2, color='r', linestyle='--', label='PSI Threshold', linewidth=2)
        plt.title(f'PSI Scores - {clf_name} (Window Size: {window_size})', fontsize=18, pad=20)
        plt.xlabel('Window Index', fontsize=16, labelpad=10)
        plt.ylabel('PSI Value', fontsize=16, labelpad=10)
        plt.legend(loc='upper right', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=14)

        # Plot 3: Accuracy with segments
        plt.subplot(3, 1, 3)
        if has_segments:
            plt.axvspan(0, front_end_idx, alpha=0.2, color='lightblue')
            plt.axvspan(front_end_idx, original_end_idx, alpha=0.2, color='salmon')
            plt.axvspan(original_end_idx, num_windows, alpha=0.2, color='lightblue')

        plt.plot(x_points, results['accuracies'], label=f'{clf_name} Accuracy',
                color=colors[clf_name], linewidth=2)
        plt.title(f'Model Accuracy - {clf_name} (Window Size: {window_size})', fontsize=18, pad=20)
        plt.xlabel('Window Index', fontsize=16, labelpad=10)
        plt.ylabel('Accuracy', fontsize=16, labelpad=10)
        plt.legend(loc='lower right', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout()
        plt.show()


def plot_classifier_comparison(monitor, window_size, front_size=None, original_size=None):
    """
    Plot comparison of all classifiers together
    
    Args:
        monitor (DriftMonitor): Drift monitor instance with results
        window_size (int): Window size used for drift detection
        front_size (int, optional): Size of front section in test data
        original_size (int, optional): Size of original test set
    """
    # Set up colors for each classifier
    colors = {
        'XGBoost': 'blue',
        'RandomForest': 'green',
        'GradientBoosting': 'purple',
        'LogisticRegression': 'orange',
        'DecisionTree': 'red'
    }

    # Check if we have results
    if not any([results['drift_scores'] for name, results in monitor.results_by_classifier.items()]):
        return

    # Create figure for comparison
    plt.figure(figsize=(18, 15))

    # Get any non-empty result set to determine num_windows
    num_windows = 0
    for name, results in monitor.results_by_classifier.items():
        if results['drift_scores']:
            num_windows = len(results['drift_scores'])
            break

    x_points = np.arange(num_windows)

    # Calculate window indices for boundaries if provided
    if front_size is not None and original_size is not None:
        front_end_idx = front_size // window_size
        original_end_idx = (front_size + original_size) // window_size
        has_segments = True
    else:
        has_segments = False

    # Plot 1: KS Drift Scores comparison
    plt.subplot(3, 1, 1)
    if has_segments:
        plt.axvspan(0, front_end_idx, alpha=0.2, color='lightblue', label='Known Distribution')
        plt.axvspan(front_end_idx, original_end_idx, alpha=0.2, color='salmon', label='Drift Distribution')
        plt.axvspan(original_end_idx, num_windows, alpha=0.2, color='lightblue')

    for name, results in monitor.results_by_classifier.items():
        if results['drift_scores']:
            plt.plot(x_points, results['drift_scores'], label=f'{name}',
                    color=colors[name], linewidth=2)

    plt.axhline(y=0.01, color='r', linestyle='--', label='KS Threshold', linewidth=2)
    plt.title(f'KS Test Drift Scores Comparison (Window Size: {window_size})', fontsize=18, pad=20)
    plt.xlabel('Window Index', fontsize=16, labelpad=10)
    plt.ylabel('KS Score (p-value)', fontsize=16, labelpad=10)
    plt.yscale('log')
    plt.legend(loc='upper right', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=14)

    # Plot 2: PSI Scores comparison
    plt.subplot(3, 1, 2)
    if has_segments:
        plt.axvspan(0, front_end_idx, alpha=0.2, color='lightblue')
        plt.axvspan(front_end_idx, original_end_idx, alpha=0.2, color='salmon')
        plt.axvspan(original_end_idx, num_windows, alpha=0.2, color='lightblue')

    for name, results in monitor.results_by_classifier.items():
        if results['psi_scores']:
            plt.plot(x_points, results['psi_scores'], label=f'{name}',
                    color=colors[name], linewidth=2)

    plt.axhline(y=0.2, color='r', linestyle='--', label='PSI Threshold', linewidth=2)
    plt.title(f'PSI Scores Comparison (Window Size: {window_size})', fontsize=18, pad=20)
    plt.xlabel('Window Index', fontsize=16, labelpad=10)
    plt.ylabel('PSI Value', fontsize=16, labelpad=10)
    plt.legend(loc='upper right', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=14)

    # Plot 3: Accuracy comparison
    plt.subplot(3, 1, 3)
    if has_segments:
        plt.axvspan(0, front_end_idx, alpha=0.2, color='lightblue')
        plt.axvspan(front_end_idx, original_end_idx, alpha=0.2, color='salmon')
        plt.axvspan(original_end_idx, num_windows, alpha=0.2, color='lightblue')

    for name, results in monitor.results_by_classifier.items():
        if results['accuracies']:
            plt.plot(x_points, results['accuracies'], label=f'{name}',
                    color=colors[name], linewidth=2)

    plt.title(f'Model Accuracy Comparison (Window Size: {window_size})', fontsize=18, pad=20)
    plt.xlabel('Window Index', fontsize=16, labelpad=10)
    plt.ylabel('Accuracy', fontsize=16, labelpad=10)
    plt.legend(loc='lower right', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.savefig(f'classifier_comparison_window_{window_size}.png', dpi=300, bbox_inches='tight')
    plt.show()