# Data Drift Detection Framework

This repository contains a comprehensive framework for detecting concept drift in machine learning models. It includes multiple approaches to drift detection, visualization tools, and interactive simulation capabilities.

## Overview

Machine learning models deployed in production often face the challenge of data drift - when the distribution of incoming data differs from the training data. This framework provides multiple methods to detect such drift:

- Statistical tests (Kolmogorov-Smirnov test)
- Population Stability Index (PSI)
- Neural network-based drift detection
- Federated learning simulation with drift

## Features

- Multiple classifier-based drift detection (XGBoost, RandomForest, GradientBoosting, etc.)
- Neural network-based drift detection (MLP, LSTM, CNN)
- Interactive drift simulation with noise injection
- Comprehensive visualization of drift metrics
- Window-based drift analysis
- Federated learning simulation with drift

## Repository Structure

```
drift-detection/
├── data/
│   ├── KDDTrain+.txt
│   ├── KDDTest+.txt
│   ├── NSL_KDD_binary_train.csv
│   ├── NSL_KDD_binary_test.csv
├── models/
│   ├── global_model.pth
├── drift_detection/
│   ├── __init__.py
│   ├── drift_monitor.py
│   ├── neural_drift_monitor.py
│   ├── viz_utils.py
│   ├── data_utils.py
│   ├── simulation.py
├── notebooks/
│   ├── drift_detection_demo.ipynb
├── requirements.txt
├── README.md
└── setup.py
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/drift-detection.git
cd drift-detection

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Dependencies

- numpy
- pandas
- matplotlib
- scikit-learn
- scipy
- XGBoost
- PyTorch
- TensorFlow
- River (for online learning)
- ipywidgets (for interactive simulation)

## Usage Examples

### Basic Drift Detection with Multiple Classifiers

```python
from drift_detection.drift_monitor import DriftMonitor

# Initialize drift monitor
monitor = DriftMonitor(window_size=500, threshold=0.05, psi_threshold=0.2)

# Detect drift
results = monitor.detect_drift(X_train, y_train, X_test, y_test)

# Visualize results
from drift_detection.viz_utils import plot_multi_classifier_results
plot_multi_classifier_results(monitor, window_size=500)
```

### Neural Network-based Drift Detection

```python
from drift_detection.neural_drift_monitor import NeuralDriftMonitor

# Initialize neural drift monitor
neural_monitor = NeuralDriftMonitor(window_size=600, threshold=0.01, psi_threshold=0.2)

# Detect drift using neural networks
nn_results = neural_monitor.detect_drift_with_nn(X_train, y_train, X_test, y_test)
```

### Interactive Drift Simulation

```python
from drift_detection.simulation import DriftDetectionSimulator

# Initialize and display simulator
simulator = DriftDetectionSimulator()
simulator.display()
```

## Dataset

This project uses the NSL-KDD dataset, which is an improved version of the original KDD Cup 1999 dataset for network intrusion detection. The framework processes this dataset to create binary classification tasks for drift detection experimentation.

## License

[MIT License](LICENSE)

## Citation

If you use this framework in your research, please cite:

```
@software{DriftDetectionFramework,
  author = {Your Name},
  title = {Data Drift Detection Framework},
  url = {https://github.com/yourusername/drift-detection},
  year = {2025},
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.