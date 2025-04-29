"""
Drift Detection Framework

A framework for detecting concept drift in machine learning models using 
statistical tests, neural networks, and interactive simulations.
"""

from .drift_monitor import DriftMonitor, run_drift_detection_comparison
from .neural_drift_monitor import NeuralDriftMonitor
from .data_utils import (
    load_and_preprocess_nsl_kdd,
    prepare_drift_experiment_data,
    inject_distorting_noise,
    split_into_windows
)

__version__ = '0.1.0'