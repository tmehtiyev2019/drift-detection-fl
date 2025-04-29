import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
import base64
from IPython.display import HTML

def create_drift_summary_plot(drift_scores, psi_scores, accuracies, 
                            classifier_name="Model", window_size=500,
                            ks_threshold=0.01, psi_threshold=0.2):
    """
    Create a comprehensive drift summary plot with KS scores, PSI values, and accuracy.
    
    Args:
        drift_scores (list): KS test p-values for each window
        psi_scores (list): PSI values for each window
        accuracies (list): Accuracy values for each window
        classifier_name (str): Name of the classifier
        window_size (int): Size of windows used
        ks_threshold (float): Threshold for KS test p-value
        psi_threshold (float): Threshold for PSI values
        
    Returns:
        Figure: Matplotlib figure object
    """
    fig = Figure(figsize=(15, 12))
    
    # Calculate x-axis points for windows
    num_windows = len(drift_scores)
    x_points = np.arange(num_windows)
    
    # Plot 1: KS Drift Scores
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(x_points, drift_scores, label=f'{classifier_name} KS Score',
            color='blue', linewidth=2)
    ax1.axhline(y=ks_threshold, color='r', linestyle='--', label='KS Threshold', linewidth=2)
    ax1.set_title(f'KS Test Drift Scores - {classifier_name} (Window Size: {window_size})', 
                fontsize=18, pad=20)
    ax1.set_xlabel('Window Index', fontsize=16, labelpad=10)
    ax1.set_ylabel('KS Score (p-value)', fontsize=16, labelpad=10)
    ax1.set_yscale('log')
    ax1.legend(loc='upper right', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Plot 2: PSI Scores
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(x_points, psi_scores, label=f'{classifier_name} PSI Score',
            color='green', linewidth=2)
    ax2.axhline(y=psi_threshold, color='r', linestyle='--', label='PSI Threshold', linewidth=2)
    ax2.set_title(f'PSI Scores - {classifier_name} (Window Size: {window_size})', 
                fontsize=18, pad=20)
    ax2.set_xlabel('Window Index', fontsize=16, labelpad=10)
    ax2.set_ylabel('PSI Value', fontsize=16, labelpad=10)
    ax2.legend(loc='upper right', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    # Plot 3: Accuracy
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(x_points, accuracies, label=f'{classifier_name} Accuracy',
            color='orange', linewidth=2)
    ax3.set_title(f'Model Accuracy - {classifier_name} (Window Size: {window_size})', 
                fontsize=18, pad=20)
    ax3.set_xlabel('Window Index', fontsize=16, labelpad=10)
    ax3.set_ylabel('Accuracy', fontsize=16, labelpad=10)
    ax3.legend(loc='lower right', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    
    fig.tight_layout()
    return fig

def highlight_drift_regions(fig, drift_window_indices, alpha=0.2, color='salmon', 
                          axis_index=None):
    """
    Highlight regions in a plot where drift was detected.
    
    Args:
        fig (Figure): Matplotlib figure object
        drift_window_indices (list): Indices of windows where drift was detected
        alpha (float): Transparency of highlighted regions
        color (str): Color of highlighted regions
        axis_index (int, optional): Index of axis to highlight (if None, highlight all)
        
    Returns:
        Figure: Updated Matplotlib figure object
    """
    axes = fig.get_axes()
    target_axes = [axes[axis_index]] if axis_index is not None else axes
    
    for ax in target_axes:
        for idx in drift_window_indices:
            ax.axvspan(idx - 0.5, idx + 0.5, alpha=alpha, color=color)
            
    return fig

def add_region_annotations(fig, region_ranges, labels, colors=None, alpha=0.2):
    """
    Add annotated regions to a drift detection plot.
    
    Args:
        fig (Figure): Matplotlib figure object
        region_ranges (list): List of (start, end) tuples for each region
        labels (list): Labels for each region
        colors (list, optional): Colors for each region
        alpha (float): Transparency of regions
        
    Returns:
        Figure: Updated Matplotlib figure object
    """
    if colors is None:
        colors = ['lightblue', 'salmon', 'lightgreen']
    
    axes = fig.get_axes()
    
    for ax in axes:
        for i, ((start, end), label) in enumerate(zip(region_ranges, labels)):
            color = colors[i % len(colors)]
            ax.axvspan(start, end, alpha=alpha, color=color, label=label)
        
        # Only add the legend to the first axis to avoid duplication
        if ax == axes[0]:
            handles, existing_labels = ax.get_legend_handles_labels()
            # Find the indices of the region labels that are not already in the legend
            new_indices = [i + len(existing_labels) - len(labels) for i, l in enumerate(labels) 
                         if l not in existing_labels]
            if new_indices:
                ax.legend(handles + [plt.Rectangle((0, 0), 1, 1, fc=colors[i % len(colors)], alpha=alpha) 
                                 for i in range(len(labels))],
                        existing_labels + labels, loc='upper right', fontsize=14)
    
    return fig

def plot_to_html(fig):
    """
    Convert a Matplotlib figure to an HTML image tag.
    
    Args:
        fig (Figure): Matplotlib figure object
        
    Returns:
        str: HTML img tag with base64-encoded figure
    """
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.read()).decode()
    return f'<img src="data:image/png;base64,{img_data}" />'

def display_figure_as_html(fig):
    """
    Display a Matplotlib figure as HTML in a Jupyter notebook.
    
    Args:
        fig (Figure): Matplotlib figure object
    """
    return HTML(plot_to_html(fig))

def plot_prediction_distributions(model_a_preds, model_b_preds, title="Prediction Distributions",
                                model_a_name="Model A", model_b_name="Model B"):
    """
    Plot histogram and KDE of prediction distributions from two models.
    
    Args:
        model_a_preds (array): Predictions from first model
        model_b_preds (array): Predictions from second model
        title (str): Plot title
        model_a_name (str): Name of first model
        model_b_name (str): Name of second model
        
    Returns:
        Figure: Matplotlib figure with distribution plots
    """
    from scipy.stats import gaussian_kde
    
    fig = Figure(figsize=(12, 8))
    
    # Plot histogram
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.hist(model_a_preds, bins=20, alpha=0.7, label=model_a_name, color='blue', density=True)
    ax1.hist(model_b_preds, bins=20, alpha=0.7, label=model_b_name, color='orange', density=True)
    ax1.set_title(f"{title} - Histogram", fontsize=14)
    ax1.set_xlabel("Prediction Value", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)
    
    # Plot KDE
    ax2 = fig.add_subplot(2, 1, 2)
    x_range = np.linspace(0, 1, 1000)
    
    if len(model_a_preds) > 1:
        kde1 = gaussian_kde(model_a_preds)
        ax2.plot(x_range, kde1(x_range), color='blue', linewidth=2, label=model_a_name)
    
    if len(model_b_preds) > 1:
        kde2 = gaussian_kde(model_b_preds)
        ax2.plot(x_range, kde2(x_range), color='orange', linewidth=2, label=model_b_name)
    
    ax2.set_title(f"{title} - Probability Density Function", fontsize=14)
    ax2.set_xlabel("Prediction Value", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(alpha=0.3)
    
    fig.tight_layout()
    return fig

def create_drift_detection_summary(results_dict, metrics=['ks_p_value', 'psi_value', 'accuracy']):
    """
    Create a summary table of drift detection results.
    
    Args:
        results_dict (dict): Results dictionary from drift detection
        metrics (list): List of metrics to include in summary
        
    Returns:
        str: HTML table with drift detection summary
    """
    html = """
    <table style="width:100%; border-collapse: collapse; text-align: center;">
        <tr style="background-color: #f2f2f2;">
            <th style="padding: 8px; border: 1px solid #ddd;">Classifier</th>
            <th style="padding: 8px; border: 1px solid #ddd;">Drift Detected Windows</th>
    """
    
    # Add metric columns
    for metric in metrics:
        metric_name = metric.replace('_', ' ').title()
        html += f'<th style="padding: 8px; border: 1px solid #ddd;">{metric_name} (Avg)</th>\n'
    
    html += '</tr>\n'
    
    # Add data rows
    for clf_name, results in results_dict.items():
        if not results.get('drift_scores', []):
            continue
            
        # Count drift detected windows
        drift_windows = []
        for i, result in enumerate(results.get('drift_results', [])):
            if result.get('is_ks_drift', False) or result.get('is_psi_drift', False):
                drift_windows.append(i)
                
        drift_count = len(drift_windows)
        total_windows = len(results.get('drift_scores', []))
        
        html += f"""
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;">{clf_name}</td>
            <td style="padding: 8px; border: 1px solid #ddd;">{drift_count}/{total_windows} ({drift_count/total_windows:.1%})</td>
        """
        
        # Add metric values
        for metric in metrics:
            if metric in results and results[metric]:
                avg_value = sum(results[metric]) / len(results[metric])
                
                # Format based on metric type
                if metric == 'ks_p_value':
                    html += f'<td style="padding: 8px; border: 1px solid #ddd;">{avg_value:.4f}</td>\n'
                elif metric == 'psi_value':
                    html += f'<td style="padding: 8px; border: 1px solid #ddd;">{avg_value:.2f}</td>\n'
                elif metric == 'accuracy':
                    html += f'<td style="padding: 8px; border: 1px solid #ddd;">{avg_value:.2%}</td>\n'
                else:
                    html += f'<td style="padding: 8px; border: 1px solid #ddd;">{avg_value:.4f}</td>\n'
            else:
                html += '<td style="padding: 8px; border: 1px solid #ddd;">N/A</td>\n'
                
        html += '</tr>\n'
    
    html += '</table>'
    return HTML(html)

def create_feature_drift_heatmap(reference_data, test_data, method='ks', threshold=0.05):
    """
    Create a heatmap showing drift in individual features.
    
    Args:
        reference_data (DataFrame): Reference data
        test_data (DataFrame): Test data
        method (str): Drift detection method ('ks' or 'psi')
        threshold (float): Threshold for drift detection
        
    Returns:
        Figure: Matplotlib figure with feature drift heatmap
    """
    from scipy.stats import ks_2samp
    import pandas as pd
    
    # Ensure we're working with DataFrames
    if not isinstance(reference_data, pd.DataFrame):
        raise ValueError("reference_data must be a pandas DataFrame")
    if not isinstance(test_data, pd.DataFrame):
        raise ValueError("test_data must be a pandas DataFrame")
    
    # Drop label column if present
    if 'label' in reference_data.columns:
        reference_data = reference_data.drop('label', axis=1)
    if 'label' in test_data.columns:
        test_data = test_data.drop('label', axis=1)
    
    # Ensure both datasets have the same columns
    common_cols = list(set(reference_data.columns).intersection(set(test_data.columns)))
    reference_data = reference_data[common_cols]
    test_data = test_data[common_cols]
    
    # Calculate drift for each feature
    drift_values = []
    
    for col in common_cols:
        if method == 'ks':
            _, p_value = ks_2samp(reference_data[col], test_data[col])
            drift_values.append(p_value)
        elif method == 'psi':
            from .drift_monitor import DriftMonitor
            monitor = DriftMonitor()
            psi = monitor._calculate_psi(
                reference_data[col].values.reshape(-1, 1), 
                test_data[col].values.reshape(-1, 1)
            )
            drift_values.append(psi)
        else:
            raise ValueError("method must be 'ks' or 'psi'")
    
    # Create a DataFrame for the heatmap
    drift_df = pd.DataFrame({
        'Feature': common_cols,
        'Drift Value': drift_values
    })
    
    # Sort by drift value
    if method == 'ks':
        drift_df = drift_df.sort_values('Drift Value')  # Lower p-value indicates more drift
    else:  # psi
        drift_df = drift_df.sort_values('Drift Value', ascending=False)  # Higher PSI indicates more drift
    
    # Create the plot
    fig = Figure(figsize=(12, len(common_cols) * 0.4))
    ax = fig.add_subplot(111)
    
    # Determine colors based on drift values
    if method == 'ks':
        colors = ['red' if val < threshold else 'green' for val in drift_df['Drift Value']]
        cmap = 'RdYlGn'  # Red (drift) to Green (no drift)
    else:  # psi
        colors = ['red' if val > threshold else 'green' for val in drift_df['Drift Value']]
        cmap = 'RdYlGn_r'  # Red (drift) to Green (no drift)
    
    # Plot horizontal bars
    bars = ax.barh(drift_df['Feature'], drift_df['Drift Value'], color=colors)
    
    # Add drift threshold line
    ax.axvline(x=threshold, color='black', linestyle='--', 
              label=f"Threshold: {threshold}")
    
    # Add value labels to the bars
    for i, bar in enumerate(bars):
        if method == 'ks':
            value_text = f"{drift_df['Drift Value'].iloc[i]:.3f}"
        else:
            value_text = f"{drift_df['Drift Value'].iloc[i]:.2f}"
            
        # Position the text based on bar width
        if bar.get_width() < threshold / 5:
            ax.text(bar.get_width() + threshold/20, bar.get_y() + bar.get_height()/2, 
                  value_text, ha='left', va='center', fontsize=10)
        else:
            ax.text(bar.get_width() - threshold/20, bar.get_y() + bar.get_height()/2, 
                  value_text, ha='right', va='center', fontsize=10, color='white')
    
    # Add titles and labels
    if method == 'ks':
        ax.set_title(f"Feature Drift Analysis (KS Test p-values)", fontsize=14)
        ax.set_xlabel("p-value (lower = more drift)", fontsize=12)
    else:
        ax.set_title(f"Feature Drift Analysis (PSI Values)", fontsize=14)
        ax.set_xlabel("PSI Value (higher = more drift)", fontsize=12)
        
    ax.set_ylabel("Feature", fontsize=12)
    ax.legend()
    
    # Adjust layout
    fig.tight_layout()
    return fig

def create_drift_monitoring_dashboard(results_dict, window_size=None, front_size=None, original_size=None):
    """
    Create an HTML dashboard summarizing drift detection results.
    
    Args:
        results_dict (dict): Results dictionary from drift detection
        window_size (int, optional): Window size used for detection
        front_size (int, optional): Size of front segment in test data
        original_size (int, optional): Size of original test segment
        
    Returns:
        str: HTML dashboard with drift detection summary and visualizations
    """
    # Import necessary modules
    import pandas as pd
    from IPython.display import HTML
    
    # Create HTML dashboard
    html = """
    <div style="font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto;">
        <h1 style="text-align: center; color: #333;">Drift Monitoring Dashboard</h1>
    """
    
    # Add window size information if provided
    if window_size:
        html += f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <p><strong>Window Size:</strong> {window_size}</p>
        """
        
        if front_size is not None and original_size is not None:
            html += f"""
            <p><strong>Data Segments:</strong> Front Clean ({front_size} samples), 
            Original Test ({original_size} samples), End Clean ({front_size} samples)</p>
            """
            
        html += "</div>"
    
    # Create summary table
    html += """
    <h2>Drift Detection Summary</h2>
    <div style="margin-bottom: 20px;">
    """
    
    summary_table = create_drift_detection_summary(results_dict)
    html += summary_table._repr_html_()
    html += "</div>"
    
    # Add visualizations for each classifier
    html += "<h2>Classifier Visualizations</h2>"
    
    for clf_name, results in results_dict.items():
        if not results.get('drift_scores', []):
            continue
            
        html += f"""
        <div style="margin-bottom: 30px;">
            <h3>{clf_name}</h3>
            <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;">
        """
        
        # Create and add plots
        fig = create_drift_summary_plot(
            results['drift_scores'], 
            results['psi_scores'], 
            results['accuracies'],
            classifier_name=clf_name,
            window_size=window_size
        )
        
        # Add segment highlighting if provided
        if front_size is not None and original_size is not None and window_size is not None:
            front_end_idx = front_size // window_size
            original_end_idx = (front_size + original_size) // window_size
            
            region_ranges = [(0, front_end_idx), 
                           (front_end_idx, original_end_idx), 
                           (original_end_idx, len(results['drift_scores']))]
            labels = ['Known Distribution', 'Test Distribution', 'Known Distribution']
            
            fig = add_region_annotations(fig, region_ranges, labels)
        
        html += f'<div>{plot_to_html(fig)}</div>'
        html += """
            </div>
        </div>
        """
    
    html += """
    </div>
    """
    
    return HTML(html)
array): Predictions from first model
        model_b_preds (