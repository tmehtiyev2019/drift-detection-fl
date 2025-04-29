import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_nsl_kdd(train_path=None, test_path=None):
    """
    Load and preprocess the NSL-KDD dataset.
    
    Args:
        train_path (str, optional): Path to KDDTrain+.txt file
        test_path (str, optional): Path to KDDTest+.txt file
        
    Returns:
        tuple: (train_df, test_df) preprocessed DataFrames
    """
    # Define column names for NSL-KDD dataset
    col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
        "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
        "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]
    
    # If paths provided, read from them; otherwise try default paths relative to current directory
    if train_path is None:
        train_path = os.path.join('data', 'KDDTrain+.txt')
    if test_path is None:
        test_path = os.path.join('data', 'KDDTest+.txt')
        
    # Read the original training and test sets
    try:
        df1 = pd.read_csv(train_path, header=None, names=col_names)
        df2 = pd.read_csv(test_path, header=None, names=col_names)
        print(f"Successfully loaded datasets from {train_path} and {test_path}")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None
    
    # Drop the difficulty column
    df1.drop(['difficulty'], axis=1, inplace=True)
    df2.drop(['difficulty'], axis=1, inplace=True)
    
    # Binarize labels: "normal" -> 0, all attacks -> 1
    df1['label'] = df1['label'].apply(lambda x: 0 if x == 'normal' else 1)
    df2['label'] = df2['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # Convert categorical features to numerical using Label Encoder
    df1 = encode_categorical_features(df1)
    df2 = encode_categorical_features(df2)
    
    return df1, df2

def encode_categorical_features(df):
    """
    Convert categorical features to numerical using Label Encoder
    
    Args:
        df (DataFrame): DataFrame to encode
        
    Returns:
        DataFrame: Encoded DataFrame
    """
    cat_features = [x for x in df.columns if df[x].dtype == "object"]
    le = LabelEncoder()
    
    for col in cat_features:
        if col in df.columns:
            i = df.columns.get_loc(col)
            df.iloc[:, i] = le.fit_transform(df.iloc[:, i].astype(str))
    
    return df

def prepare_drift_experiment_data(df_train, df_test, validation_split=0.3):
    """
    Prepare data for drift detection experiment.
    
    This function:
    1. Splits training data into train and validation sets
    2. Further splits validation set into front and end parts
    3. Creates a combined test stream with validation parts surrounding the test set
    
    Args:
        df_train (DataFrame): Training DataFrame
        df_test (DataFrame): Test DataFrame
        validation_split (float): Proportion of training data to use for validation
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test, front_size, original_size)
    """
    # Separate features and labels in training data
    X_full_train = df_train.drop(['label'], axis=1)
    y_full_train = df_train['label']

    # Stratified split on training data to maintain label distribution
    X_train, X_val, y_train, y_val = train_test_split(
        X_full_train, y_full_train, test_size=validation_split, stratify=y_full_train, random_state=42
    )

    # Further split the validation set into two equal parts
    X_val_front, X_val_end, y_val_front, y_val_end = train_test_split(
        X_val, y_val, test_size=0.5, stratify=y_val, random_state=42
    )

    # Convert validation splits to DataFrames 
    df_val_front = pd.DataFrame(X_val_front, columns=X_full_train.columns)
    df_val_front['label'] = y_val_front.values

    df_val_end = pd.DataFrame(X_val_end, columns=X_full_train.columns)
    df_val_end['label'] = y_val_end.values

    # Concatenate: front validation subset + original test set + end validation subset
    df_combined_test = pd.concat([df_val_front, df_test, df_val_end], ignore_index=True)

    # Separate the final combined test set into features and labels
    X_test = df_combined_test.drop(['label'], axis=1)
    y_test = df_combined_test['label']
    
    # Calculate sizes for plotting regions
    front_size = len(df_val_front)
    original_size = len(df_test)

    # Print dataset sizes for verification
    print("Dataset sizes:")
    print(f"Original training set: {len(df_train)} samples")
    print(f"Training subset: {len(y_train)} samples")
    print(f"Validation front subset: {len(y_val_front)} samples")
    print(f"Validation end subset: {len(y_val_end)} samples")
    print(f"Original test set: {len(df_test)} samples")
    print(f"Combined test set: {len(y_test)} samples")

    return X_train, y_train, X_test, y_test, front_size, original_size

def inject_distorting_noise(df, noise_scale=1.0, scaling_range=(0.5, 2.0), exclude_columns=['label']):
    """
    Inject heavy Gaussian noise and random scaling to distort numerical columns of the DataFrame.
    
    Args:
        df (DataFrame): Input DataFrame
        noise_scale (float): Scale of noise to add (multiplier of column std dev)
        scaling_range (tuple): (min, max) factors to randomly scale features
        exclude_columns (list): Column names to exclude from noise injection
        
    Returns:
        DataFrame: DataFrame with noise injected
    """
    df_noisy = df.copy()
    numerical_columns = [col for col in df_noisy.columns 
                         if col not in exclude_columns 
                         and df_noisy[col].dtype in ['float64', 'float32', 'int64', 'int32']]

    for col in numerical_columns:
        col_std = df_noisy[col].std()
        noise = np.random.normal(loc=0, scale=noise_scale * col_std, size=df_noisy[col].shape)
        df_noisy[col] = df_noisy[col] + noise
        scaling_factors = np.random.uniform(scaling_range[0], scaling_range[1], size=df_noisy[col].shape)
        df_noisy[col] = df_noisy[col] * scaling_factors

    return df_noisy

def split_into_windows(df, window_size):
    """
    Split a DataFrame into windows of fixed size
    
    Args:
        df (DataFrame): Input DataFrame
        window_size (int): Size of each window
        
    Returns:
        list: List of DataFrame windows
    """
    return [df.iloc[i:i+window_size] 
            for i in range(0, len(df), window_size) 
            if len(df.iloc[i:i+window_size]) == window_size]