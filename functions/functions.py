import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

def calculate_sampling_strategy(y, max_threshold=2000, multiplier=10):
    """
    Calculate the sampling strategy for SMOTE based on the target counts.

    Parameters:
        y (array-like): Target variable for the training set.
        max_threshold (int): Maximum sample count for which SMOTE will be applied.
        multiplier (int): Factor by which the minority classes will be oversampled.

    Returns:
        dict: Sampling strategy for SMOTE.
    """
    sampling_strategy = {}
    unique_classes = pd.Series(y).value_counts()

    for cls, count in unique_classes.items():
        if count < max_threshold:  # Apply SMOTE only to classes with fewer than `max_threshold` samples
            sampling_strategy[cls] = count * multiplier
    return sampling_strategy



def apply_smote(X, y, sampling_strategy, random_state=42):
    """
    Apply SMOTE to balance the classes in the training set.

    Parameters:
        X (pd.DataFrame): Training features.
        y (array-like): Training labels.
        sampling_strategy (dict): Sampling strategy for SMOTE.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Resampled training features and labels.
    """
    if sampling_strategy:
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        return smote.fit_resample(X, y)
    return X, y


def scale_features(X_train, X_test, numerical_cols):
    """
    Scale numerical features using StandardScaler.

    Parameters:
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Test feature set.
        numerical_cols (list): List of numerical feature columns.

    Returns:
        tuple: Scaled training and test feature sets.
    """
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    return X_train_scaled, X_test_scaled


def custom_train_test_split(
    X, y, test_size=0.2, max_threshold=2000, multiplier=10, random_state=42
):
    """
    Perform a stratified train-test split, apply SMOTE for undersampled classes, and scale features.

    Parameters:
        X (pd.DataFrame): Feature dataset.
        y (pd.Series): Target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        max_threshold (int): Maximum sample count for which SMOTE will be applied.
        multiplier (int): Factor by which the minority classes will be oversampled.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Scaled training and test sets with resampled training labels.
    """
    # Label encoding for the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    # Calculate SMOTE sampling strategy
    sampling_strategy = calculate_sampling_strategy(y_train, max_threshold, multiplier)

    # Apply SMOTE to balance the training set
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train, sampling_strategy, random_state)

    # Scale numerical features
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
    X_train_scaled, X_test_scaled = scale_features(X_train_resampled, X_test, numerical_cols)

    # Decode the labels for output consistency
    y_train_resampled_decoded = label_encoder.inverse_transform(y_train_resampled)
    y_test_decoded = label_encoder.inverse_transform(y_test)

    return X_train_scaled, X_test_scaled, y_train_resampled_decoded, y_test_decoded