import numpy as np
from sklearn.model_selection import train_test_split
from config import TEST_SIZE, RANDOM_STATE


def split_data(X, Y):
    """
    Q3: Split dataset into training and validation sets

    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    Y : numpy.ndarray
        Label array

    Returns:
    --------
    X_train, X_validation, y_train, y_validation : numpy.ndarray
        Split datasets
    """
    print("\n" + "=" * 70)
    print("QUESTION 3: Train-Test Split")
    print("=" * 70)

    X_train, X_validation, y_train, y_validation = train_test_split(
        X, Y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print(f"\nDataset split with test_size={TEST_SIZE}, random_state={RANDOM_STATE}")
    print(f"\nTraining set:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  Normal beats: {np.sum(y_train == 0)}")
    print(f"  Abnormal beats: {np.sum(y_train == 1)}")

    print(f"\nValidation set:")
    print(f"  X_validation shape: {X_validation.shape}")
    print(f"  y_validation shape: {y_validation.shape}")
    print(f"  Normal beats: {np.sum(y_validation == 0)}")
    print(f"  Abnormal beats: {np.sum(y_validation == 1)}")
    print("=" * 70)

    return X_train, X_validation, y_train, y_validation


def main_q3(X, Y):
    return split_data(X, Y)