import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from config import BATCH_SIZE, EPOCHS_CNN, THRESHOLD, FIGURES_DIR
import os


def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(Dropout(rate=0.2))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Question 4
def train_cnn(X_train, y_train, X_validation, y_validation):
    print("\n" + "=" * 70)
    print("QUESTION 4: CNN Model Training")
    print("=" * 70)

    X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_valid_reshaped = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))

    print(f"\nReshaped for CNN:")
    print(f"X_train_reshaped shape: {X_train_reshaped.shape}")
    print(f"X_valid_reshaped shape: {X_valid_reshaped.shape}")

    model = build_cnn_model(input_shape=(X_train_reshaped.shape[1], 1))

    print("\nModel Architecture:")
    model.summary()

    print(f"\nTraining CNN model for {EPOCHS_CNN} epochs...")
    history = model.fit(
        X_train_reshaped, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_CNN,
        verbose=1,
        validation_data=(X_valid_reshaped, y_validation)
    )

    print("\n" + "=" * 70)
    print("CNN Training Complete!")
    print("=" * 70)

    return model, X_train_reshaped, X_valid_reshaped, history

# Question 5
def evaluate_cnn(model, X_valid_reshaped, y_validation):
    print("\n" + "=" * 70)
    print("QUESTION 5: CNN Model Evaluation")
    print("=" * 70)

    y_pred_cnn = model.predict(X_valid_reshaped)

    print("\nClassification Report:")
    print(classification_report(y_validation, y_pred_cnn > THRESHOLD))

    return y_pred_cnn

# Question 6
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred > THRESHOLD)
    cm = cm / cm.astype(np.float64).sum(axis=1, keepdims=True)

    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)

    save_path = os.path.join(FIGURES_DIR, f'{title.lower().replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {save_path}")

    plt.show()

    return cm


def main_q4_q5_q6(X_train, y_train, X_validation, y_validation):
    model, X_train_reshaped, X_valid_reshaped, history = train_cnn(
        X_train, y_train, X_validation, y_validation
    )

    y_pred_cnn = evaluate_cnn(model, X_valid_reshaped, y_validation)

    print("\n" + "=" * 70)
    print("QUESTION 6: Confusion Matrix")
    print("=" * 70)

    cm = plot_confusion_matrix(y_validation, y_pred_cnn, title="CNN Confusion Matrix")

    return model, y_pred_cnn, cm, X_train_reshaped, X_valid_reshaped