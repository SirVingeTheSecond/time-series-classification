import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from sklearn.metrics import classification_report
from config import BATCH_SIZE, EPOCHS_LSTM, THRESHOLD, USE_GPU, LSTM_TRAIN_SUBSET
from src.model_cnn import plot_confusion_matrix

def configure_device():
    if USE_GPU:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print(f"GPU Available: {physical_devices[0].name}")
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            return "GPU"
        else:
            print("GPU requested but not available, falling back to CPU")
            return "CPU"
    else:
        print("CPU mode enabled (USE_GPU=False in config.py)")
        tf.config.set_visible_devices([], 'GPU')
        return "CPU"

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(rate=0.2))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Question 7
# Using reduced dataset for faster training
def train_lstm(X_train_reshaped, y_train, X_valid_reshaped, y_validation):
    print("\n" + "=" * 70)
    print("QUESTION 7: LSTM Model Training")
    print("=" * 70)

    device = configure_device()

    if LSTM_TRAIN_SUBSET is not None:
        print(f"\nUsing subset of {LSTM_TRAIN_SUBSET} samples (CPU mode)")
        X_train_subset = X_train_reshaped[:LSTM_TRAIN_SUBSET]
        y_train_subset = y_train[:LSTM_TRAIN_SUBSET]
    else:
        print(f"\nUsing full training set: {X_train_reshaped.shape[0]} samples (GPU mode)")
        X_train_subset = X_train_reshaped
        y_train_subset = y_train

    print(f"Device: {device}")
    print(f"Training samples: {X_train_subset.shape[0]}")
    print(f"Epochs: {EPOCHS_LSTM}")

    model = build_lstm_model(input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]))

    print("\nModel:")
    model.summary()

    print(f"\nTraining LSTM model...")
    history = model.fit(
        X_train_subset, y_train_subset,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_LSTM,
        verbose=1,
        validation_data=(X_valid_reshaped, y_validation)
    )

    print("\n" + "=" * 70)
    print("LSTM Training Complete!")
    print("=" * 70)

    return model, history

# Question 8
def evaluate_lstm(model, X_valid_reshaped, y_validation):
    print("\n" + "=" * 70)
    print("QUESTION 8: LSTM Model Evaluation")
    print("=" * 70)

    y_pred_lstm = model.predict(X_valid_reshaped)

    print("\nClassification Report:")
    print(classification_report(y_validation, y_pred_lstm > THRESHOLD))

    return y_pred_lstm


def main_q7_q8(X_train_reshaped, y_train, X_valid_reshaped, y_validation):
    model, history = train_lstm(X_train_reshaped, y_train, X_valid_reshaped, y_validation)

    y_pred_lstm = evaluate_lstm(model, X_valid_reshaped, y_validation)

    print("\n" + "=" * 70)
    print("Generating LSTM Confusion Matrix...")
    print("=" * 70)

    cm = plot_confusion_matrix(y_validation, y_pred_lstm, title="LSTM Confusion Matrix")

    return model, y_pred_lstm, cm