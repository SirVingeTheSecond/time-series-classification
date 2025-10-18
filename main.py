from src.data_preparation import main_q1, main_q2
from src.model_training import main_q3
from src.model_cnn import main_q4_q5_q6
from src.model_lstm import main_q7_q8


def main():
    print("\n" + "=" * 70)
    print("ECG TIME SERIES CLASSIFICATION")
    print("=" * 70)

    main_q1()
    X, Y = main_q2()
    X_train, X_validation, y_train, y_validation = main_q3(X, Y)

    model_cnn, y_pred_cnn, cm_cnn, X_train_reshaped, X_valid_reshaped = main_q4_q5_q6(
        X_train, y_train, X_validation, y_validation
    )

    model_lstm, y_pred_lstm, cm_lstm = main_q7_q8(
        X_train_reshaped, y_train, X_valid_reshaped, y_validation
    )

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    # Calculate accuracies
    cnn_acc = ((y_pred_cnn > 0.5).astype(int).flatten() == y_validation.flatten()).mean() * 100
    lstm_acc = ((y_pred_lstm > 0.5).astype(int).flatten() == y_validation.flatten()).mean() * 100

    print("\nModel comparison:")
    print(f"CNN Validation Accuracy:  {cnn_acc:.2f}%")
    print(f"LSTM Validation Accuracy: {lstm_acc:.2f}%")

if __name__ == "__main__":
    main()