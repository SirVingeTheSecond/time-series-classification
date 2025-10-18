from src.data_preparation import main_q1, main_q2
from src.model_training import main_q3
from src.model_cnn import main_q4_q5_q6

def main():
    print("\n" + "=" * 70)
    print("ECG TIME SERIES CLASSIFICATION")
    print("=" * 70)

    main_q1()
    X, Y, _ = main_q2()
    X_train, X_validation, y_train, y_validation = main_q3(X, Y)

    main_q4_q5_q6(X_train, y_train, X_validation, y_validation)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()