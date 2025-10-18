from src.data_preparation import main_q1, main_q2
from src.model_training import main_q3

def main():
    print("\n" + "=" * 70)
    print("ECG TIME SERIES CLASSIFICATION")
    print("=" * 70)

    main_q1()
    X, Y, _ = main_q2()
    main_q3(X, Y)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()