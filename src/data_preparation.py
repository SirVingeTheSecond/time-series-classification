import pandas as pd
import numpy as np
import wfdb
import os
from config import PATIENTS, NON_BEAT, ABNORMAL_BEAT, DATA_DIR


def load_annotations(patients, data_dir):
    print("Loading annotations for all patients...")
    df = pd.DataFrame()

    for patient_id in patients:
        file_name = os.path.join(data_dir, patient_id)

        try:
            annotation = wfdb.rdann(file_name, 'atr')
            types = annotation.symbol

            values, counts = np.unique(types, return_counts=True)
            df1 = pd.DataFrame({
                'types': values,
                'val': counts,
                'patient_id': [patient_id] * len(counts)
            })
            df = pd.concat([df, df1], axis=0)

        except Exception as e:
            print(f"Warning: Could not load patient {patient_id}: {e}")
            continue

    print(f"Loaded annotations for {len(df['patient_id'].unique())} patients")
    return df


def classify_beats(df, abnormal_beat):
    df['class'] = -1
    df.loc[df['types'] == 'N', 'class'] = 0
    df.loc[df['types'].isin(abnormal_beat), 'class'] = 1
    return df


def display_class_distribution(df):
    distribution = df.groupby('class').val.sum().sort_values(ascending=False)

    print("\n" + "=" * 50)
    print("Beat Class Distribution:")
    print("=" * 50)
    print(f"Normal beats (0):    {distribution.get(0, 0):>6}")
    print(f"Abnormal beats (1):  {distribution.get(1, 0):>6}")
    print(f"Non-beats (-1):      {distribution.get(-1, 0):>6}")
    print(f"Total:               {distribution.sum():>6}")
    print("=" * 50)

    return distribution


def main_q1():
    print("\n" + "=" * 70)
    print("QUESTION 1: Beat Classification")
    print("=" * 70)

    df = load_annotations(PATIENTS, DATA_DIR)

    print("\nAll beat types found:")
    print(df.groupby('types').val.sum().sort_values(ascending=False))

    df = classify_beats(df, ABNORMAL_BEAT)
    distribution = display_class_distribution(df)

    print("\n" + "=" * 50)
    print("Verification:")
    print("=" * 50)
    expected = {-1: 3186, 0: 75052, 1: 34409}
    for class_label, expected_count in expected.items():
        actual_count = distribution.get(class_label, 0)
        status = "PASS" if actual_count == expected_count else "FAIL"
        print(f"Class {class_label:2}: Expected {expected_count:5}, Got {actual_count:5} {status}")

    return df


if __name__ == "__main__":
    df_result = main_q1()