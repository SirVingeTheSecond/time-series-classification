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


def read_ecg(path):
    record = wfdb.rdrecord(path)
    annot = wfdb.rdann(path, 'atr')

    ecg_sig = record.p_signal
    assert record.fs == 360, f'Sample frequency is {record.fs}, expected 360 Hz'

    annot_type = annot.symbol
    annot_sample = annot.sample

    return ecg_sig, annot_type, annot_sample


def make_XY(ecg_sig, df_annot, interval, num_cols, abnormal_beat):
    # this function builds the X,Y matrices for each beat
    # it also returns the original symbols for Y

    num_row = len(df_annot)

    x = np.zeros((num_row, num_cols))
    y = np.zeros((num_row, 1))
    symbol = []

    # count the rows
    row_size = 0

    for annot_sample, annot_type in zip(df_annot.annot_sample.values, df_annot.annot_type.values):

        left = max([0, (annot_sample - interval * 360)])
        right = min([len(ecg_sig), (annot_sample + interval * 360)])
        xx = ecg_sig[left: right]
        if len(xx) == num_cols:
            x[row_size, :] = xx
            y[row_size, :] = int(annot_type in abnormal_beat)
            symbol.append(annot_type)
            row_size += 1

    x = x[:row_size, :]
    y = y[:row_size, :]
    return x, y, symbol


def build_dataset(patients, interval, fs, abnormal_beat, data_dir):
    print("\n" + "="*70)
    print("Building dataset from ECG signals...")
    print("="*70)

    # This specify the length of the heartbeats to be extracted
    num_cols = 2 * interval * fs
    X = np.zeros((1, num_cols))
    Y = np.zeros((1, 1))
    annot_symb = []

    # This list stores the number of extracted heartbeats for each patient
    num_beats = []

    for idx, patient_id in enumerate(patients, 1):
        file_path = os.path.join(data_dir, patient_id)

        try:
            ecg_sig, annot_type, annot_sample = read_ecg(file_path)

            # Since there are two leads of ECGs for each record, we only select the first lead to work with
            ecg_sig = ecg_sig[:, 0]

            # We simply remove the "non-beats" beats from the df_annot dataframe and only keep "normal" and "abnormal" beats
            df_annot = pd.DataFrame({
                'annot_type': annot_type,
                'annot_sample': annot_sample
            })
            df_annot = df_annot[~df_annot['annot_type'].isin(NON_BEAT)]

            # The "make_XY" builds the x and y matrices for each extracted heartbeat
            x, y, symbol = make_XY(ecg_sig, df_annot, interval, num_cols, abnormal_beat)
            annot_symb = annot_symb + symbol
            num_beats.append(x.shape[0])
            X = np.append(X, x, axis=0)
            Y = np.append(Y, y, axis=0)

            if idx % 10 == 0:
                print(f"Processed {idx}/{len(patients)} patients... Total beats: {X.shape[0]-1}")

        except Exception as e:
            print(f"Warning: Could not process patient {patient_id}: {e}")
            continue

    X = X[1:, :]
    Y = Y[1:, :]

    print("\n" + "="*70)
    print("Dataset Building Complete!")
    print("="*70)
    print(f"Total patients processed: {len(num_beats)}")
    print(f"Total heartbeats extracted: {X.shape[0]}")
    print(f"Feature dimensions: {X.shape[1]} samples per beat")
    print(f"Normal beats: {np.sum(Y == 0)}")
    print(f"Abnormal beats: {np.sum(Y == 1)}")
    print("="*70)

    return X, Y, annot_symb


def main_q2():
    print("\n" + "="*70)
    print("QUESTION 2: Build Dataset")
    print("="*70)

    from config import INTERVAL, SAMPLING_FREQUENCY

    X, Y, annot_symb = build_dataset(
        patients=PATIENTS,
        interval=INTERVAL,
        fs=SAMPLING_FREQUENCY,
        abnormal_beat=ABNORMAL_BEAT,
        data_dir=DATA_DIR
    )

    print(f"\nX shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Unique annotation symbols: {np.unique(annot_symb)}")

    return X, Y, annot_symb


if __name__ == "__main__":
    df_result = main_q1()