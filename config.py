import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Patient IDs
PATIENTS = ['100', '101', '102', '103', '104', '105', '106', '107',
            '108', '109', '111', '112', '113', '114', '115', '116',
            '117', '118', '119', '121', '122', '123', '124', '200',
            '201', '202', '203', '205', '207', '208', '209', '210',
            '212', '213', '214', '215', '217', '219', '220', '221',
            '222', '223', '228', '230', '231', '232', '233', '234']

# Beat type classifications
NON_BEAT = ['[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', "'",
            '^', '|', '~', '+', 's', 'T', '*', 'D', '=', '"', '@', 'Q', '?']

ABNORMAL_BEAT = ['L', 'R', 'V', '/', 'A', 'f', 'F', 'j', 'a', 'E', 'J', 'e', 'S']

# Signal parameters
SAMPLING_FREQUENCY = 360  # Hz
INTERVAL = 3  # seconds before and after peak
WINDOW_SIZE = 2 * INTERVAL * SAMPLING_FREQUENCY  # 2160 samples

# GPU/CPU
USE_GPU = False  # Set to False to force training using CPU

# Model parameters
TEST_SIZE = 0.33
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS_BASELINE = 10
EPOCHS_CNN = 2

# LSTM parameters depend on USE_GPU
# 'None' means use full dataset
if USE_GPU:
    LSTM_TRAIN_SUBSET = None
    EPOCHS_LSTM = 5
else:
    LSTM_TRAIN_SUBSET = None
    EPOCHS_LSTM = 1

THRESHOLD = 0.5