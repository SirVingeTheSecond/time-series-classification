# ECG Time Series Classification

Deep Learning assignment for classifying ECG heartbeats using CNN and LSTM models on the MIT-BIH Arrhythmia Database.

## Setup

1. **Install dependencies:**
```bash
   pip install -r requirements.txt
```

2. **Download the dataset (run once):**
```bash
   python download_data.py
```
   This downloads the MIT-BIH Arrhythmia Database (~90 MB) to the `data/` folder.

3. **Run the complete analysis:**
```bash
   python main.py
```

## GPU Support (Optional)

TensorFlow only supports Windows natively up to version 2.10, which requires Python 3.10. For faster LSTM training with NVIDIA GPU:

1. **Install Python 3.10** (required for TensorFlow 2.10)
   - Download from https://www.python.org/downloads/release/python-31011/

2. **Recreate virtual environment:**
```bash
   py -3.10 -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install tensorflow==2.10.0
   pip install pandas numpy matplotlib wfdb scikit-learn seaborn
```

3. **Install CUDA 11.2 + cuDNN 8.1:**
   - CUDA 11.2: https://developer.nvidia.com/cuda-11.2.0-download-archive
   - cuDNN 8.1: https://developer.nvidia.com/rdp/cudnn-archive
   - Copy cuDNN files to CUDA installation directory

4. **Enable GPU in `config.py`:**
```python
   USE_GPU = True
```

5. **Verify:**
```bash
   python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

## Results

Plots are saved to `results/figures/`.