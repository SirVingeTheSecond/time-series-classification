# Only for downloading the dataset

import wfdb
import os
from config import DATA_DIR


def download_mitdb():

    print("Downloading MIT-BIH Arrhythmia Database...")
    print(f"Target directory: {DATA_DIR}")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    try:
        wfdb.dl_database('mitdb', DATA_DIR)
        print("\nDownload complete!")

        # Verify download
        files = os.listdir(DATA_DIR)
        print(f"\nDownloaded {len(files)} files")

    except Exception as e:
        print(f"Error downloading database: {e}")


if __name__ == "__main__":
    download_mitdb()