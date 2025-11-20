from kagglehub import dataset_download
import os
import shutil

DATASET_HANDLE = "arashnic/hr-analytics-job-change-of-data-scientists"
TARGET_DIR = "data/raw"          
TARGET_FILENAME = "aug_train.csv"   

def main():
    print("Downloading dataset from KaggleHub...")

    dataset_path = dataset_download(DATASET_HANDLE)
    print(f"Dataset downloaded to: {dataset_path}")

    src_file = os.path.join(dataset_path, TARGET_FILENAME)
    if not os.path.exists(src_file):
        raise FileNotFoundError(f"Không tìm thấy file {TARGET_FILENAME} trong {dataset_path}")

    os.makedirs(TARGET_DIR, exist_ok=True)
    dst_file = os.path.join(TARGET_DIR, TARGET_FILENAME)

    shutil.copy(src_file, dst_file)
    print(f"Copied {TARGET_FILENAME} to {dst_file}")

if __name__ == "__main__":
    main()
