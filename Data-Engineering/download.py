import os
import kaggle
import zipfile

DOWNLOAD_DIR = "./data"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
DATASET_SLUG = "nguyenductuandat/lalb-ais-port-congestion-62023-122025"

print("Downloading ais_2023_2025_clean.parquet...")
try:
    kaggle.api.dataset_download_file(
        dataset=DATASET_SLUG,
        file_name="ais_2023_2025_clean.parquet",
        path=DOWNLOAD_DIR,
        force=True,
        quiet=False
    )
    
    zip_path = os.path.join(DOWNLOAD_DIR, "ais_2023_2025_clean.parquet.zip")
    if os.path.exists(zip_path):
        print("Extracting zip file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DOWNLOAD_DIR)
        os.remove(zip_path) # Clean up zip file
    print("Download and extraction complete!")
except Exception as e:
    print(f"Error during download: {e}")
