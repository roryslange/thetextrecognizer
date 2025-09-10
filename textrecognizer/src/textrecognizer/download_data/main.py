import kagglehub as kh
import pandas as pd

# downloads it into your computers cache
def download():
    path = kh.competition_download('digit-recognizer', path='train.csv')
    print("Downloaded to:", path)
    return path

# downloads the dataset locally to a parquet file
# youll need to add this to .gitignore for whatever path you choose to download it at
def download_locally(download_path):
    path = download()
    df = pd.read_csv(path)
    df.to_parquet(download_path, engine="pyarrow", compression="snappy")
