from ..download_data.main import download
import pandas as pd

def dataframeFromCsv():
    path = download()
    df = pd.read_csv(path)
    # print(df.head())
    return df
