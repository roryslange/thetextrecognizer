import pandas as pd
from ..download_data.main import download

def main():
    csv_path = download()
    df = pd.read_csv(csv_path)

    m, n = df.shape
    data_dev = df[0:1000]
    data_train = df[1000:m]
    y_train = data_train.pop('label')
    x_train = data_train

    print(y_train.shape)
    print(x_train.shape)