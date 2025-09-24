import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from ..download_data.main import download

from sklearn.linear_model import LogisticRegression

def learn():
    data_path = download()
    df = pd.read_csv(data_path)
    data = np.array(df)

    m, n = data.shape
    np.random.shuffle(data)

    data_dev = data[0:1000].T
    Y_dev = data_dev[0].T
    X_dev = data_dev[1:n].T
    X_dev = X_dev / 255.

    data_train = data[1000:m].T
    Y_train = data_train[0].T
    X_train = data_train[1:n].T
    X_train = X_train / 255.

    clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=100)
    clf.fit(X_train, Y_train)

    y_pred = clf.predict(X_dev)
    acc = accuracy_score(Y_dev, y_pred)
    print(f'accuracy: {acc}')


