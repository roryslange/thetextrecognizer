from .dataBuilder import dataframeFromCsv

def train():
    df = dataframeFromCsv()
    trainingNumRows = (int) (len(df) * .8)
    testingNumRows = len(df) - trainingNumRows
    training = df[:trainingNumRows]
    testing = df[trainingNumRows:testingNumRows:]
    print(f'training size (80%): {trainingNumRows}\ntesting size (20%): {testingNumRows}')