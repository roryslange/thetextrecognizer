import kagglehub

def hello():

    # Download latest version
    path = kagglehub.dataset_download("hojjatk/mnist-dataset")

    print("Path to dataset files:", path)
