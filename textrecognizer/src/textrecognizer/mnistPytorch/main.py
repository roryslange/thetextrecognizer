import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from ..download_data.main import download
from .MLP import MLP
from .mnist import MNIST

def main():
    parser = argparse.ArgumentParser(description="a script that trains and evaluates an MLP on MNIST dataset")
    parser.add_argument("--train", type=bool, help="boolean: should train or load a pre-trained model")
    args = parser.parse_args()

    csv_path = download()
    df = pd.read_csv(csv_path)

    m, n = df.shape
    data_dev = df[0:1000]
    data_train = df[1000:m]
    y_train = data_train.pop('label')
    x_train = data_train / 255.
    y_dev = data_dev.pop('label')
    x_dev = data_dev / 255.

    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(MNIST(y_train, x_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(MNIST(y_dev, x_dev), batch_size=64, shuffle=False)

    if args.train:
        print("training the model...")
        # train the model
        epochs = 5
        for epoch in range(epochs):
            loss_counter = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                loss_counter += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss_counter/len(train_loader):.4f}")
    else:
        model_save_path = "./src/textrecognizer/mnistPytorch/models/mnist_model.pth"
        print("skipping training the model...")
        model.load_state_dict(torch.load(model_save_path))

    # start testing
    print("Begin testing")
    test_counter = 0
    test_correct = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_counter += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * test_correct / test_counter:.2f}%")

    if args.train:
        model_save_path = "./src/textrecognizer/mnistPytorch/models/mnist_model.pth"
        print("trained new model, saving state...")
        torch.save(model.state_dict(), model_save_path)
        print(f"model saved at {model_save_path}")


