import torch
from torch.utils.data import Dataset
import pandas as pd

class MNIST(Dataset):
    def __init__(self, df_labels, df_images):
        self.labels = torch.tensor(df_labels.values, dtype=torch.long)
        self.images = torch.tensor(df_images.values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx].view(1, 28, 28)
        label = self.labels[idx]
        return image, label
