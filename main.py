from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision.models import resnet18
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image


class CNNLSTM(nn.Module):

    def __init__(self, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.resnet = resnet18(pretrained=True, num_classes=100)
        self.lstm = nn.LSTM(input_size=100, hidden_size=256, num_layers=3, dropout=0.2)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """The inputs are batched images of size (batch_size, channels, height, width)"""
        with torch.no_grad():
            x = self.resnet(x)

        hidden = None
        for batch_index in range(x.size(0)):
            frame_features = x[batch_index]
            out, hidden = self.lstm(frame_features.unsqueeze(0), hidden)

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x


class StreamDataset(Dataset):

    def __init__(self, root_dir, phase):
        self.images = np.load(Path(root_dir) / phase / "images.npy")
        self.labels = np.load(Path(root_dir) / phase / "labels.npy")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]
        return image, label


def get_loaders(root="data"):
    train_dataloader = DataLoader(StreamDataset(root, "train"), batch_size=64, shuffle=False)
    test_dataloader = DataLoader(StreamDataset(root, "test"), batch_size=64, shuffle=False)
    return train_dataloader, test_dataloader


def train(model: CNNLSTM, loader_train: DataLoader, loader_test: DataLoader):
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 3
    for epoch in range(n_epochs):
        model.train()
        for i, (image_batch, label_batch) in enumerate(loader_train):
            prediction = model(image_batch)
            loss = loss_fn(prediction, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch", epoch, "batch", i, "loss", loss)
            class_id = prediction.argmax().item()


if __name__ == '__main__':
    model = CNNLSTM()
    print("Model loaded.")
    train_dl, test_dl = get_loaders("data")
    print("Initialized data loaders.")
    train(model, train_dl, test_dl)
