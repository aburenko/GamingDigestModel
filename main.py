from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import transforms


class HighlightModel(nn.Module):

    def __init__(self):
        super(HighlightModel, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.fc_to_lstm = nn.Linear(1000, 256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, dropout=0.2, batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """The inputs are batched images of size (sequence size, channels, height, width)"""
        with torch.no_grad():
            x = self.resnet(x)

        x = self.fc_to_lstm(x)
        x, _ = self.lstm(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


class StreamDataset(Dataset):

    def __init__(self, root_dir, phase):
        self.images = np.load(Path(root_dir) / phase / "images.npy")
        # paths from scalable are e.g.: http://localhost:8686/items/br-stream/br-stream/br-stream-0000012.jpg
        self.images = [path.replace("http://localhost:8686/items/", "/home/burenko/data/") for path in self.images]
        self.labels = np.load(Path(root_dir) / phase / "labels.npy")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = preprocess(Image.open(self.images[idx]))

        label = self.labels[idx]
        return image, label


def get_loaders(root="data"):
    train_dataloader = DataLoader(StreamDataset(root, "train"), batch_size=64, shuffle=False)
    test_dataloader = DataLoader(StreamDataset(root, "test"), batch_size=64, shuffle=False)
    return train_dataloader, test_dataloader


def train(model: HighlightModel, loader_train: DataLoader, loader_test: DataLoader):
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 3
    for epoch in range(n_epochs):
        model.train()
        torch.autograd.set_detect_anomaly(True)
        for i, (image_batch, label_batch) in enumerate(loader_train):
            prediction = model(image_batch)
            loss = loss_fn(prediction, F.one_hot(label_batch, num_classes=2).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch", epoch, "batch", i, "loss", loss)
            class_id = prediction.argmax().item()


if __name__ == '__main__':
    model = HighlightModel()
    print("Model loaded.")
    train_dl, test_dl = get_loaders("data")
    print("Initialized data loaders.")
    train(model, train_dl, test_dl)
