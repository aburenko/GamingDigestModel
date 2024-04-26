import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch import optim
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import transforms
from sklearn.metrics import precision_score, recall_score

resnet_weights = ResNet18_Weights.IMAGENET1K_V1


class HighlightModel(nn.Module):

    def __init__(self):
        super(HighlightModel, self).__init__()
        self.resnet = resnet18(weights=resnet_weights)
        self.fc_to_lstm = nn.Linear(1000, 256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True)
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
        self.preprocess = resnet_weights.transforms()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        raw_image = Image.open(self.images[idx])
        image = self.preprocess(raw_image)
        label = self.labels[idx]
        return image, label


def get_loaders(root="data"):
    train_dataloader = DataLoader(StreamDataset(root, "train"), batch_size=16, shuffle=False, num_workers=2)
    test_dataloader = DataLoader(StreamDataset(root, "test"), batch_size=100, shuffle=False, num_workers=2)
    return train_dataloader, test_dataloader


def train(model: HighlightModel, loader_train: DataLoader, loader_test: DataLoader):
    is_slurm = True
    tqdm_out = open(os.devnull, 'w') if is_slurm else None
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 50
    for epoch in range(n_epochs):
        print(f"Epoch {epoch} ==========")
        model.train()
        all_true = []
        all_pred = []
        for i, (image_batch, label_batch) in enumerate(tqdm.tqdm(loader_train, file=tqdm_out)):
            prediction = model(image_batch.to(device=device))
            loss = loss_fn(prediction, F.one_hot(label_batch, num_classes=2).float().to(device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_pred.extend(torch.argmax(prediction, 1).tolist())
            all_true.extend(label_batch.tolist())
        print("TRAIN set:")
        print_stats(all_true, all_pred)

        model.eval()
        all_true = []
        all_pred = []
        for i, (image_batch, label_batch) in enumerate(tqdm.tqdm(loader_test, file=tqdm_out)):
            prediction = model(image_batch.to(device=device))

            all_pred.extend(torch.argmax(prediction, 1).tolist())
            all_true.extend(label_batch.tolist())
        print("TEST set:")
        print_stats(all_true, all_pred)


def print_stats(all_true, all_pred):
    precision = precision_score(all_true, all_pred)
    recall = recall_score(all_true, all_pred)
    print(f"Precision: {precision} and recall {recall}")
    correct_guesses = np.sum(np.bitwise_and(all_true, all_pred))
    true_highlights = np.sum(all_true)
    accuracy = np.round(correct_guesses / true_highlights, 2)
    print(f"Number of correct guesses {correct_guesses}/{true_highlights}={accuracy}")


if __name__ == '__main__':
    device = torch.device("cuda")
    model = HighlightModel()
    model.to(device)
    print("Model loaded.")
    train_dl, test_dl = get_loaders("data")
    print("Initialized data loaders.")
    train(model, train_dl, test_dl)
