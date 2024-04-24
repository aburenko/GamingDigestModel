from torchvision.models.video.resnet import VideoResNet
from torch import nn


class HighlightModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.main_model = VideoResNet()
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(250, 50),
            nn.Dropout(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        x = self.main_model.forward(x)
        return self.classifier_head.forward(x)
