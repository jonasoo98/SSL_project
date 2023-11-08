import torch
import torch.nn as nn

from torchvision import models


class ProjectionHead(nn.Module):
    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, **kwargs
    ):
        super(ProjectionHead, self).__init__(**kwargs)
        self.linear1 = nn.Linear(
            in_features=in_features, out_features=hidden_features, bias=False
        )
        self.linear2 = nn.Linear(
            in_features=hidden_features, out_features=out_features, bias=False
        )

        self.relu = nn.ReLU()
        self.batch_normalization1 = nn.BatchNorm1d(num_features=hidden_features)
        self.batch_normalization2 = nn.BatchNorm1d(num_features=out_features)

    def forward(self, x):
        x = self.linear1(x)
        x = self.batch_normalization1(x)

        x = self.relu(x)
        x = self.linear2(x)
        x = self.batch_normalization2(x)
        return x


class BackboneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50()

        # Changing the layers like mentioned in the article
        self.resnet.maxpool = nn.Sequential()
        self.resnet.fc = nn.Sequential()
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False
        )

        self.projection_head = ProjectionHead(2048, 2048, 128)

    def forward(self, x):
        x = self.resnet(x)
        x = self.projection_head(torch.squeeze(x))
        return x
