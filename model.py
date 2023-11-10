import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ContrastiveLearningModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = self.get_encoder()
        self.projection_head = ProjectionHead(2048, 2048, 128)

    def forward(self, x):
        x = self.encoder(x)
        x = self.projection_head(torch.squeeze(x))
        return x

    def get_encoder(self):
        resnet = models.resnet50()
        resnet.maxpool = nn.Identity()
        resnet.fc = nn.Identity()
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        return resnet


class DownstreamModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        model_head: nn.Module,
        num_classes: int,
        batch_size: int,
    ):
        super().__init__()

        self.encoder = encoder
        self.model_head = model_head
        self.num_classes = num_classes
        self.batch_size = batch_size

        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.model_head.parameters():
            param.requires_grad = True

    def forward(self, x):
        encoded = self.encoder(x)
        reshaped_encoded = encoded.view(self.batch_size, 2, 32, 32)
        output = self.model_head(reshaped_encoded)
        return output


class BaselineModel(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
