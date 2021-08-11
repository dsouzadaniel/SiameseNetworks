# Libraries
import torch
import torch.nn as nn
from typing import OrderedDict


# Model Definition

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('Conv1', nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(10, 10))),
            ('MaxPool1', nn.MaxPool2d(kernel_size=(2, 2))),
            ('Conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(7, 7))),
            ('Relu1', nn.ReLU()),
            ('MaxPool2', nn.MaxPool2d(kernel_size=(2, 2))),
            ('Conv3', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 4))),
            ('Relu2', nn.ReLU()),
            ('MaxPool3', nn.MaxPool2d(kernel_size=(2, 2))),
            ('Conv4', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4))),
            ('Relu3', nn.ReLU()),
            ('Flatten1', nn.Flatten()),
            ('Linear1', nn.Linear(in_features=9216, out_features=4096)),
            ('Sigmoid1', nn.Sigmoid()),
        ]))

        self.dist_metric = nn.L1Loss(reduction='none')

        self.fc = nn.Sequential(OrderedDict([
            ('LinearFinal', nn.Linear(in_features=4096, out_features=1)),
            ('SigmoidFinal', nn.Sigmoid()),
        ]))

    def forward(self, img1, img2):
        first_twin = self.model(img1)
        second_twin = self.model(img2)

        feature_vector = self.dist_metric(first_twin, second_twin)

        return self.fc(feature_vector).view(-1)
