import torch

import openpifpaf


class Cifar10Net(openpifpaf.network.BaseNetwork):
    """Small network for Cifar10."""
    def __init__(self):
        super().__init__('cifar10net', stride=16, out_features=128)
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 2, 1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 2, 1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = torch.nn.Conv2d(64, 128, 3, 2, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        return x
