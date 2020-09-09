import torch

import openpifpaf


class Cifar10Net(openpifpaf.network.BaseNetwork):
    """Small network for Cifar10."""
    def __init__(self):
        super().__init__('cifar10net', stride=16, out_features=128)
        self.conv1 = torch.nn.Conv2d(3, 6, 5, 2, 2, padding_mode='reflect')
        self.pool = torch.nn.MaxPool2d(3, 2, 1)
        self.conv2 = torch.nn.Conv2d(6, 32, 5, 2, 2, padding_mode='reflect')
        self.conv3 = torch.nn.Conv2d(32, 64, 5, 2, 2, padding_mode='reflect')
        self.conv4 = torch.nn.Conv2d(64, 128, 3, 1, 1, padding_mode='reflect')

    def forward(self, *args):
        x = args[0]
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        return x
