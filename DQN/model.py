import numpy as np
import torch.nn as nn

import torch


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),) + self.shape)


class DragonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.const_channel = 64
        self.feature = nn.Sequential(
            nn.Conv2d(3, self.const_channel, 5, 2),
            nn.ReLU(),
            nn.Conv2d(self.const_channel, self.const_channel, 3, 2),
            nn.ReLU(),
            nn.Conv2d(self.const_channel, self.const_channel, 3, 2),
            nn.ReLU(),
            nn.Conv2d(self.const_channel, self.const_channel, 3, 2),
            nn.ReLU(),
            nn.Conv2d(self.const_channel, self.const_channel, 3, 2),
            nn.ReLU(),
            nn.Conv2d(self.const_channel, self.const_channel, 3, 2),
            nn.ReLU(),
            nn.Conv2d(self.const_channel, self.const_channel, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(self.const_channel, self.const_channel, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(self.const_channel, self.const_channel, 3, 2, 1),
            nn.ReLU(),
            Reshape(self.const_channel * 6),
            nn.Linear(self.const_channel * 6, self.const_channel),
            nn.ReLU(),
            nn.Linear(self.const_channel, 2),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.feature(x)
        return x


if __name__ == '__main__':
    print(torch.cuda.is_available())
    dm = DragonModel()
    image = torch.zeros((1500, 1000, 3)).permute(2, 0, 1).unsqueeze(0)
    y = dm(image)
    print(y, y.shape)
