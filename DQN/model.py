import torch.nn as nn

import torch

from DQN.constants import SEQUENCE_LENGTH, BATCH_SIZE


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.reshape((x.size(0),) + self.shape)


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class DragonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.const_channel = 64
        self.common_feature = nn.Sequential(
            nn.Conv2d(4, self.const_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.const_channel),
            nn.ReLU(),
            nn.Conv2d(self.const_channel, self.const_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.const_channel),
            nn.ReLU(),
            nn.Conv2d(self.const_channel, self.const_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.const_channel),
            nn.ReLU(),
            Reshape(self.const_channel * 70),
            nn.Linear(self.const_channel * 70, self.const_channel),
            nn.ReLU(),
        )
        self.action_feature = nn.Sequential(
            nn.Linear(self.const_channel, 2),
            nn.Softmax(),
        )
        self.common_feature.apply(init_weights)

    def forward(self, x):
        x = self.common_feature(x)
        q_sa = self.action_feature(x)
        return q_sa


if __name__ == '__main__':
    print(torch.cuda.is_available())
    dm = DragonModel()
    image = torch.zeros((1500 // 20, 1000 // 20, 1)).permute(2, 0, 1).unsqueeze(0)
    image_single = image
    image_list = []
    for _ in range(BATCH_SIZE * SEQUENCE_LENGTH):
        image_list.append(image)

    image = torch.cat(image_list, dim=0)

    y = dm(image)
