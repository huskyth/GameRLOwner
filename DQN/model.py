import torch.nn as nn

import torch

from ChromeDragon.environment import DragonEnvironment, STATE_LENGTH
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
            nn.Conv2d(STATE_LENGTH + 1, self.const_channel, kernel_size=3, stride=2, padding=1),
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
        self.adv = nn.Sequential(
            nn.Linear(self.const_channel, 2),
        )
        self.v = nn.Linear(self.const_channel, 1)
        self.common_feature.apply(init_weights)

    def forward(self, x):
        x = self.common_feature(x)
        adv = self.adv(x)
        v = self.v(x)
        q = v + adv - (1 / adv.shape[-1] * adv.max(-1)[0])[:, None]
        return q


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    #
    # print(torch.cuda.is_available())
    # d_environment = DragonEnvironment()
    # s, is_terminate = d_environment.reset()
    #
    dm = DragonModel()
    # p0, p1 = [], []
    # for x in range(100):
    #     with torch.no_grad():
    #         p = dm(s)
    #     p0.append(p[0, 0])
    #     p1.append(p[0, 1])
    # plt.scatter(range(len(p0)), p0)
    # plt.scatter(range(len(p1)), p1)
    # plt.show()
    #
    # assert False
    image_list = []
    for _ in range(1 * SEQUENCE_LENGTH):
        image = torch.rand((1500 // 20, 1000 // 20, 5)).permute(2, 0, 1).unsqueeze(0)

        image_list.append(image)

    image = torch.cat(image_list, dim=0)

    y = dm(image)
