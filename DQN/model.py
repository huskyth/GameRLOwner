import torch.nn as nn

import torch

from DQN.constants import SEQUENCE_LENGTH, BATCH_SIZE


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.reshape((x.size(0),) + self.shape)


class DragonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.const_channel = 64
        self.common_feature = nn.Sequential(
            nn.Conv2d(4, self.const_channel, 5, 2),
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
        )
        self.action_feature = nn.Sequential(
            nn.Linear(self.const_channel, 2),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.common_feature(x)
        q_sa = self.action_feature(x)
        return q_sa


if __name__ == '__main__':
    print(torch.cuda.is_available())
    dm = DragonModel().cuda()
    image = torch.zeros((1500, 1000, 4)).permute(2, 0, 1).unsqueeze(0).cuda()
    image_single = image
    image_list = []
    for _ in range(BATCH_SIZE * SEQUENCE_LENGTH):
        image_list.append(image)

    image = torch.cat(image_list, dim=0)

    y = dm(image)

    dm.step(image_single)
