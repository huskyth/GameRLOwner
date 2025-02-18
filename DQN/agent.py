import math
import random

import numpy
import torch

from DQN.buffer import DragonBuffer
from DQN.model import DragonModel
import torch.nn.functional as F
from torch.distributions import Bernoulli


class DragonAgent:
    def __init__(self, buffer, my_summary):
        self.is_cuda = torch.cuda.is_available()
        self.q_net = DragonModel()
        self.target_q_net = DragonModel()
        if self.is_cuda:
            self.q_net = self.q_net.cuda()
            self.target_q_net = self.target_q_net.cuda()

        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.buffer = buffer
        self.gamma = 0.99

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.epsilon = 0.001
        self.count = 0
        self.sample_count = 0
        self.epsilon_start = 0.95
        self.epsilon_end = 0.01
        self.epsilon_decay = 35000
        self.my_summary = my_summary

    @torch.no_grad()
    def _get_action(self, state):
        if self.is_cuda:
            state = state.cuda()
        action = self.q_net(state)
        action = torch.argmax(action, dim=-1)
        return action.detach().cpu().item()

    def sample_action(self, state, is_test):
        self.sample_count += 1
        if is_test:
            return self._get_action(state)
        # epsilon指数衰减
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 1)
        else:
            return self._get_action(state)

    def save(self):
        saved = {"model_state_dict": self.q_net.state_dict(), "optimizer_state_dict": self.optimizer.state_dict()}
        torch.save(saved, "best.pt")

    def load(self):
        saved = torch.load("best.pt")
        self.q_net.load_state_dict(saved["model_state_dict"])
        self.optimizer.load_state_dict(saved["optimizer_state_dict"])

    def update(self):
        state, action, reward, next_state, done = self.buffer.sample()
        state = torch.cat(state)
        action = torch.tensor(action, dtype=torch.int64).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
        next_state = torch.cat(next_state)
        done = torch.tensor(done).int().unsqueeze(1)
        if self.is_cuda:
            state = state.cuda()
            action = action.cuda()
            reward = reward.cuda()
            next_state = next_state.cuda()
            done = done.cuda()

        q_value = self.q_net(state).gather(1, action)

        next_action = torch.argmax(self.q_net(next_state), dim=-1).unsqueeze(1)
        with torch.no_grad():
            q_target = self.gamma * (1 - done) * self.target_q_net(next_state).gather(1, next_action) + reward

        loss = F.smooth_l1_loss(q_target, q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.count % 50 == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            self.save()

        self.count += 1

        return loss.item()


if __name__ == '__main__':
    s1 = torch.rand(1500, 1000, 3).cuda().unsqueeze(0)
    s2 = torch.rand(1500, 1000, 3).cuda().unsqueeze(0)
    s3 = torch.zeros(1500, 1000, 3).cuda().unsqueeze(0)
    s4 = torch.zeros(1500, 1000, 3).cuda().unsqueeze(0)

    buffer_ = DragonBuffer()
    da = DragonAgent(buffer_)
    action_test = da.sample_action(s1)
    action_test_2 = da.sample_action(s3)
    buffer_.push((s1, action_test, 1, s2))
    buffer_.push((s3, action_test_2, 0, s4))

    da.update()

    da.load()

    # b = Bernoulli(0.3)
    # t = [b.sample().item() for i in range(1000)]
    # from matplotlib import pyplot as plt
    #
    # plt.hist(t)
    # plt.show()
