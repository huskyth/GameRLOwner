import random

import numpy
import torch

from DQN.buffer import DragonBuffer
from DQN.model import DragonModel
import torch.nn.functional as F


class DragonAgent:
    def __init__(self, buffer):
        self.q_net = DragonModel().cuda()
        for key, model in self.q_net.named_parameters():
            model.data.normal_(-1, 1)

        self.target_q_net = DragonModel().cuda()

        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.buffer = buffer
        self.gamma = 0.99

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-2)
        self.action_probability = 0.57
        self.count = 0

    @torch.no_grad()
    def _get_action(self, state):
        if isinstance(state, numpy.ndarray):
            state = torch.from_numpy(state).float()
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        state = state.cuda()
        action = self.q_net(state)
        action = torch.argmax(action, dim=-1)
        return action.detach().cpu().item()

    def sample_action(self, state):
        if random.uniform(0, 1) > self.action_probability:
            return torch.randint(low=0, high=2, size=(1,)).item()
        else:
            return self._get_action(state)

    def save(self):
        saved = {"model_state_dict": self.q_net.state_dict(), "optimizer_state_dict": self.optimizer.state_dict()}
        torch.save(saved, "best.pt")

    def load(self):
        saved = torch.load("best.pt")
        self.q_net.load_state_dict(saved["model_state_dict"])
        self.optimizer.load_state_dict(saved["optimizer_state_dict"])
        print(self.q_net.state_dict()['feature.2.weight'][0][0])

    def update(self):
        state, action, reward, next_state, done = self.buffer.sample()
        state = [torch.from_numpy(x).float() for x in state]
        state = torch.cat(state).cuda()
        action = torch.tensor(action, dtype=torch.int64).cuda().unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float32).cuda()
        next_state = [torch.from_numpy(x).float() for x in next_state]
        next_state = torch.cat(next_state).cuda()
        done = torch.tensor(done).int().cuda()

        q_value = self.q_net(state).gather(1, action)

        next_action = torch.argmax(self.q_net(next_state), dim=-1).unsqueeze(1)
        q_target = (1 - done) * self.target_q_net(next_state).gather(1, next_action) + reward

        loss = F.mse_loss(q_target, q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print(f"current update step {self.count}, loss = {loss}")
        if self.count % 20 == 0:
            print(f"load from q net {self.count}")
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1


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
