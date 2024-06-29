import torch

from DQN.buffer import DragonBuffer
from DQN.model import DragonModel
import torch.nn.functional as F


class DragonAgent:
    def __init__(self, buffer):
        self.q_net = DragonModel().cuda()
        for key, model in self.q_net.named_parameters():
            model.data.normal_(-0.8, 0.8)

        self.target_q_net = DragonModel().cuda()

        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.buffer = buffer
        self.gamma = 0.99

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-2)

    @torch.no_grad()
    def get_action(self, state):
        action = self.q_net(state)
        action = torch.argmax(action, dim=-1)
        return action.detach().cpu().item()

    def update(self):
        state, action, reward, next_state = self.buffer.sample()
        state = torch.cat(state).cuda()
        action = torch.tensor(action, dtype=torch.int64).cuda().unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float32).cuda()
        next_state = torch.cat(next_state).cuda()

        q_value = self.q_net(state).gather(1, action)

        next_action = torch.argmax(self.q_net(next_state), dim=-1).unsqueeze(1)
        q_target = self.target_q_net(next_state).gather(1, next_action) + reward

        loss = F.mse_loss(q_target, q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    s1 = torch.rand(1500, 1000, 3).cuda().unsqueeze(0)
    s2 = torch.rand(1500, 1000, 3).cuda().unsqueeze(0)
    s3 = torch.zeros(1500, 1000, 3).cuda().unsqueeze(0)
    s4 = torch.zeros(1500, 1000, 3).cuda().unsqueeze(0)

    buffer_ = DragonBuffer()
    da = DragonAgent(buffer_)
    action_test = da.get_action(s1)
    action_test_2 = da.get_action(s3)
    buffer_.push((s1, action_test, 1, s2))
    buffer_.push((s3, action_test_2, 0, s4))

    da.update()
