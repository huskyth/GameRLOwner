import torch

from DQN.model import DragonModel


class DragonAgent:
    def __init__(self):
        self.q_net = DragonModel().cuda()
        self.target_q_net = DragonModel().cuda()

        self.target_q_net.load_state_dict(self.q_net.state_dict())

    @torch.no_grad()
    def get_action(self, state):
        action = self.q_net(state)
        action = torch.argmax(action, dim=-1)
        return action.detach().cpu().item()


if __name__ == '__main__':
    da = DragonAgent()
    init = torch.zeros(1500, 1000, 3).cuda().unsqueeze(0)
    action = da.get_action(init)
    print(action)
