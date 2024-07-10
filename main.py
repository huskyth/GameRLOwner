import torch

from ChromeDragon.environment import DragonEnvironment
from ChromeDragon.tensor_board_tool import MySummary
from DQN.agent import DragonAgent
from DQN.buffer import DragonBuffer
from collections import Counter

import gc

EPOCH = 100000000

IS_TEST = False
my_summary = MySummary(use_wandb=not IS_TEST)


@torch.no_grad()
def my_probability(state, model):
    return model(state)


def main(is_test):
    d_environment = DragonEnvironment()
    d_buffer = DragonBuffer()
    d_agent = DragonAgent(d_buffer, my_summary)
    # d_agent.load()
    mean_return = 0
    loss = 0
    log_rate = 10
    for epo in range(EPOCH):
        s, is_terminate = d_environment.reset()
        while not is_terminate:
            pre_s = s
            action = d_agent.sample_action(pre_s, is_test)
            s, reward, is_terminate = d_environment.step(action)
            d_buffer.push((pre_s, action, reward, s, is_terminate))
            mean_return += reward

            if len(d_buffer.buffer) > 2000:
                loss += d_agent.update()
                d_agent.save()
                gc.collect()

        if not is_test and (epo + 1) % log_rate == 0:
            my_summary.add_float(x=0, y=mean_return / log_rate, title="Smooth Return")
            my_summary.add_float(x=0, y=loss / log_rate, title="Smooth Loss")
            print(f"epoch {epo}, loss {loss / log_rate} return  {mean_return / log_rate}")
            mean_return = 0
            loss = 0


if __name__ == '__main__':
    main(IS_TEST)
