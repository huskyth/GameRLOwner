import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from wrappers import wrap_mario

import torch
import numpy as np
from ChromeDragon.tensor_board_tool import MySummary
from sm_agent import DragonAgent
from DQN.buffer import DragonBuffer
from collections import Counter

EPOCH = 100000000

IS_TEST = False
my_summary = MySummary(use_wandb=not IS_TEST)

IS_RENDER = False


@torch.no_grad()
def my_probability(state, model):
    return model(state)


def train(is_test):
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    d_environment = wrap_mario(env)

    d_buffer = DragonBuffer()
    d_agent = DragonAgent(d_buffer, my_summary)
    # d_agent_.load()
    log_rate = 10
    return_value = 0
    loss = 0
    for epo in range(EPOCH):
        is_terminate = False
        s = d_environment.reset()
        s = torch.from_numpy(np.array(s).transpose((2, 0, 1))[None])

        c = Counter()
        while not is_terminate:
            pre_s = s
            action = d_agent.sample_action(pre_s, is_test)
            s, reward, is_terminate, _ = d_environment.step(action)
            s = torch.from_numpy(np.array(s).transpose((2, 0, 1))[None])
            return_value += reward
            reward = np.sign(reward) * (np.sqrt(abs(reward) + 1) - 1) + 0.001 * reward
            d_buffer.push((pre_s, action, reward, s, is_terminate))
            c.update(str(action))
            if IS_RENDER:
                env.render()

        if len(d_buffer.buffer) > 2000:
            print("start training ...")
            loss += d_agent.update()
            d_agent.save()
        if not is_test and (epo + 1) % log_rate == 0:
            my_summary.add_float(x=0, y=return_value / log_rate, title="Mean Reward")
            my_summary.add_float(x=0, y=loss / log_rate, title="Loss")
            print(f"epoch {epo} loss: {loss / log_rate}, mean reward: {return_value / log_rate}")
            loss = 0

            return_value = 0


if __name__ == '__main__':
    train(IS_TEST)
