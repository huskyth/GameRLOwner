import os.path
import random
import time

import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import cv2

from ChromeDragon.environment import DragonEnvironment
from wrappers import wrap_mario


def handle(state):
    state = np.array(state) * 255.0
    return state


def dir_mk(step):
    path = str(step)
    if not os.path.exists(path):
        os.mkdir(path)
    state_path = path + os.sep + "state"
    if not os.path.exists(state_path):
        os.mkdir(state_path)
    state_prime_path = path + os.sep + "state_prime"
    if not os.path.exists(state_prime_path):
        os.mkdir(state_prime_path)
    return path, state_path, state_prime_path


def _write_state(state, path):
    n = state.shape[2]
    for i in range(n):
        temp = state[:, :, i]
        cv2.imwrite(f"{path}{os.sep}index_{i}.png", temp)


def write(state, step, action, state_prime, reward, info):
    path, state_path, state_prime_path = dir_mk(step)
    state = handle(state)
    state_prime = handle(state_prime)
    _write_state(state, state_path)
    _write_state(state_prime, state_prime_path)
    with open(f"{path}" + os.sep + "action.txt", 'w') as f:
        f.write(f"action is {action}")
    with open(f"{path}" + os.sep + "reward.txt", 'w') as f:
        f.write(f"reward is {reward}")
    with open(f"{path}" + os.sep + "info.txt", 'w') as f:
        f.write(f"info is {info}")


def super_query():
    # TODO://gym==0.23.0
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = wrap_mario(env)
    done = True
    state = env.reset()
    pre_state = state
    write(state=state, step=0, action=None, state_prime=state, reward=None, info=None)
    for i in range(1, 5000):
        action = random.randint(0, len(SIMPLE_MOVEMENT) - 1)
        state, reward, done, info = env.step(4)
        time.sleep(0.01)
        write(state=pre_state, step=i, action=f"{action}, {SIMPLE_MOVEMENT[action]}", state_prime=state, reward=reward,
              info=info)
        pre_state = state
        env.render()
        if i == 16:
            break

    time.sleep(1)

    env.close()


def dragon_query():
    env = DragonEnvironment()
    done = True
    state = env.reset()
    pre_state = state
    write(state=state, step=0, action=None, state_prime=state, reward=None, info=None)
    for i in range(1, 5000):
        action = random.randint(0, 1)
        state, reward, done, info = env.step(action)
        time.sleep(0.01)
        write(state=pre_state, step=i, action=f"{action}, {SIMPLE_MOVEMENT[action]}", state_prime=state, reward=reward,
              info=info)
        pre_state = state
        env.render()
        if i == 16:
            break

    time.sleep(1)

    env.close()


if __name__ == '__main__':
    dragon_query()
