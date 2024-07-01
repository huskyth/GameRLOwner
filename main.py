import cv2

from ChromeDragon.environment import DragonEnvironment
from DQN.agent import DragonAgent
from DQN.buffer import DragonBuffer
from DQN.model import DragonModel

EPOCH = 1000
MAX_STEP = 1000
from collections import Counter


def epoch_once(d_environment, d_agent, d_buffer, epo):
    s, is_terminate = d_environment.reset()
    step = 0
    return_value = 0
    action_list = ""
    reward_list = ""
    c = Counter()
    while step <= MAX_STEP and not is_terminate:
        step += 1
        pre_s = s
        action = d_agent.sample_action(pre_s)
        s, reward, is_terminate = d_environment.step(action)
        d_buffer.push((pre_s, action, reward, s, is_terminate))

        return_value += reward
        action_list += str(action) + ","
        reward_list += str(reward) + ","
        c.update(str(action))

    d_agent.update()
    d_agent.save()
    print(f"return value: {return_value}")
    print(f"reward_list: {reward_list}")
    print(action_list)
    print(c)


if __name__ == '__main__':

    d_environment_ = DragonEnvironment()
    d_buffer_ = DragonBuffer()
    d_agent_ = DragonAgent(d_buffer_)
    d_agent_.load()

    for epo in range(EPOCH):
        epoch_once(d_environment_, d_agent_, d_buffer_, epo)
        if epo % 5 == 0:
            d_agent_.save()
