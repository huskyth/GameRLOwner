import torch

from ChromeDragon.environment import DragonEnvironment
from ChromeDragon.tensor_board_tool import MySummary
from DQN.agent import DragonAgent
from DQN.buffer import DragonBuffer
from collections import Counter

EPOCH = 10000

IS_TEST = False
my_summary = MySummary(use_wandb=not IS_TEST)

global_return = None


@torch.no_grad()
def my_probability(state, model):
    return model(state)


def epoch_once(d_environment, d_agent, d_buffer, is_test):
    global global_return
    s, truncated, is_terminate = d_environment.reset()
    step = 0
    return_value = 0
    action_list = ""
    reward_list = ""
    c = Counter()
    p = my_probability(s)
    my_summary.add_float(x=0, y=p[0, 0].item(), title="P0 Value")
    my_summary.add_float(x=0, y=p[0, 1], title="P1 Step")
    while not is_terminate and not truncated:
        step += 1
        pre_s = s
        action = d_agent.sample_action(pre_s, is_test)
        s, reward, truncated, is_terminate = d_environment.step(action)
        d_buffer.push((pre_s, action, reward, s, is_terminate))

        return_value += reward
        action_list += str(action) + ","
        reward_list += str(reward) + ","
        c.update(str(action))
    if global_return is None:
        global_return = return_value
    if not is_test:
        global_return = 0.9 * global_return + 0.1 * return_value
        my_summary.add_float(x=0, y=return_value, title="Return Value")
        my_summary.add_float(x=0, y=step, title="Total Step")
        my_summary.add_float(x=0, y=global_return, title="Smooth Reward")
        d_agent.update()
        d_agent.save()
        print(f"return value: {return_value}")
        print(f"reward_list: {reward_list}")
        print(action_list)
        print(c)


if __name__ == '__main__':

    d_environment_ = DragonEnvironment()
    d_buffer_ = DragonBuffer()
    d_agent_ = DragonAgent(d_buffer_, my_summary)
    # d_agent_.load()

    for epo in range(EPOCH):
        epoch_once(d_environment_, d_agent_, d_buffer_, IS_TEST)
        if epo % 5 == 0 and not IS_TEST:
            d_agent_.save()
