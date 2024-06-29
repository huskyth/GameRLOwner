from ChromeDragon.environment import DragonEnvironment
from DQN.agent import DragonAgent
from DQN.buffer import DragonBuffer
from DQN.model import DragonModel

EPOCH = 100
MAX_STEP = 100


def epoch_once(d_environment, d_agent, d_buffer):
    s, is_terminate = d_environment.reset()
    step = 0
    return_value = 0
    while step <= MAX_STEP and not is_terminate:
        step += 1
        if is_terminate:
            break
        pre_s = s
        action = d_agent.sample_action(pre_s)
        s, reward, is_terminate = d_environment.step(action)
        d_buffer.push((pre_s, action, reward, s))
        return_value += reward
    d_agent.update()
    d_agent.save()
    print(f"return value: {return_value}")


if __name__ == '__main__':

    d_environment_ = DragonEnvironment()
    d_buffer_ = DragonBuffer()
    d_agent_ = DragonAgent(d_buffer_)

    for epo in range(EPOCH):
        epoch_once(d_environment_, d_agent_, d_buffer_)
