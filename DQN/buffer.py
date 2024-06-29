import random
from collections import deque


class DragonBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=100)

    def sample(self, batch):
        batch = min(batch, len(self.buffer))
        return zip(*random.sample(self.buffer, batch))

    def push(self, transition):
        self.buffer.append(transition)


if __name__ == '__main__':
    df = DragonBuffer()
    for i in range(200):
        s, a, r, s_ = str(i), "a", 1, str(i + 1)
        df.push((s, a, r, s_))
    s, a, r, s_ = df.sample(20)
    print(s)
    print(a)
    print(r)
    print(s_)
