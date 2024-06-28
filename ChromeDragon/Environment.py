class DragonEnvironment:
    FRAME = 0.02
    SPEED = 5
    DRAGON_WIDTH = 30
    DRAGON_HEIGHT = 50
    JUMP_V = -300
    FLOOR_Y = 800
    G = 9.8 * 30  # g
    SIZE = [1500, 1000]

    def __init__(self):
        self.is_dead = False
        self.is_jump = False
        self.jump_times = 2
        self.dragon_x = 200
        self.dragon_y = 760
        self.raven_list = [[700, 1], [1700, 2]]
        self.cactus_list = [[500, 2], [1000, 1], [1500, 1], [2000, 2]]

    def step(self, action):
        pass

    def reset(self):
        pass
