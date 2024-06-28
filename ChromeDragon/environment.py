import random

import pygame


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
        self.is_dead, self.is_jump = None, None
        self.jump_times = None
        self.dragon_x, self.dragon_y, self.dragon_v = None, None, None
        self.raven_list, self.cactus_list = None, None
        self.screen = None
        self.reset_param()

    def step(self, action):
        pass

    def reset_param(self):
        self.is_dead = False
        self.is_jump = False
        self.jump_times = 2
        self.dragon_x = 200
        self.dragon_y = 760
        self.dragon_v = 0
        self.raven_list = [[700, 1], [1700, 2]]
        self.cactus_list = [[500, 2], [1000, 1], [1500, 1], [2000, 2]]

    def game_init(self):
        pygame.init()
        self.screen = pygame.display.set_mode(DragonEnvironment.SIZE)

    def _data_update_once(self):
        if not self.is_dead:
            self.cactus_list = [
                [x[0] - DragonEnvironment.SPEED, x[1]] if x[0] - DragonEnvironment.SPEED > -200 else [2200,
                                                                                                      random.choice(
                                                                                                          [1,
                                                                                                           2])]
                for x in
                self.cactus_list]
            self.raven_list = [
                [x[0] - DragonEnvironment.SPEED, x[1]] if x[0] - DragonEnvironment.SPEED > -200 else [2200,
                                                                                                      random.choice(
                                                                                                          [1,
                                                                                                           2])]
                for x in
                self.raven_list]

            if not self.is_jump:
                self.dragon_v += DragonEnvironment.G * DragonEnvironment.FRAME
            else:
                self.dragon_v = DragonEnvironment.JUMP_V
                self.is_jump = False
            self.dragon_y = self.dragon_y + DragonEnvironment.FRAME * self.dragon_v if self.dragon_y + DragonEnvironment.FRAME * self.dragon_v < (
                    DragonEnvironment.FLOOR_Y - DragonEnvironment.DRAGON_HEIGHT) else (
                    DragonEnvironment.FLOOR_Y - DragonEnvironment.DRAGON_HEIGHT)
            if self.dragon_y >= (DragonEnvironment.FLOOR_Y - DragonEnvironment.DRAGON_HEIGHT):
                self.jump_times = 2

    def reset(self):
        self.reset_param()
        self.game_init()
