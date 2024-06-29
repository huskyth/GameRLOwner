import random
import time

import cv2
import pygame
from pygame.color import THECOLORS as COLORS

from ChromeDragon.tools import rect_cover


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
        self._reset_param()

    def _check_dead(self):
        dragon_rect = (self.dragon_x, self.dragon_y, DragonEnvironment.DRAGON_WIDTH, DragonEnvironment.DRAGON_HEIGHT)
        if dragon_rect[1] + dragon_rect[3] > 900:
            return True
        for x in self.cactus_list:
            down_rect = (x[0], 730, x[1] * 40, 100)
            if rect_cover(dragon_rect, down_rect, up=False):
                return True
        for x in self.raven_list:
            down_rect = (x[0], 800 - x[1] * 50, 100, 20)
            if rect_cover(dragon_rect, down_rect, up=False):
                return True
        return False

    def _jump_data_update(self):
        if self.jump_times > 0:
            self.jump_times -= 1
            self.is_jump = True

    def step(self, action):
        assert action in [False, True]
        if action is True:
            self._jump_data_update()

        self._data_update_once()
        self._draw_once()
        self.is_dead = self._check_dead()
        return self._get_state(), self.is_dead

    def _reset_param(self):
        self.is_dead = False
        self.is_jump = False
        self.jump_times = 2
        self.dragon_x = 200
        self.dragon_y = 760
        self.dragon_v = 0
        self.raven_list = [[700, 1], [1700, 2]]
        self.cactus_list = [[500, 2], [1000, 1], [1500, 1], [2000, 2]]

    def _get_state(self):
        return pygame.surfarray.array3d(pygame.display.get_surface()).transpose((1, 0, 2))

    def _game_init(self):
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
        self._reset_param()
        self._game_init()
        self._draw_once()
        return self._get_state(), False

    def _draw_background(self):
        # white background
        self.screen.fill(COLORS['lightblue'])
        pygame.draw.rect(self.screen, COLORS['black'], (-100, 902, 3000, 200), 5)
        pygame.draw.rect(self.screen, COLORS['darkgray'], (-100, 802, 3000, 100), 0)

    def _draw_cactus(self):
        for x in self.cactus_list:
            pygame.draw.rect(self.screen, COLORS['darkgreen'], (x[0], 730, 40 * x[1], 100), 0)

    def _draw_raven(self):
        for x in self.raven_list:
            pygame.draw.rect(self.screen, COLORS['black'], (x[0], 800 - x[1] * 50, 100, 20), 0)

    def _draw_dragon(self):
        pygame.draw.rect(self.screen, COLORS['darkred'], (
            self.dragon_x, self.dragon_y, DragonEnvironment.DRAGON_WIDTH, DragonEnvironment.DRAGON_HEIGHT), 0)

    def _quit(self):
        pygame.quit()

    def _draw_once(self):
        # background
        self._draw_background()
        # anamy
        self._draw_cactus()
        self._draw_raven()
        # choose item
        self._draw_dragon()
        # flip
        pygame.display.flip()


def is_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
    return False


if __name__ == '__main__':
    de = DragonEnvironment()
    i = 0
    temp, is_terminate = de.reset()

    while True:
        if is_quit() or is_terminate:
            time.sleep(10)
            print("reset ")
            _, is_terminate = de.reset()
            continue
        temp, is_terminate = de.step(False)
        i += 1

        pygame.time.delay(10)
