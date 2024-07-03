import random
import time

import cv2
import pygame
import torch
from pygame.color import THECOLORS as COLORS

from ChromeDragon.tools import rect_cover
import numpy as np

from DQN.constants import *


class DragonEnvironment:

    def __init__(self):
        self.is_dead, self.is_jump = None, None
        self.jump_times = None
        self.dragon_x, self.dragon_y, self.dragon_v = None, None, None
        self.raven_list, self.cactus_list = None, None
        self.screen = None
        self.reward = None
        self._reset_param()
        self._game_init()

    def _check_dead(self):
        dragon_rect = (self.dragon_x, self.dragon_y, DRAGON_WIDTH, DRAGON_HEIGHT)
        if dragon_rect[1] + dragon_rect[3] > DEAD_BOTTOM:
            return True
        for x in self.cactus_list:
            down_rect = (x[0], CACTUS_Y, x[1] * CACTUS_WIDTH, CACTUS_HEIGHT)
            if rect_cover(dragon_rect, down_rect, up=False):
                return True
        for x in self.raven_list:
            down_rect = (x[0], RAVEN_BOTTOM - x[1] * RAVEN_Y_BLANK, RAVEN_WIDTH, RAVEN_HEIGHT)
            if rect_cover(dragon_rect, down_rect, up=False):
                return True
        return False

    def _check_go_through(self):
        reward = 0
        for x in self.cactus_list:
            if self.dragon_y + DRAGON_HEIGHT < CACTUS_Y and (
                    DRAGON_WIDTH // 2 + self.dragon_x) == x[0] + CACTUS_WIDTH * x[1] // 2:
                reward += 1
        for x in self.raven_list:
            if self.dragon_y + DRAGON_HEIGHT < RAVEN_BOTTOM - x[1] * RAVEN_Y_BLANK and (
                    DRAGON_WIDTH // 2 + self.dragon_x) == x[0] + RAVEN_WIDTH // 2:
                reward += 1
                self.reward = reward

    def _jump_data_update(self):
        if self.jump_times > 0:
            self.jump_times -= 1
            self.is_jump = True

    def step(self, action):
        assert isinstance(action, int)
        assert action in [0, 1]
        self.reward = 1
        if action == 1:
            self._jump_data_update()
        count = 0
        while count < STATE_STEPS:
            self._data_update_once()
            self._draw_once()
            count += 1

        self.is_dead = self._check_dead()
        if self.is_dead:
            self.reward = -3

        return self._get_state(), self.reward, self.is_dead, False

    def _reset_param(self):
        self.is_dead = False
        self.is_jump = False
        self.jump_times = 2
        self.dragon_x = DRAGON_X_INIT
        self.dragon_y = DRAGON_Y_INIT
        self.dragon_v = 0
        self.raven_list = [[700 // DISTANCE_RATE, 1], [1700 // DISTANCE_RATE, 2]]
        self.cactus_list = [[500 // DISTANCE_RATE, 2], [1000 // DISTANCE_RATE, 1], [1500 // DISTANCE_RATE, 1],
                            [2000 // DISTANCE_RATE, 2]]

    def _get_state(self):
        raw_state = pygame.surfarray.array3d(pygame.display.get_surface())
        raw_state = cv2.resize(raw_state, (SIZE[0] // 4, SIZE[1] // 4), interpolation=cv2.INTER_AREA)
        raw_state = (raw_state - raw_state.min()) / (raw_state.max() - raw_state.min())
        jump_times = torch.ones((*raw_state.shape[:2], 1)) * self.jump_times
        concatenated_state = np.concatenate((raw_state, jump_times), axis=-1)
        state = torch.from_numpy(concatenated_state[None]).float()
        assert len(state.shape) == 4
        if state.shape[3] == 4:
            state = state.permute(0, 3, 1, 2)
        return state

    def _game_init(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SIZE)

    def _data_update_once(self):
        number_of_go_through = 0
        if not self.is_dead:
            for x in self.cactus_list:
                if x[0] - SPEED > LEFT_BOUND:
                    cactus_center = x[0] + CACTUS_WIDTH * x[1] // 2
                    if 0 >= self.dragon_x + DRAGON_WIDTH // 2 - cactus_center >= - SPEED:
                        number_of_go_through += 1
                    x[0], x[1] = x[0] - SPEED, x[1]
                else:
                    x[0], x[1] = RANDOM_X, random.choice([1, 2])
            for x in self.raven_list:
                if x[0] - SPEED > LEFT_BOUND:
                    raven_center = x[0] + RAVEN_WIDTH // 2
                    if 0 >= self.dragon_x + DRAGON_WIDTH // 2 - raven_center >= - SPEED:
                        number_of_go_through += 1
                    x[0], x[1] = x[0] - SPEED, x[1]
                else:
                    x[0], x[1] = RANDOM_X, random.choice([1, 2])

            if not self.is_jump:
                self.dragon_v += G * FRAME
            else:
                self.dragon_v = JUMP_V
                self.is_jump = False
            self.dragon_y = self.dragon_y + FRAME * self.dragon_v if self.dragon_y + FRAME * self.dragon_v < (
                    FLOOR_Y - DRAGON_HEIGHT) else (
                    FLOOR_Y - DRAGON_HEIGHT)
            if self.dragon_y >= (FLOOR_Y - DRAGON_HEIGHT):
                self.jump_times = 2
        return number_of_go_through

    def reset(self):
        self._reset_param()
        self._data_update_once()
        self._draw_once()
        return self._get_state(), False, False

    def _draw_background(self):
        # white background
        self.screen.fill(COLORS['lightblue'])
        black_ = (-100 // DISTANCE_RATE, 902 // DISTANCE_RATE, 3000 // DISTANCE_RATE, 200 // DISTANCE_RATE)
        darkgray_ = (-100 // DISTANCE_RATE, 802 // DISTANCE_RATE, 3000 // DISTANCE_RATE, 100 // DISTANCE_RATE)
        pygame.draw.rect(self.screen, COLORS['black'], black_, 5)
        pygame.draw.rect(self.screen, COLORS['darkgray'], darkgray_, 0)

    def _draw_cactus(self):
        for x in self.cactus_list:
            pygame.draw.rect(self.screen, COLORS['darkgreen'], (x[0], CACTUS_Y, CACTUS_WIDTH * x[1], CACTUS_HEIGHT), 0)

    def _draw_raven(self):
        for x in self.raven_list:
            pygame.draw.rect(self.screen, COLORS['black'],
                             (x[0], RAVEN_BOTTOM - x[1] * RAVEN_Y_BLANK, RAVEN_WIDTH, RAVEN_HEIGHT), 0)

    def _draw_dragon(self):
        pygame.draw.rect(self.screen, COLORS['darkred'], (
            self.dragon_x, self.dragon_y, DRAGON_WIDTH, DRAGON_HEIGHT), 0)

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
    temp, _, is_terminate = de.reset()

    while True:
        if is_quit() or is_terminate:
            time.sleep(10)
            print("reset ")
            _, _, is_terminate = de.reset()
            continue
        temp, _, is_terminate, _ = de.step(1)
        i += 1

        pygame.time.delay(10)
