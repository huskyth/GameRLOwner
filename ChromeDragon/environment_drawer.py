import numpy as np
import cv2
from pygame.color import THECOLORS as COLORS

from ChromeDragon.environment import DragonEnvironment

i = np.zeros((*DragonEnvironment.SIZE, 4)).transpose((1, 0, 2))
i[:, :, :] = np.array((0, 255, 0, 255))
cv2.imshow("a", i)
cv2.waitKey(0)
print(i)
