def rgba2bgra(rgba):
    temp = list(rgba)
    temp[0], temp[2] = temp[2], temp[0]
    return tuple(temp)


def rect_cover(rect1, rect2, up=True):
    # dragon
    left_up1 = (rect1[0], rect1[1])
    left_down1 = (rect1[0], left_up1[1] + rect1[3])
    right_up1 = (left_up1[0] + rect1[2], rect1[1])
    right_down1 = (left_up1[0] + rect1[2], left_up1[1] + rect1[3])
    # tunnel
    left_up2 = (rect2[0], rect2[1])
    left_down2 = (rect2[0], left_up2[1] + rect2[3])
    right_up2 = (left_up2[0] + rect2[2], rect2[1])
    right_down2 = (left_up2[0] + rect2[2], left_up2[1] + rect2[3])
    # check
    if left_up2[0] <= right_up1[0] <= right_up2[0]:  # x,肯定是右侧线接触，因此判断dragon的right即可
        if up and (left_up2[1] <= right_up1[1] <= left_down2[1]):
            return True
        elif (not up) and (left_up2[1] <= right_down1[1] <= left_down2[1]):
            return True
    return False
