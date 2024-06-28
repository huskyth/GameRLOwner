def rgba2bgra(rgba):
    temp = list(rgba)
    temp[0], temp[2] = temp[2], temp[0]
    return tuple(temp)
