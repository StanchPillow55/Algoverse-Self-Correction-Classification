def iscube(a):
    if a < 0:
        a = -a
    cube_root = round(a ** (1/3))
    return cube_root ** 3 == a