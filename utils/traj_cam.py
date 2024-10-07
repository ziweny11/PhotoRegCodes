import numpy as np


#straight line
def straight_right_train(T0, T1, freq):
    stamp = []
    for i in range(freq):
        t = i / (freq - 1)
        T = (1 - t) * T0 + t * T1
        stamp.append(T)
    return stamp





#2d ellipse
def ellipse_train(b, a, center, freq):
    x0, z0, y0 = center[0], center[1], center[2]
    angles = np.linspace(0, 2 * np.pi, freq)
    x_coords = x0 + a * np.cos(angles)
    y_coords = y0 + b * np.sin(angles)
    z_coords = np.full_like(x_coords, z0)
    points = [np.array([x,z,y]) for x, z, y in zip(x_coords, z_coords, y_coords)]
    return points

