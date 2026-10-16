import numpy as np


def dead_reckon(T_world_start, steps):
    # Add each step's parts to the running x, y and heading, as if they were all in world axes.
    x, y, theta = T_world_start[0, 2], T_world_start[1, 2], np.arctan2(T_world_start[1, 0], T_world_start[0, 0])
    poses = [T_world_start]
    for step in steps:
        x += step[0, 2]
        y += step[1, 2]
        theta += np.arctan2(step[1, 0], step[0, 0])
        c, s = np.cos(theta), np.sin(theta)
        poses.append(np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]]))
    return np.stack(poses)
