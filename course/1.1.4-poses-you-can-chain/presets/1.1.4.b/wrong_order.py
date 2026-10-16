import numpy as np


def dead_reckon(T_world_start, steps):
    # The step goes on the left: S @ T. The end of the lap still looks right; the middle does not.
    poses = [np.asarray(T_world_start, dtype=float)]
    for step in steps:
        poses.append(step @ poses[-1])
    return np.stack(poses)
