def world_to_ego(pose, p_world):
    x, y, theta = pose
    return np.asarray(p_world, dtype=float) - np.array([x, y])
