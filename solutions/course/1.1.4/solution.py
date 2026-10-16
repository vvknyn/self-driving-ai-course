"""Reference answer for 1.1.4 Poses you can chain.  Labs never import this; the grader never reads it."""
import numpy as np

__all__ = ["relative_pose", "dead_reckon"]


def _inverse(T):
    """Closed-form inverse of a pose matrix of any size: [[R^T, -R^T t], [0, 1]]."""
    k = len(T) - 1
    inverse = np.eye(k + 1)
    inverse[:k, :k] = T[:k, :k].T
    inverse[:k, k] = -T[:k, :k].T @ T[:k, k]
    return inverse


def relative_pose(T_world_a, T_world_b):
    """Pose b as seen from pose a: T_a<-b = T_a<-world . T_world<-b."""
    return _inverse(T_world_a) @ T_world_b


def dead_reckon(T_world_start, steps):
    """The start, then each pose built by putting the next step on the right: T_world<-k+1 = T_world<-k . T_k<-k+1."""
    poses = [np.asarray(T_world_start, dtype=float)]
    for step in steps:
        poses.append(poses[-1] @ step)
    return np.stack(poses)
