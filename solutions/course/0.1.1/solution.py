"""Reference answers for 0.1.1 Meet your car.  Labs never import this; the grader never reads it."""
import numpy as np

__all__ = ["lane_error_stats", "steps_off_lane"]


def lane_error_stats(lat_err):
    err = np.asarray(lat_err, dtype=float)
    return {
        "mean_abs": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
        "rms": float(np.sqrt(np.mean(err**2))),
    }


def steps_off_lane(lat_err, threshold=0.9):
    return int(np.sum(np.abs(lat_err) > threshold))
