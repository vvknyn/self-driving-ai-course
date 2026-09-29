"""Reference answer for 0.1.2 Images are arrays.  Labs never import this; the grader never reads it."""
import numpy as np

__all__ = ["paint_masks"]

YELLOW, WHITE = np.array([230.0, 200.0, 40.0]), np.array([240.0, 240.0, 240.0])
CLOSE = 60.0  # RGB distance that still counts as paint; noise moves a pixel far less than the gap to other colours


def paint_masks(frame):
    pixels = frame.astype(float)
    return tuple(np.linalg.norm(pixels - colour, axis=-1) < CLOSE for colour in (YELLOW, WHITE))
