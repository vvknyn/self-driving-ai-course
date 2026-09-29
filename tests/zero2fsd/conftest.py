import os

os.environ.setdefault("MPLBACKEND", "Agg")  # before any matplotlib import: tests never open a window
