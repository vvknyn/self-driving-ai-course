# %% [markdown]
# # Mini unit
#
# A fixture lab: markdown cells lose their leading `# `.

# %%
%pip install -q "git+https://github.com/vvknyn/self-driving-ai-course@main"

# %% tags=["setup"]
import numpy as np
from zero2fsd.grade import check

# %% tags=["exercise:0.1.1.a"]
def _mean_abs(lat_err):
    raise NotImplementedError


# %% tags=["exercise:0.1.1.a"]
def lane_error_stats(lat_err):
    return _mean_abs(lat_err)


# %%
check("0.1.1.a", lane_error_stats)

# %% tags=["exercise:0.1.1.b"]
def steps_off_lane(lat_err, threshold=0.9):
    raise NotImplementedError


# %%
check("0.1.1.b", steps_off_lane)
