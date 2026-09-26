# Module 00 — Driving ML Gym

## Session card

**Today's win:** See pedestrian recall, not just accuracy, on the checked-in crops.

**Time:** 25 / 55 / 90 minutes

A day counts when you export an artifact or pass the tests for a fill you wrote. Opening the notebook does not. Set pause_week to true if you need a week off; the count stays where it is.

**Your stack so far**

- [ ] m00 — Driving ML Gym
- [ ] m01 — Cameras & IPM
- [ ] m02 — HydraNet
- [ ] m03 — BEV transform
- [ ] m04 — Occupancy
- [ ] m05 — Vector tracking
- [ ] m06 — Planning
- [ ] m07 — Control
- [ ] m08 — Capstone
- [ ] m09 — System architecture

## Practice

- Explain `(B, C, H, W)` in one sentence each
- Train a tiny CNN on checked-in driving crops
- Compare cross-entropy vs focal loss on minority recall
- Export metrics and inspect an error gallery
- Run the break-it demo on accuracy vs minority recall

Notebook: [`notebooks/00_driving_ml_gym.ipynb`](../../notebooks/00_driving_ml_gym.ipynb)

## Why

- Class imbalance is normal in driving data — rare objects still matter for safety.
- The loss defines what “good” means; accuracy alone can hide failure on pedestrians.
- You cannot improve what you do not inspect — error galleries beat one headline number.
- Tesla’s data engine is one example of teams that scale labeled crops and hard-example mining.

## Derive

- Softmax → cross-entropy: minimize negative log probability of the true class.
- Focal modulating factor `(1 - p_t)^gamma` down-weights easy examples.
- Optional autograd appendix: [`modules/00_nn_scratch`](../00_nn_scratch/) (not the default Week 1 path).

## Commands

```bash
pytest modules/00_ml_gym
python modules/00_ml_gym/train.py
python modules/00_ml_gym/break_it_fix_it.py
```

Student fills in `losses.py` (`focal_loss`), `metrics.py` (`minority_recall`), and `error_gallery.py` (`build_error_gallery`) raise `NotImplementedError` until you implement them. `tests/test_assignment_solutions.py` loads reference code from `solutions/00_ml_gym/`.
