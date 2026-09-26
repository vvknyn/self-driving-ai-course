# Zero2FSD — Module 00 — Driving ML Gym

## Session card

Zero2FSD · Week 1

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

Follow the notebook beats in order:

- **Images as numbers** — `(B, C, H, W)` and PNG → tensor
- **Classification as scores** — softmax and cross-entropy
- **Neuron and layer** — ReLU and why depth needs a bend
- **The train loop** — forward, loss, backward, step
- **Why convolutions** — `DrivingClassifier` stem / stage2 / pool / fc
- **Metrics that lie** — accuracy vs pedestrian recall
- **The loss is the objective** — focal loss fill
- **Inspect and ship** — minority recall fill, error gallery from scratch

Notebook: [`notebooks/00_driving_ml_gym.ipynb`](../../notebooks/00_driving_ml_gym.ipynb)

## Why

- Class imbalance is normal in driving data — rare objects still matter for safety.
- The loss defines what “good” means; accuracy alone can hide failure on pedestrians.
- You cannot improve what you do not inspect — error galleries beat one headline number.
- Tesla’s data engine is one example of teams that scale labeled crops and hard-example mining.

## Derive

- Softmax → cross-entropy (beat 2); focal modulating factor `(1 - p_t)^gamma` (beat 7).
- Optional autograd appendix: [`modules/00_nn_scratch`](../00_nn_scratch/) and `notebooks/00_neural_networks_and_autograd.ipynb` (not the default Week 1 path).

## Commands

```bash
pytest modules/00_ml_gym
python modules/00_ml_gym/train.py
python modules/00_ml_gym/break_it_fix_it.py
```

Student fills in `losses.py` (`focal_loss`), `metrics.py` (`minority_recall`), and `error_gallery.py` (`build_error_gallery`) raise `NotImplementedError` until you implement them. `tests/test_assignment_solutions.py` loads reference code from `solutions/00_ml_gym/`.
