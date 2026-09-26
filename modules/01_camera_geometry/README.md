# Module 01 — Cameras & IPM

## Session card

**Today's win:** Warp one camera to the ground plane, then measure how pitch error grows with range.

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

- Build intrinsics and ground homography from first principles
- Project ego ground points to pixels and warp a front camera to BEV
- Stitch three cameras into one metric canvas
- Quantify pitch sensitivity on forward range

Notebook: [`notebooks/01_cameras_and_ipm.ipynb`](../../notebooks/01_cameras_and_ipm.ipynb)

## Why

- Pinhole geometry: similar triangles — depth divides lateral image position.
- A flat road is a plane; homography links ground (X, Y) to pixels when Z = 0.
- Wrong extrinsics → wrong meters; a small pitch bias grows with range.

## Derive

- x = f · X / Z from similar triangles.
- Drop r₃ on Z = 0: H = K [r₁ r₂ t].
- Pitch error shifts the horizon and smears BEV lane alignment.

## Commands

```bash
pytest modules/01_camera_geometry
python modules/01_camera_geometry/calibrate_rig.py
python modules/01_camera_geometry/break_it_fix_it.py
```

Sample data under `data/m01_sample/` is synthetic (CC0). Student fills: `build_intrinsic_matrix`, `build_ground_homography`, and from-scratch `pitch_shift_meters`. Reference implementations live in `solutions/01_camera_geometry/`.
