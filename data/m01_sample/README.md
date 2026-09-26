# Module 01 sample frames

Synthetic calibrated road renders (not real photographs) for the cameras & IPM lab.

## Regenerate

```bash
python data/m01_sample/generate_frames.py
```

## Files

- `front.png`, `left.png`, `right.png` — 320×180 BGR renders
- `calib.json` — intrinsics, extrinsics, pitch/yaw/roll, ego frame note
- `points.json` — ground points with projected pixels per camera

## License

Synthetic course data, CC0.
