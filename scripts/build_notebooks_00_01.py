#!/usr/bin/env python3
"""Build Module 00 and 01 primary notebooks (nbformat)."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "modules") not in sys.path:
    sys.path.insert(0, str(_REPO / "modules"))
from common.progress import come_back_cue, session_card_text


def _md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source)


def _code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source)


def build_m00() -> nbf.NotebookNode:
    import importlib.util

    _spine_path = Path(__file__).resolve().parent / "m00_instruct_spine.py"
    _spec = importlib.util.spec_from_file_location("m00_instruct_spine", _spine_path)
    _mod = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_mod)
    cells = _mod.build_m00_cells(_md, _code, session_card_text, come_back_cue)
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


def build_m01() -> nbf.NotebookNode:
    cells = [
        _md(
            "# Zero2FSD — Week 2 — Cameras & IPM\n\n"
            "**1-camera IPM is required before the 3-camera stitch.** "
            "Student fills live in the module `.py` files.\n\n"
            "Synthetic calibrated frames are checked in under `data/m01_sample` (CC0), "
            "not captured from a real vehicle rig."
        ),
        _md(
            session_card_text("m01").strip()
            + "\n\nXP is not awarded for opening this notebook."
        ),
        _code(
            "import sys\nfrom pathlib import Path\n\n"
            "repo = Path.cwd()\nif not (repo / 'modules' / '01_camera_geometry').exists():\n"
            "    repo = repo.parent\n"
            "sys.path.insert(0, str(repo / 'modules'))\n"
            "from common.progress import format_stack\n\n"
            "progress_path = repo / 'artifacts' / 'progress.json'\n"
            "if progress_path.is_file():\n"
            "    import json\n"
            "    with progress_path.open() as f:\n"
            "        progress = json.load(f)\n"
            "    print(format_stack(progress))\n"
            "else:\n"
            "    print('No progress file yet. Opening this notebook awards 0 XP.')"
        ),
        _md(
            "**Principle 1 — Pinhole projection.**\n\n"
            "A pinhole maps a 3D ray to a pixel by similar triangles. "
            "A point at camera coordinates `(X, Y, Z)` hits the sensor at `x = f X / Z`, `y = f Y / Z`. "
            "Depth is in the denominator, so far objects shrink. "
            "Pixel coordinates add the principal point: `u = fx X/Z + cx`. "
            "`K` packs `fx`, `fy`, `cx`, `cy` into a 3×3 matrix. "
            "You will build `K` yourself."
        ),
        _md(
            "**Principle 2 — Homography on the road plane.**\n\n"
            "A homography is a 3×3 map between two planes. "
            "The road is the plane `Z=0` in the ego frame (X forward, Y left, Z up). "
            "A camera point is `P_cam = R P_ego + t`. Expanding `R` into columns `r1`, `r2`, `r3`, the `r3` term is multiplied by `Z`. "
            "On the road `Z` is 0, so `r3` drops out and `H = K [r1 r2 t]` maps `(X, Y, 1)` to homogeneous pixels. "
            "You will build that `H`. Using `R = I` on a pitched camera is the wrong plane-to-image map."
        ),
        _md(
            "**Principle 3 — Extrinsics set meters on the ground.**\n\n"
            "Meters on the ground come from extrinsics, not from the network. "
            "If pitch is wrong, the ray from a pixel hits `Z=0` in the wrong place. "
            "The same angular error covers more ground farther away, so the meter error grows with range. "
            "A 2° pitch bias that looks small in the image is a large longitudinal error at 40 m. "
            "You will measure that, not memorize a slogan."
        ),
        _code(
            "import sys\nimport json\nfrom pathlib import Path\nimport numpy as np\nimport cv2\n"
            "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n\n"
            "repo = Path.cwd()\nif not (repo / 'modules' / '01_camera_geometry').exists():\n"
            "    repo = repo.parent\n"
            "sys.path.insert(0, str(repo / 'modules' / '01_camera_geometry'))\n"
            "data = repo / 'data' / 'm01_sample'\n\n"
            "from camera_model import PinholeCamera, build_intrinsic_matrix\n"
            "from calibrate_rig import build_tesla_style_rig\n"
            "from ipm import IPMTransformer, build_ground_homography\n"
            "from stitch import stitch_three_cameras\n"
            "from pitch_sensitivity import pitch_shift_meters\n\n"
            "with open(data / 'calib.json') as f:\n"
            "    calib = json.load(f)\n"
            "frames = {k: cv2.imread(str(data / f'{k}.png')) for k in ('front', 'left', 'right')}\n"
            "for k, v in frames.items():\n"
            "    plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))\n"
            "    plt.title(k)\n    plt.axis('off')\n    plt.show()"
        ),
        _md(
            "**Similar triangles.** Pick focal length `f = 400` px and lateral offset `X = 10` m. "
            "Compute the image coordinate `x = f X / Z` at `Z = 20` m and at `Z = 40` m."
        ),
        _code(
            "f, X = 400.0, 10.0\n"
            "x_20 = f * X / 20.0\n"
            "x_40 = f * X / 40.0\n"
            "print('x at Z=20 m:', x_20)\n"
            "print('x at Z=40 m:', x_40)\n"
            "print('Doubling depth halves the image coordinate — the pinhole principle.')"
        ),
        _md(
            "**FILL — `build_intrinsic_matrix(fx, fy, cx, cy) -> (3, 3)`**\n\n"
            "$$K = \\begin{bmatrix} f_x & 0 & c_x \\\\ 0 & f_y & c_y \\\\ 0 & 0 & 1 \\end{bmatrix}$$\n\n"
            "Implement it in `modules/01_camera_geometry/camera_model.py`, not in this notebook."
        ),
        _code(
            "try:\n"
            "    K = build_intrinsic_matrix(calib['fx'], calib['fy'], calib['cx'], calib['cy'])\n"
            "    assert abs(K[0, 0] - calib['fx']) < 1e-6\n"
            "    assert abs(K[1, 1] - calib['fy']) < 1e-6\n"
            "    assert abs(K[0, 2] - calib['cx']) < 1e-6\n"
            "    assert abs(K[2, 2] - 1.0) < 1e-6\n"
            "    print('K ok\\n', K)\n"
            "except NotImplementedError:\n"
            "    print('STOP: implement build_intrinsic_matrix in camera_model.py')"
        ),
        _code(
            "cams = build_tesla_style_rig(calib['width'], calib['height'])\n"
            "front = cams['front']\n"
            "for pt in [(10, 0, 0), (20, 1.5, 0), (30, -1.5, 0)]:\n"
            "    pix, ok = front.project_ego_to_pixel([pt])\n"
            "    print(pt, '->', pix[0], 'valid', ok[0])"
        ),
        _md(
            "**Extrinsics: ego → camera.** `P_cam = R P_ego + T`. "
            "`R` rotates ego axes into the camera frame; `T` is the translation column (same convention as `PinholeCamera`)."
        ),
        _code(
            "print('front.R (rounded):\\n', np.round(front.R, 3))\n"
            "print('front.T.ravel():', front.T.ravel())\n"
            "print('calib pitch_deg:', calib['pitch_deg'])\n"
            "print('Pitch is not zero, so R is not a pure axis swap.')"
        ),
        _md(
            "**Derive the ground homography.** On the road plane `Z = 0`, the third column of the extrinsic block is unused:\n\n"
            "$$H = K \\,[r_1 \\; r_2 \\; t]$$\n\n"
            "`r3` is the column multiplied by `Z`, hence absent on the road."
        ),
        _md(
            "**FILL — `build_ground_homography(K, R, t)`** in `modules/01_camera_geometry/ipm.py`. "
            "Do not paste the solution here."
        ),
        _code(
            "try:\n"
            "    H = build_ground_homography(front.K, front.R, front.T)\n"
            "    pt = np.array([15.0, 0.0, 1.0])\n"
            "    uv = H @ pt\n    uv = uv[:2] / uv[2]\n"
            "    pix, _ = front.project_ego_to_pixel([[15, 0, 0]])\n"
            "    err_h = np.max(np.abs(uv - pix[0]))\n"
            "    print('H pixel', uv, 'project', pix[0], 'max abs err', err_h)\n"
            "    assert err_h < 1e-2\n"
            "    H_wrong = build_ground_homography(front.K, np.eye(3), front.T)\n"
            "    uv_w = H_wrong @ pt\n    uv_w = uv_w[:2] / uv_w[2]\n"
            "    err_wrong = np.linalg.norm(uv_w - pix[0])\n"
            "    print('R=I error px', err_wrong)\n"
            "    assert err_wrong > 5.0\n"
            "except NotImplementedError:\n"
            "    print('STOP: implement build_ground_homography in ipm.py')"
        ),
        _code(
            "ipm = IPMTransformer(front, x_range=(4, 40), y_range=(-10, 10), bev_resolution=0.1)\n"
            "bev = ipm.warp_to_bev(frames['front'])\n"
            "plt.imshow(cv2.cvtColor(bev, cv2.COLOR_BGR2RGB))\n"
            "plt.title('1-cam IPM')\nplt.axis('off')\nplt.show()"
        ),
        _code(
            "stitched = stitch_three_cameras(frames, cams)\n"
            "plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))\n"
            "plt.title('3-cam stitch')\nplt.axis('off')\nplt.show()"
        ),
        _md(
            "**FROM SCRATCH — `pitch_shift_meters(calib, delta_deg, range_m) -> float`** in "
            "`modules/01_camera_geometry/pitch_sensitivity.py`.\n\n"
            "Add `delta_deg` to the calib pitch, then return estimated forward range minus true range "
            "for the ground point `(range_m, 0, 0)`. "
            "`delta_deg = 0` should be ~0; absolute error should grow with range. "
            "Implement the function in the `.py` file — do not spell out the algorithm in this notebook."
        ),
        _code(
            "from extrinsics import create_euler_rotation, camera_position_to_translation\n\n"
            "ranges = [10, 20, 40]\n"
            "delta_deg = 2.0\n\n"
            "def _scaffold_pitch_shift(rng, delta):\n"
            "    pix, _ = front.project_ego_to_pixel([[rng, 0, 0]])\n"
            "    R2 = create_euler_rotation(calib['pitch_deg'] + delta, 0, 0)\n"
            "    T2 = camera_position_to_translation(R2, np.array(calib['cam_position_ego']))\n"
            "    cam2 = PinholeCamera(\n"
            "        'biased', front.fx, front.fy, front.cx, front.cy,\n"
            "        front.width, front.height, R2, T2,\n"
            "    )\n"
            "    est, _ = cam2.project_pixels_to_ground(pix)\n"
            "    return float(est[0, 0] - rng)\n\n"
            "try:\n"
            "    shifts = [pitch_shift_meters(calib, delta_deg, r) for r in ranges]\n"
            "    for r, s in zip(ranges, shifts):\n"
            "        print(f'range {r} m shift {s:.4f} m')\n"
            "    plt.plot(ranges, shifts, marker='o')\n"
            "    plt.xlabel('range (m)')\n"
            "    plt.ylabel('forward shift (m)')\n"
            "    plt.title('pitch_shift_meters (student fill)')\n"
            "    plt.grid(True, alpha=0.3)\n"
            "    plt.show()\n"
            "    assert abs(shifts[2]) > abs(shifts[0])\n"
            "except NotImplementedError:\n"
            "    print('STOP: implement pitch_shift_meters in pitch_sensitivity.py')\n"
            "    scaffold = [_scaffold_pitch_shift(r, delta_deg) for r in ranges]\n"
            "    for r, s in zip(ranges, scaffold):\n"
            "        print(f'[scaffold] range {r} m shift {s:.4f} m')\n"
            "    plt.plot(ranges, scaffold, marker='o', linestyle='--')\n"
            "    plt.xlabel('range (m)')\n"
            "    plt.ylabel('forward shift (m)')\n"
            "    plt.title('Reference scaffold (not student pitch_shift_meters)')\n"
            "    plt.grid(True, alpha=0.3)\n"
            "    plt.show()"
        ),
        _md(
            "**Free response:** Which principle explains the smear when pitch is wrong? "
            "Why does the meter error grow with range?"
        ),
        _md("_Write 3–6 sentences here._"),
        _md(
            "**Assignment:** The rig's pitch is 2° off the truth. "
            "What would you change so a lane point at 25 m lands back on the correct ground coordinate? "
            "See `python modules/01_camera_geometry/break_it_fix_it.py`."
        ),
        _md("_Write 3–6 sentences here._"),
        _code(
            "from calibrate_rig import main as calib_main\n"
            "m01_metrics = calib_main()\n"
            "print('metrics outputs:', m01_metrics.get('outputs', {}))\n"
            "print('run_id:', m01_metrics.get('run_id'))"
        ),
        _md(
            "**Tests**\n\n"
            "```bash\npython3 -m pytest modules/01_camera_geometry -q\n```\n\n"
            "Scaffold tests always run. "
            "`test_assignment_solutions.py` imports `solutions/01_camera_geometry`; "
            "student fill tests skip until implemented and fail if the implementation is wrong."
        ),
        _md(
            "**Come back cue**\n\n"
            f"{come_back_cue('m01')}\n\n"
            "Suggested slot: 25 minutes. 55 or 90 if you are also writing the principle cells."
        ),
        _code(
            "import sys\nimport json\nfrom pathlib import Path\n\n"
            "repo = Path.cwd()\nif not (repo / 'modules' / '01_camera_geometry').exists():\n"
            "    repo = repo.parent\n"
            "sys.path.insert(0, str(repo / 'modules'))\n"
            "from common.progress import come_back_cue\n\n"
            "print(come_back_cue('m01'))\n"
            "progress_path = repo / 'artifacts' / 'progress.json'\n"
            "if progress_path.is_file():\n"
            "    with progress_path.open() as f:\n"
            "        xp = json.load(f).get('xp', 0)\n"
            "    print(f'XP so far: {xp}')"
        ),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    nbf.write(build_m00(), root / "notebooks" / "00_driving_ml_gym.ipynb")
    nbf.write(build_m01(), root / "notebooks" / "01_cameras_and_ipm.ipynb")
    print("Wrote notebooks/00_driving_ml_gym.ipynb and notebooks/01_cameras_and_ipm.ipynb")
