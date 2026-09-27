#!/usr/bin/env python3
"""Regenerate the Module 01 lesson notebook.

Writes ``notebooks/01_cameras_and_ipm.ipynb`` next to this course staging tree.
The notebook is the lesson: run it top to bottom. This script does not execute it.

    python staging/self-driving-ai-course/scripts/build_m01_lesson_notebook.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(source: str):
    return new_markdown_cell(textwrap.dedent(source).strip() + "\n")


def code(source: str):
    return new_code_cell(textwrap.dedent(source).strip() + "\n")


def build() -> nbformat.NotebookNode:
    cells = []

    cells.append(md("""
    # Module 01 — From camera pixels to a top-down map

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/notebooks/01_cameras_and_ipm.ipynb)

    A forward camera gives you a perspective picture. Lane lines that are parallel on the road meet in that picture, and a meter of road covers fewer pixels as it gets farther away. A planner does not want that picture. It wants a top-down map in meters.

    This notebook builds that map from similar triangles, then from the matrices a calibrated camera actually uses, then from a warp that assumes the road is flat. The last part measures how a small pitch error, and a car that is not on the ground, break the flat-road assumption.

    The frames in `data/m01_sample/` are synthetic drawings, not photographs. Run the cells from top to bottom. **Predict** before you execute.
    """))

    cells.append(md("""
    This cell finds the course repository, or clones it, and puts the camera module on the import path. It also puts figures back on the notebook backend so plots show up in the cell output.
    """))

    cells.append(code("""
    import importlib.util
    import os, subprocess, sys
    from pathlib import Path

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    def keep_inline():
        # Some imports select a file-only backend. Put figures back on the
        # notebook backend so plt.show() renders in this cell's output.
        matplotlib.use("module://matplotlib_inline.backend_inline", force=True)
        ip = None
        try:
            from IPython import get_ipython
            ip = get_ipython()
        except Exception:
            ip = None
        if ip is not None:
            ip.run_line_magic("matplotlib", "inline")

    %matplotlib inline

    def find_repo(start: Path) -> Path:
        for p in [start, start.parent]:
            if (p / "modules" / "01_camera_geometry").is_dir():
                return p.resolve()
        return start.resolve()

    REPO = find_repo(Path.cwd())
    if not (REPO / "modules" / "01_camera_geometry").is_dir():
        dest = Path.cwd() / "self-driving-ai-course"
        if not (dest / "modules" / "01_camera_geometry").is_dir():
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/vvknyn/self-driving-ai-course.git", str(dest)],
                check=True,
            )
        REPO = dest.resolve()
        os.chdir(REPO)
    else:
        os.chdir(REPO)

    missing = []
    for mod, pkg in (("cv2", "opencv-python"), ("PIL", "pillow")):
        if importlib.util.find_spec(mod) is None:
            missing.append(pkg)
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)

    sys.path.insert(0, str(REPO / "modules" / "01_camera_geometry"))
    sys.path.insert(0, str(REPO / "modules"))

    import cv2
    from PIL import Image
    from calibrate_rig import build_tesla_style_rig
    from camera_model import PinholeCamera
    from config import IPMConfig
    from extrinsics import camera_position_to_translation, create_euler_rotation
    from ipm import IPMTransformer
    from stitch import stitch_three_cameras

    def load_solution(mod_name, filename):
        path = REPO / "solutions" / "01_camera_geometry" / filename
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    keep_inline()
    DATA = REPO / "data" / "m01_sample"
    print("repo:", REPO)
    print("sample frames:", DATA)
    print("ready")
    """))

    cells.append(md("""
    The print starts with `repo:`. That directory is the course checkout. Images and calibration are read from the `sample frames:` path under it. `ready` means the camera module imported. Figures from here on use the notebook backend.
    """))

    cells.append(md("## 1. The problem"))
    cells.append(md("""
    Below is the front camera frame. Two solid lane lines run along the road. On the ground those lines are parallel: the gap between them, in meters, does not change with distance.

    **Predict:** in the image the gap is much wider near the bottom of the frame than near the horizon, and the two fitted slopes have opposite signs. They meet at one point.
    """))

    cells.append(code("""
    bgr = cv2.imread(str(DATA / "front.png"))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print("image height, width, channels:", rgb.shape[0], rgb.shape[1], rgb.shape[2])

    yellow = (rgb[:, :, 0] > 200) & (rgb[:, :, 1] > 150) & (rgb[:, :, 2] < 80)
    ys, xs = np.where(yellow)
    print("yellow pixels:", int(yellow.sum()))
    split_col = rgb.shape[1] / 2.0
    print("split column:", f"{split_col:.1f}")

    def fit_lane_slope(u, v):
        design = np.column_stack([v.astype(np.float64), np.ones(len(v))])
        slope, intercept = np.linalg.lstsq(design, u.astype(np.float64), rcond=None)[0]
        return float(slope), float(intercept)

    left = xs < split_col
    slope_l, icept_l = fit_lane_slope(xs[left], ys[left])
    slope_r, icept_r = fit_lane_slope(xs[~left], ys[~left])
    print("left slope:", f"{slope_l:.6f}")
    print("right slope:", f"{slope_r:.6f}")
    print("slope difference:", f"{slope_l - slope_r:.6f}")
    v_meet = (icept_r - icept_l) / (slope_l - slope_r)
    u_meet = slope_l * v_meet + icept_l
    print("vanishing column, row:", f"{u_meet:.4f}", f"{v_meet:.4f}")

    rows_both = []
    for row in range(int(ys.min()), int(ys.max()) + 1):
        on_row = ys == row
        if np.any(xs[on_row] < split_col) and np.any(xs[on_row] >= split_col):
            u_left = float(xs[on_row][xs[on_row] < split_col].mean())
            u_right = float(xs[on_row][xs[on_row] >= split_col].mean())
            rows_both.append((row, u_right - u_left))
    far_row, far_sep = rows_both[0]
    near_row, near_sep = rows_both[-1]
    print("far row:", far_row, "separation px:", f"{far_sep:.1f}")
    print("near row:", near_row, "separation px:", f"{near_sep:.1f}")
    print("near / far separation:", f"{near_sep / far_sep:.2f}")

    overlay = rgb.copy()
    v0 = int(np.floor(min(float(ys.min()), v_meet)))
    v1 = int(np.ceil(float(ys.max())))
    for slope, intercept, color in (
        (slope_l, icept_l, (255, 40, 40)),
        (slope_r, icept_r, (40, 90, 255)),
    ):
        p0 = (int(round(slope * v0 + intercept)), v0)
        p1 = (int(round(slope * v1 + intercept)), v1)
        cv2.line(overlay, p0, p1, color, 1)
    cv2.circle(overlay, (int(round(u_meet)), int(round(v_meet))), 4, (255, 255, 255), -1)

    keep_inline()
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.6))
    axes[0].imshow(rgb)
    axes[0].set_title("front camera")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("fitted lane lines")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()
    """))

    cells.append(md("""
    The frame is `180` rows by `320` columns, with `3` color channels. OpenCV reads the file in blue-green-red order; the cell converts it before drawing so the sky and the yellow lines look right.

    There are `1426` yellow pixels. Split at column `160.0`, the left line has slope `-3.942878` and the right line has slope `3.942878`. The difference is `-7.885755`. The slopes have opposite signs, so the image lines are not parallel. As the row index grows (down the image), one line walks left and the other walks right.

    On row `84`, the highest row that still has both lines, the gap is `24.0` pixels. On row `122` it is `298.0` pixels. The near gap is `12.42` times the far gap. The fitted lines meet at column `159.5000`, row `82.7753` (the white dot). That meeting point is the **vanishing point**: the image of the direction the parallel lines share.

    A planner that treated pixel gaps as meters would think the lane was opening up in front of the car. It is not. The next sections undo that perspective.
    """))

    cells.append(md("## 2. The pinhole camera"))
    cells.append(md("""
    A **pinhole camera** is the model where every light ray goes through one point, the pinhole, and hits a flat sensor. **Similar triangles** then relate a point in front of the camera to a pixel.

    Put the pinhole at the origin. `X` is how far the point sits to the right of the optical axis, in meters. `Z` is how far it sits in front of the camera, in meters. The **focal length** `f` is the distance from the pinhole to the sensor, measured in pixels. The **principal point** `c` is the pixel where the optical axis hits the sensor.

    The horizontal pixel is `u = f · X / Z + c`. The part that moves is the offset from the principal point.

    **Predict:** doubling the depth cuts that offset in half. The next cell uses a round focal length so you can check the fraction by hand, then draws the three depths on a blank frame the same size as the road image.
    """))

    cells.append(code("""
    focal = 100.0
    lateral_m = 2.0
    principal = 160.0
    print("focal px:", f"{focal:.1f}")
    print("lateral m:", f"{lateral_m:.1f}")
    print("principal column:", f"{principal:.1f}")
    for depth in (10.0, 20.0, 40.0):
        offset = focal * lateral_m / depth
        column = offset + principal
        print(
            f"Z {depth:.0f} offset {offset:.4f} column {column:.4f}"
        )
    print("offset ratio Z10 / Z20:", f"{(focal * lateral_m / 10.0) / (focal * lateral_m / 20.0):.1f}")
    print("offset ratio Z10 / Z40:", f"{(focal * lateral_m / 10.0) / (focal * lateral_m / 40.0):.1f}")

    blank = np.zeros((rgb.shape[0], rgb.shape[1], 3), dtype=np.uint8)
    blank[:] = (30, 30, 30)
    print("blank frame height, width:", blank.shape[0], blank.shape[1])
    keep_inline()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(blank)
    ax.axvline(principal, color="white", linewidth=0.6, linestyle="--")
    for depth, color in ((10.0, "gold"), (20.0, "deepskyblue"), (40.0, "tomato")):
        column = focal * lateral_m / depth + principal
        ax.scatter([column], [rgb.shape[0] / 2.0], s=40, c=color, label=f"Z={depth:.0f}")
    ax.set_xlim(0, rgb.shape[1])
    ax.set_ylim(rgb.shape[0], 0)
    ax.legend(loc="upper right")
    ax.set_title("same lateral offset, three depths")
    plt.tight_layout()
    plt.show()
    """))

    cells.append(md("""
    At depth `10` the offset is `20.0000` pixels (column `180.0000`). At depth `20` the offset is `10.0000` (column `170.0000`). At depth `40` it is `5.0000` (column `165.0000`). The printed ratios are `2.0` and `4.0`.

    Doubling the depth halved the offset. That is the whole of perspective in one fraction: the sensor sees the angle `X / Z`, and the focal length turns that angle into pixels. The dashed line is the principal column `160.0`. Farther points sit closer to it. The blank frame is the same `180` by `320` size as the road image, so you can compare these columns with the picture above.

    Vertical pixels work the same way, with their own focal length and principal row. This notebook ignores lens distortion. `PinholeCamera` can apply a radial term later; the sample rig sets that term to zero.
    """))

    cells.append(md("""
    **Try this.** Cut the focal length in half and keep the same point at depth `10`. **Predict:** the offset from the principal point also halves, from `20.0000` pixels to `10.0000`.
    """))

    cells.append(code("""
    focal_half = 50.0
    offset_half = focal_half * lateral_m / 10.0
    print("focal px:", f"{focal_half:.1f}")
    print("Z 10 offset:", f"{offset_half:.4f}")
    print("offset ratio vs focal 100:", f"{offset_half / 20.0:.1f}")
    """))

    cells.append(md("""
    Focal length `50.0` puts the same point `10.0000` pixels off center. The ratio versus the `20.0000` pixel offset is `0.5`. A shorter focal length is a wider field of view: the same object covers fewer pixels, which is what a wide front camera is for. It does not remove the divide-by-depth shrinking. Far objects are still small.
    """))

    cells.append(md("## 3. Intrinsics"))
    cells.append(md("""
    The **intrinsic matrix** `K` packs the focal lengths and the principal point into one matrix. **Homogeneous coordinates** means we keep an extra component and divide only at the end. For a point already in the camera frame, `(X, Y, Z)`,

    `K @ (X, Y, Z)` produces `(u * Z, v * Z, Z)`. Dividing by the third entry gives the pixel `(u, v)`.

    `fx` and `fy` are focal lengths in pixels. `cx` and `cy` are the principal point. Focal lengths sit on the diagonal, the principal point sits in the last column, and the corner is a one. There is no depth inside `K`. The divide by `Z` does that job.

    **Predict:** a matrix built that way matches both the matrix stored on `PinholeCamera` and the reference `build_intrinsic_matrix`. The same round focal length as the previous section, applied to a point that is not on the centerline, reproduces the column you already saw for the nearest depth.
    """))

    cells.append(code("""
    import json
    import camera_model as camera_model_module

    with (DATA / "calib.json").open(encoding="utf-8") as handle:
        calib = json.load(handle)

    fx = float(calib["fx"])
    fy = float(calib["fy"])
    cx = float(calib["cx"])
    cy = float(calib["cy"])
    print("fx:", f"{fx:.6f}")
    print("fy:", f"{fy:.6f}")
    print("cx:", f"{cx:.1f}")
    print("cy:", f"{cy:.1f}")
    print("calib width, height:", int(calib["width"]), int(calib["height"]))

    K_hand = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    print("K shape:", K_hand.shape[0], K_hand.shape[1])
    print("K:")
    print(np.array2string(K_hand, precision=6, suppress_small=True))

    cameras = build_tesla_style_rig(img_w=int(calib["width"]), img_h=int(calib["height"]))
    front = cameras["front"]
    print("max abs diff vs camera.K:", f"{np.max(np.abs(K_hand - front.K)):.6e}")

    sol_cam = load_solution("sol_m01_cam_lesson", "camera_model.py")
    K_ref = sol_cam.build_intrinsic_matrix(fx, fy, cx, cy)
    print("max abs diff vs reference K:", f"{np.max(np.abs(K_hand - K_ref)):.6e}")

    try:
        camera_model_module.build_intrinsic_matrix(fx, fy, cx, cy)
        print("module build_intrinsic_matrix: returned")
    except NotImplementedError:
        print("module build_intrinsic_matrix: NotImplementedError")

    point_cam = np.array([2.0, 1.0, 10.0])
    print("camera-frame point:", " ".join(f"{v:.1f}" for v in point_cam))
    K_toy = np.array(
        [[focal, 0.0, principal], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    print("toy focal, principal column, principal row:", f"{focal:.1f}", f"{principal:.1f}", f"{cy:.1f}")
    multiplied = K_toy @ point_cam
    print("K @ point:", " ".join(f"{v:.4f}" for v in multiplied))
    print("third component:", f"{multiplied[2]:.4f}")
    u_toy = multiplied[0] / multiplied[2]
    v_toy = multiplied[1] / multiplied[2]
    print("pixel column, row:", f"{u_toy:.4f}", f"{v_toy:.4f}")
    print("column offset from principal:", f"{u_toy - principal:.4f}")
    """))

    cells.append(md("""
    The sample front camera has `fx = fy = 92.376043`, principal point `(160.0, 90.0)`, and a `320` by `180` image. `K` is `3` by `3`. The hand-built matrix matches `PinholeCamera.K` and the reference `build_intrinsic_matrix` with max absolute difference `0.000000e+00` in both cases.

    The function that tests import from `camera_model.py` still raises `NotImplementedError`. The class does not call it. It writes the same nine numbers itself. The exercise below is that function.

    For the toy point `(2.0, 1.0, 10.0)` with focal length `100.0`, the product is `1800.0000  1000.0000  10.0000`. The third component is the depth, `10.0000`. Dividing gives column `180.0000` and row `100.0000`. The column offset is `20.0000`, the same offset as the depth-`10` point in the previous section. Homogeneous coordinates did not change the geometry. They let one matrix multiply stand for "scale by the focal length, add the principal point, and remember to divide by depth."
    """))

    cells.append(md("""
    **Exercise — `build_intrinsic_matrix`.** Return the `3` by `3` matrix from `fx`, `fy`, `cx`, and `cy`. Leave the `TODO` as it is to use the reference implementation. The check compares your matrix with `front.K`.
    """))

    cells.append(code("""
    def build_intrinsic_matrix_student(fx_value, fy_value, cx_value, cy_value):
        # TODO: return the 3x3 intrinsic matrix
        raise NotImplementedError

    def get_intrinsic_fn():
        try:
            build_intrinsic_matrix_student(1.0, 1.0, 0.0, 0.0)
        except NotImplementedError:
            print("Using reference build_intrinsic_matrix (TODO not implemented)")
            return sol_cam.build_intrinsic_matrix
        print("Using your build_intrinsic_matrix")
        return build_intrinsic_matrix_student

    intrinsic_fn = get_intrinsic_fn()
    K_check = intrinsic_fn(front.fx, front.fy, front.cx, front.cy)
    intrinsic_diff = float(np.max(np.abs(K_check - front.K)))
    print("max abs diff vs camera.K:", f"{intrinsic_diff:.6e}")
    assert K_check.shape == (3, 3)
    assert intrinsic_diff < 1e-9
    print("✅ correct")
    """))

    cells.append(md("""
    The check prints `✅`. The max absolute difference versus `camera.K` is `0.000000e+00`. The reference function is used because the `TODO` still raises. Replace the `TODO` and run the cell again if you want the check to call your function instead.
    """))

    cells.append(md("""
    <details><summary>Solution</summary>

    ```python
    def build_intrinsic_matrix(fx, fy, cx, cy):
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    ```

    </details>
    """))

    cells.append(md("## 4. Extrinsics"))
    cells.append(md("""
    **Extrinsics** answer a different question: where is a point that we measured on the car, once we stand at the camera?

    A **rotation** turns a vector without changing its length. In two dimensions the matrix is `[[cos θ, -sin θ], [sin θ, cos θ]]`. It sends the unit x-axis to `(cos θ, sin θ)`.

    **Predict:** a quarter turn counterclockwise sends a point on the horizontal axis onto the vertical axis, and the distance from the origin does not change.
    """))

    cells.append(code("""
    theta_deg = 90.0
    theta = np.deg2rad(theta_deg)
    rotation_2d = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )
    point_2d = np.array([4.0, 0.0])
    rotated = rotation_2d @ point_2d
    print("theta degrees:", f"{theta_deg:.1f}")
    print("input x, y:", f"{point_2d[0]:.1f}", f"{point_2d[1]:.1f}")
    print("rotated x, y:", f"{rotated[0]:.4f}", f"{rotated[1]:.4f}")
    """))

    cells.append(md("""
    `90.0` degrees sends `(4.0, 0.0)` to `(0.0000, 4.0000)`. The x-axis swung up onto the y-axis. A rotation in three dimensions is the same idea with one more axis. A **translation** then slides the origin. The camera does both: `P_cam = R @ P_ego + T`.
    """))

    cells.append(md("""
    The car and the camera do not share axes.

    **Ego frame** (ISO 8855, used for the rest of this course): `X` forward, `Y` left, `Z` up. The origin is on the ground at the rear axle.

    **Camera frame:** `X` right, `Y` down, `Z` forward, along the optical axis.

    ```
    Z_ego (up)
      |
      |     camera * ----> Z_cam (look direction)
      |    /
      |   /   pitched down
      +--------> X_ego (forward)
     rear axle
    ```

    With no pitch, yaw, or roll, only the axis names change. **Predict:** ego forward becomes the camera's look direction. Ego left becomes negative camera `X`, because camera `X` points right. Ego up becomes negative camera `Y`, because camera `Y` points down.
    """))

    cells.append(code("""
    axis_change = np.array(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    def fmt(vec):
        return " ".join(f"{v:.1f}" for v in np.asarray(vec, dtype=np.float64))

    print("ego forward -> camera:", fmt(axis_change @ np.array([1.0, 0.0, 0.0])))
    print("ego left -> camera:", fmt(axis_change @ np.array([0.0, 1.0, 0.0])))
    print("ego up -> camera:", fmt(axis_change @ np.array([0.0, 0.0, 1.0])))

    print("pitch deg:", f"{float(calib['pitch_deg']):.1f}")
    print("yaw deg:", f"{float(calib['yaw_deg']):.1f}")
    print("roll deg:", f"{float(calib['roll_deg']):.1f}")
    cam_pos = np.array(calib["cam_position_ego"], dtype=np.float64)
    print("camera position ego x, y, z:", fmt(cam_pos))

    R_ego = create_euler_rotation(
        float(calib["pitch_deg"]), float(calib["yaw_deg"]), float(calib["roll_deg"])
    )
    T_ego = camera_position_to_translation(R_ego, cam_pos).reshape(3)
    print("T x, y, z:", " ".join(f"{v:.6f}" for v in T_ego))
    print("max abs diff vs camera.T:", f"{np.max(np.abs(T_ego - front.T.reshape(3))):.6e}")
    print("max abs diff vs camera.R:", f"{np.max(np.abs(R_ego - front.R)):.6e}")

    def project_chain(point_ego):
        point_cam = front.R @ point_ego + front.T.reshape(3)
        scaled = front.K @ point_cam
        pixel = scaled[:2] / scaled[2]
        repo_px, valid = front.project_ego_to_pixel(point_ego)
        print("ego x, y, z:", " ".join(f"{v:.6f}" for v in point_ego))
        print("P_cam x, y, z:", " ".join(f"{v:.6f}" for v in point_cam))
        print("K @ P_cam:", " ".join(f"{v:.6f}" for v in scaled))
        print("divide by Z_cam:", f"{scaled[2]:.6f}")
        print("pixel column, row:", f"{pixel[0]:.6f}", f"{pixel[1]:.6f}")
        print("project_ego_to_pixel:", f"{repo_px[0, 0]:.6f}", f"{repo_px[0, 1]:.6f}", "valid", bool(valid[0]))
        print(
            "pixel diff column, row:",
            f"{pixel[0] - repo_px[0, 0]:.6f}",
            f"{pixel[1] - repo_px[0, 1]:.6f}",
        )
        return pixel

    print("--- point on the centerline ---")
    project_chain(np.array([10.0, 0.0, 0.0]))
    print("--- point 1 m to the left ---")
    project_chain(np.array([20.0, 1.0, 0.0]))
    """))

    cells.append(md("""
    The plain axis change does what the diagram says. Ego forward becomes camera `0.0 0.0 1.0` (straight out the lens). Ego left becomes `-1.0 0.0 0.0` (camera `X` points right, so left is negative). Ego up becomes `0.0 -1.0 0.0` (camera `Y` points down, so up is negative).

    The real front camera is not that ideal pose. It sits `2.0` m forward, `0.0` m to the side, and `1.4` m up, pitched down by `4.0` degrees. Yaw and roll are `0.0`. `create_euler_rotation` builds `R`, and `camera_position_to_translation` builds `T = -R @ camera_position`. Both match the camera object: max absolute difference `0.000000e+00`.

    Take the ground point `10.000000` m ahead on the centerline, ego `(10.000000, 0.000000, 0.000000)`. In the camera frame it is `(0.000000, 0.838538, 8.078171)`. Camera `X` is `0.000000` because the point is neither left nor right of the lens. Camera `Y` is positive, so the point is below the optical axis (the road is under a camera that looks slightly down). Camera `Z` is `8.078171` m, not the ego `X` of `10.000000`, because the camera itself is already `2.0` m forward of the rear axle and the pitch tilts the optical axis.

    `K @ P_cam` is `1292.507434  804.496243  8.078171`. Dividing by `8.078171` gives pixel `(160.000000, 99.588904)`. The column is the principal point, as it should be for a point on the centerline. `project_ego_to_pixel` returns the same pixel. The difference is `0.000000` and `0.000000`.

    The point `20.000000` m ahead and `1.000000` m to the left (ego `Y` is left) has camera `X` of `-1.000000`. Its pixel is `(154.883294, 90.721318)`. It moved left of column `160.000000`, which is the image of "left" once the axes have been swapped. That pixel is what the homography in the next section has to reproduce in one matrix.
    """))

    cells.append(md("## 5. Stretching the image"))
    cells.append(md("""
    An obvious attempt: the far road is compressed vertically, so stretch the image vertically until it "looks" top-down.

    **Predict:** the two lane slopes stay unequal. The lines still meet at one column. Changing only the height cannot make parallel lines parallel, because a stretch never divides by depth.
    """))

    cells.append(code("""
    tall = cv2.resize(
        rgb,
        (rgb.shape[1], rgb.shape[0] * 2),
        interpolation=cv2.INTER_NEAREST,
    )
    print("stretched height, width:", tall.shape[0], tall.shape[1])
    print("height ratio:", f"{tall.shape[0] / rgb.shape[0]:.1f}")

    yellow_tall = (tall[:, :, 0] > 200) & (tall[:, :, 1] > 150) & (tall[:, :, 2] < 80)
    ys_t, xs_t = np.where(yellow_tall)
    left_t = xs_t < tall.shape[1] / 2.0
    slope_lt, icept_lt = fit_lane_slope(xs_t[left_t], ys_t[left_t])
    slope_rt, icept_rt = fit_lane_slope(xs_t[~left_t], ys_t[~left_t])
    print("stretched left slope:", f"{slope_lt:.6f}")
    print("stretched right slope:", f"{slope_rt:.6f}")
    print("stretched slope difference:", f"{slope_lt - slope_rt:.6f}")
    v_meet_t = (icept_rt - icept_lt) / (slope_lt - slope_rt)
    u_meet_t = slope_lt * v_meet_t + icept_lt
    print("stretched vanishing column, row:", f"{u_meet_t:.4f}", f"{v_meet_t:.4f}")
    print(
        "|slope difference| ratio, original / stretched:",
        f"{abs(slope_l - slope_r) / abs(slope_lt - slope_rt):.3f}",
    )

    keep_inline()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].imshow(rgb)
    axes[0].set_title("original")
    axes[0].axis("off")
    axes[1].imshow(tall)
    axes[1].set_title("height times 2")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()
    """))

    cells.append(md("""
    The only change is the height: `360` rows instead of `180`, ratio `2.0`. Width stays `320`. Nearest-neighbour resize keeps the yellow pixels yellow, so the same fit still applies.

    The slopes become `-1.970265` and `1.970265`. The difference becomes `-3.940529`. Its absolute value shrank by a factor of `2.001`, almost exactly the stretch. The lines still do not share a slope. They still meet, at column `159.5000` and row `166.0268`. The column did not move. The row moved because every row index was scaled.

    A vertical stretch is an affine resize. Perspective is a divide by depth. Those are different operations, so the lanes still converge. The next section uses the divide that the pinhole model already has.
    """))

    cells.append(md("## 6. The ground plane is a homography"))
    cells.append(md("""
    Assume every point we care about is on the ground. In the ego frame that height is `0.000000`, the value already printed for the centerline point. Then the third column of `R` multiplies that height and drops out. What remains is a single `3` by `3` matrix, a **homography**:

    `H = K @ [r1 r2 t]`

    `r1` and `r2` are the first two columns of `R`. `t` is the translation. `H` maps `(X, Y, 1)` on the ground to `(u * w, v * w, w)`. Divide by `w` and you have a pixel. `H` inverse maps a pixel back to `(X, Y)`.

    **Predict:** the centerline ground point from the previous section comes back to the same pixel as the full chain, and applying `H` inverse returns that same ground point up to rounding error.
    """))

    cells.append(code("""
    r1 = front.R[:, 0:1]
    r2 = front.R[:, 1:2]
    t_col = front.T.reshape(3, 1)
    H = front.K @ np.hstack([r1, r2, t_col])
    print("H shape:", H.shape[0], H.shape[1])

    sol_ipm = load_solution("sol_m01_ipm_lesson", "ipm.py")
    H_ref = sol_ipm.build_ground_homography(front.K, front.R, front.T)
    print("max abs diff vs reference H:", f"{np.max(np.abs(H - H_ref)):.6e}")

    ground = np.array([10.0, 0.0, 1.0])
    q = H @ ground
    uv = q[:2] / q[2]
    print("H @ (10, 0, 1) before divide:", " ".join(f"{v:.6f}" for v in q))
    print("homography pixel column, row:", f"{uv[0]:.6f}", f"{uv[1]:.6f}")
    chain_px, chain_ok = front.project_ego_to_pixel(np.array([10.0, 0.0, 0.0]))
    print("chain pixel column, row:", f"{chain_px[0, 0]:.6f}", f"{chain_px[0, 1]:.6f}")
    print(
        "pixel diff column, row:",
        f"{uv[0] - chain_px[0, 0]:.6e}",
        f"{uv[1] - chain_px[0, 1]:.6e}",
    )

    back = np.linalg.inv(H) @ np.array([uv[0], uv[1], 1.0])
    xy = back[:2] / back[2]
    round_err = float(np.linalg.norm(xy - np.array([10.0, 0.0])))
    print("inverse X, Y:", f"{xy[0]:.6f}", f"{xy[1]:.6e}")
    print("round-trip error m:", f"{round_err:.3e}")

    pixel_errs = []
    for x_m, y_m in ((10.0, 0.0), (20.0, 1.0), (30.0, -1.5)):
        q_i = H @ np.array([x_m, y_m, 1.0])
        uv_i = q_i[:2] / q_i[2]
        pix_i, _ = front.project_ego_to_pixel(np.array([x_m, y_m, 0.0]))
        pixel_errs.append(float(np.hypot(uv_i[0] - pix_i[0, 0], uv_i[1] - pix_i[0, 1])))
    print("max pixel error on 3 ground points:", f"{max(pixel_errs):.6e}")
    """))

    cells.append(md("""
    `H` is `3` by `3`. It matches the reference `build_ground_homography` with max absolute difference `0.000000e+00`.

    `H @ (10, 0, 1)` divides down to pixel `(160.000000, 99.588904)`, the same pixel the full chain printed, difference `0.000000e+00` in both column and row. The inverse brings back `X = 10.000000` and a `Y` of `1.793714e-15`. The round-trip error is `5.623e-15` meters. That is floating-point noise, not a bias in the model.

    On three ground points, including the off-center ones, the worst disagreement with `project_ego_to_pixel` is `3.177644e-14` pixels. On flat ground the homography is not an approximation of the pinhole. It is the pinhole with the ground height substituted in.

    `IPMTransformer` does not call `H` when it warps. It projects each ground cell with `project_ego_to_pixel`. On `Z = 0` that is this matrix. The exercise asks you to build `H` anyway, because it is the compact form and the tests check it.
    """))

    cells.append(md("""
    **Exercise — `build_ground_homography`.** Return `H = K @ [r1 r2 t]` for a ground plane `Z = 0`. `t` may arrive as shape `(3,)` or `(3, 1)`. Leave the `TODO` as it is to use the reference implementation.
    """))

    cells.append(code("""
    def build_ground_homography_student(K_value, R_value, t_value):
        # TODO: H = K @ [r1 r2 t] for the ground plane
        raise NotImplementedError

    def get_homography_fn():
        try:
            build_ground_homography_student(front.K, front.R, front.T)
        except NotImplementedError:
            print("Using reference build_ground_homography (TODO not implemented)")
            return sol_ipm.build_ground_homography
        print("Using your build_ground_homography")
        return build_ground_homography_student

    homography_fn = get_homography_fn()
    H_check = homography_fn(front.K, front.R, front.T)
    check_errs = []
    for x_m, y_m in ((10.0, 0.0), (20.0, 1.0), (30.0, -1.5)):
        q_i = H_check @ np.array([x_m, y_m, 1.0])
        uv_i = q_i[:2] / q_i[2]
        pix_i, valid_i = front.project_ego_to_pixel(np.array([x_m, y_m, 0.0]))
        err_i = float(np.hypot(uv_i[0] - pix_i[0, 0], uv_i[1] - pix_i[0, 1]))
        check_errs.append(err_i)
        print(
            f"ground {x_m:.1f} {y_m:.1f} pixel error {err_i:.6e} valid {bool(valid_i[0])}"
        )
    print("max pixel error:", f"{max(check_errs):.6e}")
    tolerance_px = 1e-3
    print("pixel tolerance:", f"{tolerance_px:.0e}")
    assert max(check_errs) < tolerance_px
    print("✅ correct")
    """))

    cells.append(md("""
    The check prints `✅`. The largest pixel error against `project_ego_to_pixel` is `3.177644e-14`, far under the printed pixel tolerance `1e-03`. The reference function is used because the `TODO` still raises.
    """))

    cells.append(md("""
    <details><summary>Solution</summary>

    ```python
    def build_ground_homography(K, R, t):
        t_vec = np.asarray(t, dtype=np.float64).reshape(3, 1)
        return K @ np.hstack([R[:, 0:2], t_vec])
    ```

    </details>
    """))

    cells.append(md("""
    A **bird's-eye view** (BEV) is a picture whose rows and columns are meters on the ground, not angles. **Inverse perspective mapping** (IPM) builds it by walking a grid on `Z = 0` and copying the color each cell projects to.

    `IPMTransformer` walks a meter grid on the ground and copies the color each cell projects to. The first row of that picture is the far edge. A lane at constant lateral position should be a single column at every forward distance.

    **Predict:** the two solid lanes in the sample, which are parallel on the road, each collapse to one column, and the gap between those columns is the gap between the lanes in meters.
    """))

    cells.append(code("""
    x_range = (4.0, 40.0)
    y_range = (-10.0, 10.0)
    bev_resolution = 0.1
    print("forward range m:", f"{x_range[0]:.1f}", f"{x_range[1]:.1f}")
    print("lateral range m:", f"{y_range[0]:.1f}", f"{y_range[1]:.1f}")
    print("meters per pixel:", f"{bev_resolution:.1f}")

    ipm = IPMTransformer(front, x_range, y_range, bev_resolution)
    print("bev height, width:", ipm.bev_height, ipm.bev_width)
    print("row 0 forward m:", f"{ipm.bev_pixel_to_metric(0, 0)[0]:.1f}")

    lane_y = (5.6, -5.6)
    for y_m in lane_y:
        columns = {
            ipm.metric_to_bev_pixel(float(x_m), float(y_m))[0]
            for x_m in np.linspace(6.0, 38.0, 17)
        }
        print(f"lane y {y_m:.1f} columns:", " ".join(str(c) for c in sorted(columns)))
    col_left = ipm.metric_to_bev_pixel(20.0, lane_y[0])[0]
    col_right = ipm.metric_to_bev_pixel(20.0, lane_y[1])[0]
    print("lane columns:", col_left, col_right)
    print("separation m:", f"{(col_right - col_left) * bev_resolution:.1f}")

    image_seps = []
    for x_m in (8.0, 30.0):
        pair = []
        for y_m in lane_y:
            pix, ok = front.project_ego_to_pixel(np.array([x_m, y_m, 0.0]))
            pair.append(float(pix[0, 0]))
            print(
                f"ego x {x_m:.1f} y {y_m:.1f} pixel {pix[0, 0]:.2f} {pix[0, 1]:.2f} valid {bool(ok[0])}"
            )
        sep = abs(pair[1] - pair[0])
        print(f"image separation at x {x_m:.1f}:", f"{sep:.2f}")
        image_seps.append(sep)
    print("image separation ratio, near / far:", f"{image_seps[0] / image_seps[1]:.2f}")

    bev_bgr = ipm.warp_to_bev(bgr)
    bev_rgb = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2RGB)
    print("warp shape height, width, channels:", bev_rgb.shape[0], bev_rgb.shape[1], bev_rgb.shape[2])

    yellow_bev = (bev_rgb[:, :, 0] > 180) & (bev_rgb[:, :, 1] > 140) & (bev_rgb[:, :, 2] < 100)
    for x_m in (8.0, 36.0):
        row = int(round((x_range[1] - x_m) / bev_resolution))
        cols = np.where(yellow_bev[row])[0]
        left_cols = cols[cols < ipm.bev_width // 2]
        right_cols = cols[cols >= ipm.bev_width // 2]
        print(
            f"paint at forward m {x_m:.1f} row {row} "
            f"width m {((left_cols.max() - left_cols.min()) * bev_resolution):.2f} "
            f"{((right_cols.max() - right_cols.min()) * bev_resolution):.2f}"
        )

    keep_inline()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    axes[0].imshow(rgb)
    axes[0].set_title("perspective")
    axes[0].axis("off")
    axes[1].imshow(bev_rgb)
    axes[1].set_title("top-down, far at the top")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()
    """))

    cells.append(md("""
    The grid is `360` by `200` pixels: `(40.0 - 4.0) / 0.1` rows and `(10.0 - (-10.0)) / 0.1` columns. Row `0` is forward distance `40.0` m. The top of the BEV picture is the far end of the road, same as in the camera image.

    The lane at `5.6` m is column `44` at every sampled forward distance. The lane at `-5.6` m is column `156`. One column each. The separation is `11.2` m, which is `5.6 - (-5.6)`.

    In the camera image those same two lanes are not a constant gap. At forward distance `8.0` m the pixels are `74.96` and `245.04` apart in column, a gap of `170.08` pixels, both on row `104.85`. At `30.0` m the pixels are `141.54` and `178.46`, a gap of `36.91` pixels, both on row `88.17`. Same two lines, `11.2` m apart on the road, and the near gap is `4.61` times the far gap. The BEV undoes that.

    The yellow paint is not a hairline. A stroke that is a few pixels wide in the image covers `1.20` m of ground at `8.0` m forward and `6.00` m of ground at `36.0` m forward. The geometric lane is one column. The paint gets fatter with range because a pixel subtends more meters when it is far away. That is a drawing artifact of the sample, not a bend in the road.
    """))

    cells.append(md("## 7. How the warp picks a color"))
    cells.append(md("""
    A ground cell rarely projects to an integer pixel. **Bilinear interpolation** blends the four surrounding pixels. **Nearest neighbour** copies the single closest pixel.

    The next cell labels a unit square with four corner values and samples a point inside it. **Predict:** the blend is none of the four corner values, and nearest neighbour copies the closest corner instead.
    """))

    cells.append(code("""
    u_frac, v_frac = 0.25, 0.75
    corners = {(0, 0): 1.0, (1, 0): 3.0, (0, 1): 5.0, (1, 1): 7.0}
    terms = [
        (1 - u_frac) * (1 - v_frac) * corners[(0, 0)],
        u_frac * (1 - v_frac) * corners[(1, 0)],
        (1 - u_frac) * v_frac * corners[(0, 1)],
        u_frac * v_frac * corners[(1, 1)],
    ]
    print("sample column, row:", f"{u_frac:.2f}", f"{v_frac:.2f}")
    print("four contributions:", " ".join(f"{term:.4f}" for term in terms))
    print("bilinear value:", f"{sum(terms):.4f}")
    nearest = min(corners, key=lambda corner: (corner[0] - u_frac) ** 2 + (corner[1] - v_frac) ** 2)
    print("nearest corner column, row:", nearest[0], nearest[1])
    print("nearest value:", f"{corners[nearest]:.1f}")
    """))

    cells.append(md("""
    The four contributions are `0.1875`, `0.1875`, `2.8125`, and `1.3125`. They add to `4.5000`. No source pixel has that value. Bilinear invented it, weighted toward the corner `(0, 1)` because the sample is only `0.25` across and `0.75` down.

    The nearest corner is `(0, 1)`, value `5.0`. Nearest neighbour does not invent colors. It copies a block. `IPMTransformer.warp_to_bev` uses bilinear sampling (`cv2.INTER_LINEAR`) for the same reason this sum is `4.5000`: a ground cell that lands between pixels should take a bit of each.
    """))

    cells.append(md("""
    **Try this.** Warp the same front image with nearest-neighbour sampling instead of bilinear. One argument changes. **Predict:** nearest invents no new colors, so the count of BEV pixels whose color is absent from the source image is none. The picture gets blocky along edges.
    """))

    cells.append(code("""
    bev_nearest = cv2.remap(
        bgr,
        ipm.map_x,
        ipm.map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    bev_nearest = bev_nearest.copy()
    bev_nearest[~ipm.valid_mask] = 0

    source_colors = {tuple(pixel) for pixel in np.unique(bgr.reshape(-1, 3), axis=0)}
    print("source colors:", len(source_colors))

    def invented(image):
        count = 0
        for pixel in image.reshape(-1, 3):
            key = tuple(int(channel) for channel in pixel)
            if key != (0, 0, 0) and key not in source_colors:
                count += 1
        return count

    invented_linear = invented(bev_bgr)
    invented_nearest = invented(bev_nearest)
    print("bilinear pixels with a new color:", invented_linear)
    print("nearest pixels with a new color:", invented_nearest)

    differ = np.any(bev_bgr != bev_nearest, axis=-1)
    best = None
    window = 30
    for row in range(0, differ.shape[0] - window, 5):
        for col in range(0, differ.shape[1] - window, 5):
            score = int(differ[row:row + window, col:col + window].sum())
            if best is None or score > best[0]:
                best = (score, row, col)
    print("crop size:", window)
    print("crop row, column:", best[1], best[2])
    print("pixels that differ inside the crop:", best[0])

    r0, c0 = best[1], best[2]
    keep_inline()
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(cv2.cvtColor(bev_bgr[r0:r0 + window, c0:c0 + window], cv2.COLOR_BGR2RGB))
    axes[0].set_title("bilinear")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(bev_nearest[r0:r0 + window, c0:c0 + window], cv2.COLOR_BGR2RGB))
    axes[1].set_title("nearest")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()
    """))

    cells.append(md("""
    The source image has `5` colors. Bilinear writes `17205` BEV pixels whose color is none of those five: blends along the lane edges and the horizon. Nearest writes `0` new colors. It can only copy a source pixel or leave a cell black.

    The `30` by `30` crop at row `145`, column `110` contains `742` pixels where the two warps disagree. The nearest crop is made of flat blocks. The bilinear crop softens the same edges. For a metric map, bilinear is the better default. Nearest is useful when you need to know the color was really in the image, for example when you are counting paint rather than drawing a smooth road.
    """))

    cells.append(md("## 8. Three cameras, one map"))
    cells.append(md("""
    One forward camera looks ahead. It misses some ground close to the car, off to either side. The sample rig has a left camera and a right camera, each yawed outward. `stitch_three_cameras` warps all three onto the same meter grid and averages the colors where more than one camera can see the cell.

    **Predict:** the stitched picture covers every cell of this grid. The cells the front camera leaves black are near the bumper and out to the sides, and some cells are seen by all three cameras.
    """))

    cells.append(code("""
    frames = {}
    for name in ("front", "left", "right"):
        frames[name] = cv2.imread(str(DATA / f"{name}.png"))
        print(f"{name} height, width:", frames[name].shape[0], frames[name].shape[1])

    stitched = stitch_three_cameras(frames, cameras, x_range, y_range, bev_resolution)
    print("stitched height, width:", stitched.shape[0], stitched.shape[1])

    def coverage(image):
        return float(np.any(image > 0, axis=-1).mean())

    print("front coverage:", f"{coverage(bev_bgr):.4f}")
    print("stitched coverage:", f"{coverage(stitched):.4f}")
    front_on = np.any(bev_bgr > 0, axis=-1)
    stitched_on = np.any(stitched > 0, axis=-1)
    added = stitched_on & ~front_on
    added_rows, added_cols = np.where(added)
    added_x = x_range[1] - added_rows * bev_resolution
    added_y = y_range[1] - added_cols * bev_resolution
    print("cells only the side cameras color:", int(added.sum()))
    print("those cells, forward m min max:", f"{added_x.min():.1f}", f"{added_x.max():.1f}")
    print("those cells with |lateral| > 6 m:", int(np.sum(np.abs(added_y) > 6.0)))

    weight = np.zeros(front_on.shape, dtype=np.int32)
    for name in ("front", "left", "right"):
        view = IPMTransformer(cameras[name], x_range, y_range, bev_resolution)
        weight += view.valid_mask.astype(np.int32)
    print("max cameras on one cell:", int(weight.max()))
    print("cells seen by 2 or more:", int(np.sum(weight >= 2)))
    print("cells seen by all 3:", int(np.sum(weight == 3)))
    print("cells seen by exactly 1:", int(np.sum(weight == 1)))

    keep_inline()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    axes[0].imshow(bev_rgb)
    axes[0].set_title("front only")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
    axes[1].set_title("front + left + right")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()
    """))

    cells.append(md("""
    All three frames are `180` by `320`. The stitched map is the same `360` by `200` grid as the front warp.

    Front coverage is `0.9682`. Stitched coverage is `1.0000`. The side cameras color `2292` cells the front warp left black. Those cells sit between `4.1` m and `7.6` m forward, and `1974` of them are more than `6` m off the centerline. That is the near left and near right corner, beside the bumper, where a forward camera is looking over the road rather than at it.

    Overlap is the common case, not the exception. The busiest cell is seen by `3` cameras. `69278` cells are seen by `2` or more, `7276` by all three, and `2722` by exactly one. Where masks overlap, the stitch averages the colors. It does not feather a seam. On this synthetic road the cameras agree, so the average looks like one picture. On a real rig, a bad calibration shows up as a doubled lane line in that overlap.
    """))

    cells.append(md("## 9. Where the flat road breaks"))
    cells.append(md("""
    The homography trusts two numbers you can get wrong: the pitch of the camera, and the assumption that the pixel came from the ground.

    **Pitch** here is the nose-down angle in `create_euler_rotation`. Suppose the camera is slightly more nose-down than the calibration says. You still observe the true pixel of a ground point, but you intersect that pixel with the ground using the wrong pitch. `pitch_shift_meters` is that mistake, in meters: estimated forward distance minus the true forward distance.

    **Predict:** no extra pitch gives a numerical zero. A small extra nose-down pitch is a modest miss nearby and a much larger miss far away. The error grows faster than the distance.
    """))

    cells.append(code("""
    def range_error(calib_dict, delta_deg, range_m):
        width = int(calib_dict["width"])
        height = int(calib_dict["height"])
        position = np.array(calib_dict["cam_position_ego"], dtype=np.float64)
        pitch = float(calib_dict["pitch_deg"])
        yaw = float(calib_dict["yaw_deg"])
        roll = float(calib_dict["roll_deg"])
        rotation_true = create_euler_rotation(pitch, yaw, roll)
        translation_true = camera_position_to_translation(rotation_true, position)
        true_cam = PinholeCamera(
            "true", calib_dict["fx"], calib_dict["fy"], calib_dict["cx"], calib_dict["cy"],
            width, height, rotation_true, translation_true,
        )
        pixel, valid = true_cam.project_ego_to_pixel(np.array([[range_m, 0.0, 0.0]]))
        if not valid[0]:
            return 0.0
        rotation_bias = create_euler_rotation(pitch + delta_deg, yaw, roll)
        translation_bias = camera_position_to_translation(rotation_bias, position)
        bias_cam = PinholeCamera(
            "bias", calib_dict["fx"], calib_dict["fy"], calib_dict["cx"], calib_dict["cy"],
            width, height, rotation_bias, translation_bias,
        )
        estimated, ok = bias_cam.project_pixels_to_ground(pixel)
        if not ok[0]:
            return 0.0
        return float(estimated[0, 0] - range_m)

    delta_deg = 1.0
    print("extra pitch deg:", f"{delta_deg:.1f}")
    ranges = np.array([10.0, 20.0, 30.0, 40.0])
    errors = np.array([range_error(calib, delta_deg, float(distance)) for distance in ranges])
    print("range_m  error_m  estimated_m")
    for distance, error in zip(ranges, errors):
        print(f"{distance:.0f}  {error:.4f}  {distance + error:.4f}")
    print("abs error ratio 40 m / 10 m:", f"{abs(errors[-1] / errors[0]):.2f}")
    print("distance ratio 40 m / 10 m:", f"{ranges[-1] / ranges[0]:.1f}")
    zero_error = range_error(calib, 0.0, 20.0)
    print("extra pitch 0 at 20 m:", f"{zero_error:.3e}")

    sol_pitch = load_solution("sol_m01_pitch_lesson", "pitch_sensitivity.py")
    ref_40 = sol_pitch.pitch_shift_meters(calib, delta_deg, 40.0)
    print("reference pitch_shift_meters at 40 m:", f"{ref_40:.4f}")
    print("diff vs table at 40 m:", f"{abs(ref_40 - errors[-1]):.3e}")

    keep_inline()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(ranges, errors, marker="o")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("true forward distance (m)")
    ax.set_ylabel("estimated minus true (m)")
    ax.set_title("extra nose-down pitch")
    plt.tight_layout()
    plt.show()
    """))

    cells.append(md("""
    With `0` extra degrees, the error at `20` m is `-1.421e-14`. That is roundoff. The calibration, used on itself, returns the true distance.

    With `1.0` extra degree of nose-down pitch the table is:

    | true forward m | error m | estimated m |
    | --- | --- | --- |
    | `10` | `-0.7478` | `9.2522` |
    | `20` | `-3.3191` | `16.6809` |
    | `30` | `-7.2636` | `22.7364` |
    | `40` | `-12.2325` | `27.7675` |

    The sign is negative: the wrong pitch reads the point as closer than it is. A ray that actually hits the road at `40` m is explained as a road hit at `27.7675` m. The distance grew by a factor of `4.0` (`40 / 10`). The absolute error grew by a factor of `16.36`. A small angle error turns into a miss that scales roughly with distance squared, because you are tilting a long lever. The reference `pitch_shift_meters` agrees with this table at `40` m: difference `0.000e+00`.

    This is why a camera that nods under braking cannot keep a flat-ground map in meters. `1.0` degree is a small motion of the nose. At `40` m the miss is `12.2325` m, which is the difference between "the lead car is far enough" and "it is not."
    """))

    cells.append(md("""
    The second break is the ground itself. The sample draws a lead vehicle as one quad at a single forward distance, with the bottom on the road and the top above it. IPM treats every pixel as a ground pixel. A point on the vehicle that is above the road is shoved to wherever that ray eventually hits `Z = 0`.

    **Predict:** the point on the road under the vehicle stays at its true forward distance. A point partway up the vehicle is placed much farther away. The roof, which is above the camera, does not hit the road in front of the car at all. In the BEV image the vehicle color is smeared forward, not parked on its footprint.
    """))

    cells.append(code("""
    box = np.array(
        [
            [22.0, -0.9, 0.0],
            [22.0, 0.9, 0.0],
            [22.0, 0.9, 1.5],
            [22.0, -0.9, 1.5],
        ],
        dtype=np.float64,
    )
    print("box forward m, all corners:", " ".join(f"{v:.1f}" for v in box[:, 0]))
    print("box lateral m min max:", f"{box[:, 1].min():.1f}", f"{box[:, 1].max():.1f}")
    print("box height m min max:", f"{box[:, 2].min():.1f}", f"{box[:, 2].max():.1f}")
    print("camera height m:", f"{cam_pos[2]:.1f}")

    for height in (0.0, 0.5, 1.0, 1.5):
        point = np.array([[22.0, 0.0, height]])
        pixel, valid = front.project_ego_to_pixel(point)
        estimated, hit = front.project_pixels_to_ground(pixel)
        print(
            f"height m {height:.1f} pixel row {pixel[0, 1]:.2f} "
            f"valid {bool(valid[0])} ground hit {bool(hit[0])}"
        )
        if hit[0]:
            print(
                f"  ground forward m {estimated[0, 0]:.3f} "
                f"shift m {estimated[0, 0] - 22.0:.3f}"
            )

    car = (
        (bev_bgr[:, :, 0] > 100)
        & (bev_bgr[:, :, 0] < 180)
        & (bev_bgr[:, :, 1] < 90)
        & (bev_bgr[:, :, 2] < 70)
    )
    car_image = (
        (bgr[:, :, 0] > 100)
        & (bgr[:, :, 0] < 180)
        & (bgr[:, :, 1] < 90)
        & (bgr[:, :, 2] < 70)
    )
    print("car pixels in the camera image:", int(car_image.sum()))
    print("car pixels in the front BEV:", int(car.sum()))
    car_rows, car_cols = np.where(car)
    car_x = x_range[1] - car_rows * bev_resolution
    car_y = y_range[1] - car_cols * bev_resolution
    print("BEV car forward m min max:", f"{car_x.min():.1f}", f"{car_x.max():.1f}")
    print("BEV car lateral m min max:", f"{car_y.min():.1f}", f"{car_y.max():.1f}")
    """))

    cells.append(md("""
    Every corner of the quad is at forward distance `22.0` m. Laterally it runs from `-0.9` m to `0.9` m. The top is at `1.5` m, and the camera is at `1.4` m, so the roof is above the lens.

    The road point under the vehicle, height `0.0`, maps to pixel row `90.01` and comes back to forward distance `22.000` m. Shift `-0.000` m. Height `0.5` m is pixel row `87.70` and is placed at `33.111` m, a shift of `11.111` m. Height `1.0` m lands at `72.000` m, a shift of `50.000` m, which is already past the `40.0` m edge of this grid. Height `1.5` m is a valid pixel (row `83.08`) but `ground hit` is false: the ray from a camera at `1.4` m through a point at `1.5` m is going up, so it never meets the road ahead.

    The camera image contains `80` vehicle pixels. The front BEV contains `5543` pixels of that color, spread from `20.9` m to `40.0` m forward and from `-1.7` m to `2.1` m laterally. The warp asked each ground cell "which image pixel do you see?" and a lot of distant ground cells see the vehicle, because the vehicle sticks up into their line of sight. The car is smeared along the road out to the far edge of the map. Its true footprint was a single distance, `22.0` m.

    That smear is not a bug in the resampler. It is the flat-ground assumption meeting an object with height. A pedestrian does the same thing. This is why a later module learns a bird's-eye map from the image instead of forcing every ray onto `Z = 0`. The homography remains the right tool for lane paint that really is on the road, and the wrong tool for anything standing on it.
    """))

    cells.append(md("""
    **Exercise — `pitch_shift_meters`.** Given the calibration dict, an extra pitch in degrees, and a true forward distance, return estimated forward distance minus the true distance. Use the same camera model as the table above. Leave the `TODO` as it is to use the reference implementation.

    The check expects a numerical zero when the extra pitch is `0`, and a larger absolute error at `40` m than at `10` m when the extra pitch is `1.0` degree.
    """))

    cells.append(code("""
    def pitch_shift_meters_student(calib_dict, delta_deg_value, range_m):
        # TODO: estimated forward distance minus range_m
        raise NotImplementedError

    def get_pitch_fn():
        try:
            pitch_shift_meters_student(calib, 0.0, 20.0)
        except NotImplementedError:
            print("Using reference pitch_shift_meters (TODO not implemented)")
            return sol_pitch.pitch_shift_meters
        print("Using your pitch_shift_meters")
        return pitch_shift_meters_student

    pitch_fn = get_pitch_fn()
    err_zero = pitch_fn(calib, 0.0, 20.0)
    err_10 = pitch_fn(calib, 1.0, 10.0)
    err_40 = pitch_fn(calib, 1.0, 40.0)
    print("delta 0 deg at 20 m:", f"{err_zero:.3e}")
    print("delta 1 deg at 10 m:", f"{err_10:.4f}")
    print("delta 1 deg at 40 m:", f"{err_40:.4f}")
    assert abs(err_zero) < 1e-2
    assert abs(err_40) > abs(err_10)
    assert abs(err_10 - errors[0]) < 1e-6
    assert abs(err_40 - errors[-1]) < 1e-6
    print("✅ correct")
    """))

    cells.append(md("""
    The check prints `✅`. Extra pitch `0` at `20` m returns `-1.421e-14`. Extra pitch `1.0` degree returns `-0.7478` m at `10` m and `-12.2325` m at `40` m, the same numbers as the table. The reference function is used because the `TODO` still raises.
    """))

    cells.append(md("""
    <details><summary>Solution</summary>

    ```python
    def pitch_shift_meters(calib, delta_deg, range_m):
        width = int(calib["width"])
        height = int(calib["height"])
        position = np.array(calib["cam_position_ego"], dtype=np.float64)
        pitch = float(calib["pitch_deg"])
        yaw = float(calib["yaw_deg"])
        roll = float(calib["roll_deg"])
        rotation_true = create_euler_rotation(pitch, yaw, roll)
        translation_true = camera_position_to_translation(rotation_true, position)
        true_cam = PinholeCamera(
            "true", calib["fx"], calib["fy"], calib["cx"], calib["cy"],
            width, height, rotation_true, translation_true,
        )
        pixel, valid = true_cam.project_ego_to_pixel(np.array([[range_m, 0.0, 0.0]]))
        if not valid[0]:
            return 0.0
        rotation_bias = create_euler_rotation(pitch + delta_deg, yaw, roll)
        translation_bias = camera_position_to_translation(rotation_bias, position)
        bias_cam = PinholeCamera(
            "bias", calib["fx"], calib["fy"], calib["cx"], calib["cy"],
            width, height, rotation_bias, translation_bias,
        )
        estimated, ok = bias_cam.project_pixels_to_ground(pixel)
        if not ok[0]:
            return 0.0
        return float(estimated[0, 0] - range_m)
    ```

    </details>
    """))

    cells.append(md("## 10. Recap"))
    cells.append(md("""
    - A pinhole divides by depth. With focal length `100.0` and lateral offset `2.0` m, the pixel offset is `20.0000` at `10` m, `10.0000` at `20` m, and `5.0000` at `40` m.
    - `K` holds `fx`, `fy`, `cx`, and `cy`. Extrinsics take an ego point to the camera with `P_cam = R @ P_ego + T`. The ego frame is `X` forward, `Y` left, `Z` up. The camera frame is `X` right, `Y` down, `Z` forward.
    - Stretching the frame to height `360` left the lanes meeting at column `159.5000`. On the ground plane, `H = K [r1 r2 t]` maps meters to pixels and back. The round trip on the `10` m point errs by `5.623e-15` m.
    - The two solid lanes, `11.2` m apart, are columns `44` and `156` in the BEV for every forward distance. In the image their gap shrinks from `170.08` pixels at `8.0` m to `36.91` pixels at `30.0` m. Three cameras cover the grid; the front camera alone leaves `2292` near, wide cells black.
    - A `1.0` degree pitch error reads `40` m as `27.7675` m (error `-12.2325` m), and the miss grows faster than range (`16.36` times, against a `4.0` times increase in distance). A vehicle point `0.5` m off the ground at `22.0` m is placed at `33.111` m. Flat-ground IPM is the right map for paint on the road and the wrong map for anything with height.

    ### Go deeper

    - [Shree Nayar — Pinhole and perspective projection](https://www.youtube.com/watch?v=_EhY31MSbNM), and the two camera-matrix lectures [Linear camera model](https://www.youtube.com/watch?v=qByYk6JggQU) and [Intrinsic and extrinsic matrices](https://www.youtube.com/watch?v=2XM2Rb2pfyQ), from [First Principles of Computer Vision](https://fpcv.cs.columbia.edu/)
    - [Stanford CS231A — Camera models notes](https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf)
    - [3Blue1Brown — Essence of linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
    - [OpenCV — Camera calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
    """))

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nb.metadata["colab"] = {"provenance": []}
    return nb


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "notebooks" / "01_cameras_and_ipm.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    nbformat.write(nb, out)
    print(f"Wrote {out} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
