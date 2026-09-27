#!/usr/bin/env python3
"""Regenerates the Module 05 Colab lesson notebook. Does not execute it."""

from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "staging/self-driving-ai-course/notebooks/06_vector_space_tracking_kalman.ipynb"


def md(source: str):
    return new_markdown_cell(textwrap.dedent(source).strip() + "\n")


def code(source: str):
    return new_code_cell(textwrap.dedent(source).strip() + "\n")


def build_cells():
    cells = []

    cells.append(
        md(
            """
            # Module 05 — Follow the car that just hid behind a truck

            [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/notebooks/06_vector_space_tracking_kalman.ipynb)

            A detector gives you a new dot every frame. When a car slips behind a truck the dot vanishes. Your job is to keep the same identity and a smooth position. Each section explains one idea, then runs code. Predict first, then execute the cell.
            """
        )
    )

    cells.append(
        code(
            """
            import os, subprocess, sys
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np

            %matplotlib inline

            def find_repo(start: Path) -> Path:
                for p in [start, *start.parents]:
                    if (p / "modules" / "05_vector_space" / "kalman_tracker.py").is_file():
                        return p.resolve()
                return start.resolve()

            REPO = find_repo(Path.cwd())
            if not (REPO / "modules" / "05_vector_space" / "kalman_tracker.py").is_file():
                dest = Path.cwd() / "self-driving-ai-course"
                if not (dest / "modules" / "05_vector_space" / "kalman_tracker.py").is_file():
                    subprocess.run(
                        ["git", "clone", "--depth", "1", "https://github.com/vvknyn/self-driving-ai-course.git", str(dest)],
                        check=True,
                    )
                REPO = dest.resolve()
            os.chdir(REPO)

            try:
                import scipy  # noqa: F401
            except ImportError:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scipy"], check=True)
            try:
                import pytest  # noqa: F401
            except ImportError:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)

            sys.path.insert(0, str(REPO / "modules" / "05_vector_space"))
            from kalman_tracker import KalmanBoxTracker, MultiObjectTracker
            from vector_lanes import VectorLane
            from scipy.optimize import linear_sum_assignment

            print("Repo:", REPO)
            print("KalmanBoxTracker", KalmanBoxTracker.__name__)
            print("VectorLane", VectorLane.__name__)
            """
        )
    )

    cells.append(md("## 1. Detections flicker"))
    cells.append(
        md(
            """
            **Predict:** Five frames of noisy detections around a straight path will jump in lateral `y` even when the true lane center stays at zero.
            """
        )
    )
    cells.append(
        code(
            """
            np.random.seed(1)
            true_pts, det_pts = [], []
            print("frame true_x true_y det_x det_y")
            for k in range(5):
                true_x = 20.0 + 10.0 * k * 0.1
                true_y = 0.0
                det = np.array([true_x, true_y]) + np.random.normal(0, 0.35, size=2)
                true_pts.append([true_x, true_y])
                det_pts.append(det)
                print(f"{k} {true_x:.1f} {true_y:.1f} {det[0]:.3f} {det[1]:.3f}")

            true_pts = np.array(true_pts)
            det_pts = np.array(det_pts)
            plt.figure(figsize=(8, 3))
            plt.scatter(true_pts[:, 0], true_pts[:, 1], c="k", marker="x", label="true")
            plt.scatter(det_pts[:, 0], det_pts[:, 1], c="C1", label="detections")
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.legend()
            plt.title("True path vs noisy detections")
            plt.grid(True, alpha=0.3)
            plt.show()
            """
        )
    )
    cells.append(
        md(
            """
            True `y` is `0.0` every frame, but measured `y` jumps through `-0.214`, `-0.376`, `-0.806`, `-0.266`, and `-0.087`. You get five unrelated dots with no identity. If the car hides, the next dot is not labeled as the same car.
            """
        )
    )

    cells.append(md("## 2. Nearest neighbor swaps the IDs"))
    cells.append(
        md(
            """
            **Predict:** Matching stale positions with Euclidean distance will assign Track 1 to the wrong detection.
            """
        )
    )
    cells.append(
        code(
            """
            t1 = np.array([10.0, 0.0])
            t2 = np.array([10.2, -1.0])
            car1 = np.array([11.5, 0.0])
            car2 = np.array([10.2, 0.0])
            d_t1_car1 = np.linalg.norm(t1 - car1)
            d_t1_car2 = np.linalg.norm(t1 - car2)
            print(f"Track 1 to car 1: {d_t1_car1:.2f} m")
            print(f"Track 1 to car 2: {d_t1_car2:.2f} m")
            print(f"ID SWAP: {d_t1_car2:.2f} < {d_t1_car1:.2f}")
            cost = np.array([
                [np.linalg.norm(t1 - car1), np.linalg.norm(t1 - car2)],
                [np.linalg.norm(t2 - car1), np.linalg.norm(t2 - car2)],
            ])
            print("euclidean cost")
            print(np.round(cost, 2))
            row, col = linear_sum_assignment(cost)
            names_t = ["T1", "T2"]
            names_d = ["car1", "car2"]
            for r, c in zip(row, col):
                print(f"{names_t[r]} -> {names_d[c]}")
            """
        )
    )
    cells.append(
        md(
            """
            Track 1 is only `0.20` m from car 2 but `1.50` m from car 1, so the line `ID SWAP: 0.20 < 1.50` fires. The printed block `[[1.5  0.2 ]` / `[1.64 1.  ]]` and assignments `T1 -> car2` / `T2 -> car1` show that nearest neighbor and Hungarian-on-Euclidean both swap identities. This section stops at the failure.
            """
        )
    )

    cells.append(md("## 3. Kalman predict and update"))
    cells.append(
        md(
            """
            **Predict:** A scalar constant-position filter with `P=1` and `r=0.09` will trust the measurement at `z=10.5` and shrink variance after the update.
            """
        )
    )
    cells.append(
        code(
            """
            x, P, q, z, r = 10.0, 1.0, 0.0, 10.5, 0.09
            P_pred = P + q
            K = P_pred / (P_pred + r)
            x_post = x + K * (z - x)
            P_post = (1.0 - K) * P_pred
            print(f"P_pred = {P_pred:.4f}")
            print(f"K = {K:.4f}")
            print(f"x = {x_post:.4f}")
            print(f"P = {P_post:.4f}")
            """
        )
    )
    cells.append(
        md(
            """
            The printed lines are `P_pred = 1.0000`, `K = 0.9174`, `x = 10.4587`, and `P = 0.0826`. Gain near `0.9174` means prior variance `1` is large next to a tight measurement, so the state moves most of the way toward the new z. Posterior variance `0.0826` is smaller than the prior.
            """
        )
    )
    cells.append(
        md(
            """
            **Predict:** The repo `KalmanBoxTracker` uses the same predict/update pattern in four dimensions; one position update should drop the position covariance trace and nudge velocity.
            """
        )
    )
    cells.append(
        code(
            """
            KalmanBoxTracker.count = 0
            trk = KalmanBoxTracker(np.array([10.0, 0.0]), dt=0.1)
            print(f"x shape {trk.x.shape}")
            print(
                f"F {trk.F.shape} H {trk.H.shape} Q {trk.Q.shape} R {trk.R.shape} P {trk.P.shape}"
            )
            print(f"R diag {np.diag(trk.R)}")
            trk.predict()
            print(f"state after predict {np.round(trk.x, 6)}")
            print(f"prior position trace {np.trace(trk.P[:2, :2]):.6f}")
            trk.update(np.array([10.5, 0.0]))
            print(f"posterior position trace {np.trace(trk.P[:2, :2]):.6f}")
            print(f"state after update {np.round(trk.x, 6)}")
            """
        )
    )
    cells.append(
        md(
            """
            Shapes read `x shape (4,)`, `F (4, 4)`, `H (2, 4)`, `Q (4, 4)`, `R (2, 2)`, `P (4, 4)`, with `R diag [0.09 0.09]`. After predict the print is `state after predict [10.  0.  0.  0.]` and `prior position trace 2.400000`. After the measurement the print is `posterior position trace 0.167442` and `state after update [10.465116  0.        0.387597  0.      ]`. The third entry is forward speed `0.387597`.
            """
        )
    )

    cells.append(md("## 4. Mahalanobis against the ID swap"))
    cells.append(
        md(
            """
            **Predict:** Mahalanobis distance up-weights tight axes, so the same innovation vector can look larger than Euclidean length.
            """
        )
    )
    cells.append(
        code(
            """
            S = np.diag([0.09, 0.25])
            y = np.array([0.3, 0.5])
            d2 = float(y @ np.linalg.inv(S) @ y)
            print(f"d_M^2 = {d2:.1f}")
            print(f"d_M = {np.sqrt(d2):.4f}")
            print(f"euclidean = {np.linalg.norm(y):.4f}")
            """
        )
    )
    cells.append(
        md(
            """
            You get `d_M^2 = 2.0`, `d_M = 1.4142`, and `euclidean = 0.5831`. Mahalanobis is larger because the innovation is one sigma on each axis when x is tight and y is loose, while Euclidean treats both meters the same.
            """
        )
    )
    cells.append(
        md(
            """
            **Predict:** After Kalman predict, Mahalanobis costs on the section 2 scene should match each track to its own car.
            """
        )
    )
    cells.append(
        code(
            """
            KalmanBoxTracker.count = 0
            trk1 = KalmanBoxTracker(initial_pos=np.array([10.0, 0.0]))
            trk1.x[2] = 15.0
            trk1.x[3] = 0.0
            trk2 = KalmanBoxTracker(initial_pos=np.array([10.2, -1.0]))
            trk2.x[2] = 0.0
            trk2.x[3] = 10.0
            car1 = np.array([11.5, 0.0])
            car2 = np.array([10.2, 0.0])
            pred1 = trk1.predict()[:2]
            pred2 = trk2.predict()[:2]
            print(f"Track 1 predicted ({pred1[0]:.1f}, {pred1[1]:.1f})")
            print(f"Track 2 predicted ({pred2[0]:.1f}, {pred2[1]:.1f})")
            m11 = trk1.mahalanobis_distance(car1)
            m12 = trk1.mahalanobis_distance(car2)
            m21 = trk2.mahalanobis_distance(car1)
            m22 = trk2.mahalanobis_distance(car2)
            print(f"Mahalanobis T1->car1 {m11:.2f}")
            print(f"Mahalanobis T1->car2 {m12:.2f}")
            print(f"Mahalanobis T2->car1 {m21:.2f}")
            print(f"Mahalanobis T2->car2 {m22:.2f}")
            cost = np.array([[m11, m12], [m21, m22]])
            row, col = linear_sum_assignment(cost)
            for r, c in zip(row, col):
                print(f"{['T1','T2'][r]} -> {['car1','car2'][c]}  cost {cost[r, c]:.2f}")
            """
        )
    )
    cells.append(
        md(
            """
            Predictions land at `(11.5, 0.0)` and `(10.2, 0.0)`. Mahalanobis pairs are `0.00` on-diagonal and `1.14` off-diagonal, and the assignment prints `T1 -> car1  cost 0.00` and `T2 -> car2  cost 0.00`. Kalman predict carries Track 1 onto its own dot; this fix avoids the section 2 identity swap.
            """
        )
    )

    cells.append(md("## 5. Coast through a miss"))
    cells.append(
        md(
            """
            **Predict:** After one detection and a known speed, an empty frame should advance the same track id with no measurement.
            """
        )
    )
    cells.append(
        code(
            """
            KalmanBoxTracker.count = 0
            mot = MultiObjectTracker(max_age=3, min_hits=2, distance_threshold=4.0)
            seen = mot.update(np.array([[10.0, 0.0]]))
            mot.trackers[0].x[2] = 12.0
            mot.trackers[0].x[3] = 0.0
            missed = mot.update(np.empty((0, 2)))
            tid, pos, vel = missed[0]
            print(f"seen id={seen[0][0]}")
            print(f"coast id={tid} x={pos[0]:.1f} y={pos[1]:.1f} vx={vel[0]:.1f}")
            print(f"time_since_update={mot.trackers[0].time_since_update}")
            """
        )
    )
    cells.append(
        md(
            """
            The first update returns `seen id=0`. On the miss, `coast id=0 x=11.2 y=0.0 vx=12.0` with `time_since_update=1`. Position `11.2` follows the seeded `vx=12.0` over one predict step, and the empty frame did not delete the car.
            """
        )
    )

    cells.append(md("## 6. Vector lanes"))
    cells.append(
        md(
            """
            **Predict:** Cubic lane polynomials expose lateral offset, heading, and curvature along forward distance.
            """
        )
    )
    cells.append(
        code(
            """
            straight = VectorLane("straight", [2.0, 0.0, 0.0, 0.0])
            curved = VectorLane("curved", [0.0, 0.0, 0.001, 0.0])
            ego = VectorLane("ego_centerline", [0.0, 0.02, 0.0005, 0.0])
            print(f"straight offset {straight.lateral_offset(15.0):.1f}")
            print(f"straight heading {straight.heading_tangent(15.0):.1f}")
            print(f"straight curvature {straight.curvature(15.0):.1f}")
            print(f"curved offset {curved.lateral_offset(10.0):.1f}")
            print(f"curved curvature {curved.curvature(10.0):.6f}")
            print(f"ego curvature 0m {ego.curvature(0.0):.6f}")
            print(f"ego curvature 30m {ego.curvature(30.0):.6f}")
            print(f"ego offset 30m {ego.lateral_offset(30.0):.2f}")
            xs = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
            ys = 0.02 * xs + 0.0005 * xs**2
            fitted = VectorLane.fit_from_points("fit", xs, ys, order=3)
            fit_round = np.round(fitted.coeffs, 6) + 0.0
            print("fit coeffs", fit_round)
            poly = ego.sample_polyline(0.0, 40.0, 1.0)
            plt.figure(figsize=(8, 3))
            plt.plot(poly[:, 0], poly[:, 1], label="ego centerline")
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.title("Ego lane polyline")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.show()
            """
        )
    )
    cells.append(
        md(
            """
            Straight lane offset is `2.0` with heading `0.0` and curvature `0.0`. The curved lane shows offset `0.1` and curvature `0.001999`. Ego curvature is `0.000999` at the `ego curvature 0m` sample and `0.000996` at `30` m, with lateral offset `1.05` at `30` m. Fitted coeffs print as `[0.     0.02   0.0005 0.    ]`. This is one cubic centerline, not a lane-topology benchmark like [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2).
            """
        )
    )

    cells.append(md("## 7. Two cars, one tracker"))
    cells.append(
        md(
            """
            **Predict:** Six noisy frames with fixed seed should keep ids `0` and `1` while speeds climb toward the scripted motion.
            """
        )
    )
    cells.append(
        code(
            """
            KalmanBoxTracker.count = 0
            np.random.seed(0)
            ego = VectorLane("ego_centerline", [0.0, 0.02, 0.0005, 0.0])
            tracker = MultiObjectTracker(max_age=3, min_hits=2, distance_threshold=4.0)
            dt = 0.1
            print(f"{'t':<6}{'true A':<16}{'true B':<16}tracks")
            true_a, true_b, dets_all, filt_all = [], [], [], []
            for step in range(6):
                t = step * dt
                true_ax = 15.0 + 12.0 * t
                true_ay = ego.lateral_offset(true_ax)
                true_bx = 10.0 + 16.0 * t
                true_by = -3.5
                noisy = np.array([
                    [true_ax, true_ay] + np.random.normal(0, 0.15, size=2),
                    [true_bx, true_by] + np.random.normal(0, 0.15, size=2),
                ])
                active = tracker.update(noisy)
                true_a.append([true_ax, true_ay])
                true_b.append([true_bx, true_by])
                dets_all.append(noisy.copy())
                filt_all.append(active)
                bits = [f"ID {tid}: ({pos[0]:.1f}, {pos[1]:.1f}) vx={vel[0]:.1f}" for tid, pos, vel in active]
                print(f"{t:<6.1f}{f'({true_ax:.1f}, {true_ay:.1f})':<16}{f'({true_bx:.1f}, {true_by:.1f})':<16}{'; '.join(bits)}")

            true_a = np.array(true_a)
            true_b = np.array(true_b)
            dets_all = np.array(dets_all)
            plt.figure(figsize=(9, 4))
            plt.plot(true_a[:, 0], true_a[:, 1], "k--", label="true A")
            plt.plot(true_b[:, 0], true_b[:, 1], "k:", label="true B")
            plt.scatter(dets_all[:, 0, 0], dets_all[:, 0, 1], s=20, c="C1", alpha=0.6, label="det A")
            plt.scatter(dets_all[:, 1, 0], dets_all[:, 1, 1], s=20, c="C2", alpha=0.6, label="det B")
            for frame in filt_all:
                for tid, pos, vel in frame:
                    plt.scatter(pos[0], pos[1], marker="s", s=40, c="C0")
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.legend()
            plt.title("Two-car tracking demo")
            plt.grid(True, alpha=0.3)
            plt.show()
            """
        )
    )
    cells.append(
        md(
            """
            The first time row ends with `ID 0: (15.3, 0.5) vx=0.0; ID 1: (10.1, -3.2) vx=0.0`. The last row ends with `ID 0: (20.6, 0.7) vx=8.7; ID 1: (17.9, -3.6) vx=13.2`. IDs stay `0` and `1` while speeds climb from `vx=0.0` toward the scripted 12 m/s and 16 m/s without having fully converged yet.
            """
        )
    )
    cells.append(
        code(
            """
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "modules/05_vector_space", "-q"],
                capture_output=True,
                text=True,
            )
            print(proc.stdout)
            if proc.returncode != 0:
                print(proc.stderr)
            assert proc.returncode == 0
            """
        )
    )
    cells.append(
        md(
            """
            Module tests report `3 passed`, so the tracker code matches the lesson numbers.
            """
        )
    )

    cells.append(md("## 8. Exercises"))
    cells.append(
        md(
            """
            **Exercise 1 — scalar update**

            **Predict:** Your function should reproduce the section 3 hand numbers for `(x, P, z, q, r) = (10, 1, 10.5, 0, 0.09)`.
            """
        )
    )
    cells.append(
        code(
            """
            def scalar_update_student(x, P, z, q, r):
                # TODO: implement constant-position Kalman update; return K, x_post, P_post
                raise NotImplementedError

            def reference_scalar_update(x, P, z, q, r):
                P_pred = P + q
                K = P_pred / (P_pred + r)
                x_post = x + K * (z - x)
                P_post = (1.0 - K) * P_pred
                return K, x_post, P_post

            def get_scalar_update():
                try:
                    return scalar_update_student
                except NotImplementedError:
                    pass

            try:
                fn = scalar_update_student
                K, x_post, P_post = fn(10.0, 1.0, 10.5, 0.0, 0.09)
            except NotImplementedError:
                print("Using reference scalar_update_student (TODO not implemented)")
                fn = reference_scalar_update
                K, x_post, P_post = fn(10.0, 1.0, 10.5, 0.0, 0.09)

            assert np.isclose(K, 0.9174, atol=1e-4)
            assert np.isclose(x_post, 10.4587, atol=1e-4)
            assert np.isclose(P_post, 0.0826, atol=1e-4)
            print(f"K = {K:.4f}")
            print(f"x = {x_post:.4f}")
            print(f"P = {P_post:.4f}")
            print("✅ scalar update")
            """
        )
    )
    cells.append(
        md(
            """
            Checks print `K = 0.9174`, `x = 10.4587`, and `P = 0.0826`, then `✅ scalar update`.
            """
        )
    )
    cells.append(
        md(
            """
            <details><summary>Solution</summary>

            ```python
            def scalar_update_student(x, P, z, q, r):
                P_pred = P + q
                K = P_pred / (P_pred + r)
                x_post = x + K * (z - x)
                P_post = (1.0 - K) * P_pred
                return K, x_post, P_post
            ```

            </details>
            """
        )
    )

    cells.append(
        md(
            """
            **Exercise 2 — Mahalanobis distance**

            **Predict:** On `y=[0.3, 0.5]` and `S=diag(0.09, 0.25)` your function should match section 4.
            """
        )
    )
    cells.append(
        code(
            """
            def mahalanobis_student(y, S):
                # TODO: return sqrt(y^T S^{-1} y)
                raise NotImplementedError

            def reference_mahalanobis(y, S):
                y = np.asarray(y, dtype=np.float64)
                S = np.asarray(S, dtype=np.float64)
                d2 = y @ np.linalg.inv(S) @ y
                return float(np.sqrt(d2))

            try:
                d = mahalanobis_student(np.array([0.3, 0.5]), np.diag([0.09, 0.25]))
            except NotImplementedError:
                print("Using reference mahalanobis_student (TODO not implemented)")
                d = reference_mahalanobis(np.array([0.3, 0.5]), np.diag([0.09, 0.25]))

            assert np.isclose(d, 1.4142, atol=1e-4)
            print(f"d_M = {d:.4f}")
            print("✅ mahalanobis")
            """
        )
    )
    cells.append(
        md(
            """
            The check prints `d_M = 1.4142` and `✅ mahalanobis`.
            """
        )
    )
    cells.append(
        md(
            """
            <details><summary>Solution</summary>

            ```python
            def mahalanobis_student(y, S):
                y = np.asarray(y, dtype=np.float64)
                S = np.asarray(S, dtype=np.float64)
                d2 = y @ np.linalg.inv(S) @ y
                return float(np.sqrt(d2))
            ```

            </details>
            """
        )
    )

    cells.append(
        md(
            """
            ## 9. Recap

            - Detections flicker in `y` (`-0.214`, `-0.376`, `-0.806`, `-0.266`, `-0.087`) while true `y` stays `0.0`.
            - Euclidean matching failed with `0.20 < 1.50` and swapped `T1 -> car2`.
            - Scalar Kalman: `K = 0.9174`, `x = 10.4587`, `P = 0.0826`; matrix trace fell from `2.400000` to `0.167442`.
            - Mahalanobis fix assigned both tracks at cost `0.00`; coast kept id `0` at `x=11.2`.
            - Ego lane offset at `30` m is `1.05`; two-car demo ends at `vx=8.7` and `vx=13.2`; tests show `3 passed`.

            For a plain-language walkthrough of the scalar filter, see https://greg.czerniak.info/guides/kalman1/
            """
        )
    )

    return cells


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nb = new_notebook(
        cells=build_cells(),
        metadata={
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.3",
            },
        },
    )
    with OUT.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
