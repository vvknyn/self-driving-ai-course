# Module 01: Multi-Camera Geometry & Inverse Perspective Mapping (IPM)

> "To understand 3D from 2D, you must master the coordinate transforms between photons, sensors, and the metric physical ground."

---

## 🟢 Tier 1: Intuition & Mental Models (Andrew Ng Style)

### The Fundamental Dilemma of 2D Vision
When light hits an automotive camera sensor, the 3D physical world is projected onto a flat 2D plane of pixels. In doing so, **one entire spatial dimension—depth along the optical ray—is completely lost**.
- An identical $50 \times 50$ pixel patch on an image could represent a $1.8$-meter pedestrian 50 meters away, or an $18$-centimeter smartphone 5 meters away!
- A self-driving vehicle cannot steer or brake based on pixels. A trajectory planner operates in **metric 3D Cartesian space** (meters, seconds, meters/second).

### What is Inverse Perspective Mapping (IPM)?
If we make one simplifying assumption—that the road surface ahead is a **flat, horizontal plane** ($Z_{\text{road}} = 0$)—the mathematical projection becomes a bijective (1-to-1) mapping.
We can computationally "unproject" pixels from the perspective camera image and project them onto the top-down ground plane. This is called **Inverse Perspective Mapping (IPM)** or Bird's-Eye View (BEV) warping.

```
       PERSPECTIVE CAMERA VIEW                     TOP-DOWN BIRD'S-EYE VIEW (IPM)
      ┌───────────────────────────┐                 ┌───────────────────────────┐
      │         \   |   /         │                 │         |       |         │
      │          \  |  /          │                 │         |       |         │
      │           \ | /           │   Homography H  │         |  Ego  |         │
      │   Road     \|/   Sky      │ ──────────────> │         |  Lane |         │
      │  Lines      V   Horizon   │                 │         |       |         │
      │            / \            │                 │         |       |         │
      │           /   \           │                 │         |       |         │
      └───────────────────────────┘                 └───────────────────────────┘
       Lines converge at horizon                     Lines are parallel & metric!
```

### Andrew Ng Diagnostic Table: Camera Geometry Failure Modes
When deploying camera geometry on physical vehicles, things go wrong. Use this diagnostic table:

| Symptom | Root Cause | Diagnostic Test | Solution |
|---|---|---|---|
| Lane lines flare outwards at distance | Camera pitch angle calibrated too high | Inspect horizon line in camera feed | Adjust pitch extrinsic parameter down |
| Lane lines converge at distance | Camera pitch angle calibrated too low | Calculate road intersection distance | Adjust pitch extrinsic parameter up |
| Obstacles appear stretched like giant smears | 3D object violates flat ground assumption | Check if bounding box extends above road | Use 3D Occupancy / LSS instead of planar IPM |
| Severe distance error during hard braking | Chassis dynamic pitch deflection ($\Delta \theta \approx 2^\circ$) | Log IMU pitch gyro during braking | Implement dynamic pitch compensation network |

---

## 🟡 Tier 2: Code From Scratch (Andrej Karpathy Style)

Let's spell out the exact tensor mechanics. No OpenCV black boxes—just pure linear algebra.

### 1. Generating Intrinsic Matrix $K$
Given camera sensor width $W$, height $H$, and horizontal field of view $\text{HFOV}$:
```python
import numpy as np

def compute_intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Computes the 3x3 pinhole intrinsic camera matrix K."""
    hfov_rad = np.deg2rad(hfov_deg)
    # Focal length in pixels: fx = (W/2) / tan(HFOV/2)
    fx = (width / 2.0) / np.tan(hfov_rad / 2.0)
    fy = fx  # Square pixels assumption
    cx = width / 2.0
    cy = height / 2.0
    
    K = np.array([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    return K
```

### 2. Homography Unprojection from Scratch
Given intrinsic matrix $K$, rotation matrix $R$, and camera translation $T = [X, Y, Z]^T$:
```python
def compute_ipm_homography(K: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Computes the 3x3 ground-plane planar homography H.
    Maps ego-ground coordinates [X_ego, Y_ego, 1]^T to pixel coordinates [s*u, s*v, s]^T.
    """
    # In ego space, Z_ego = 0 for road plane.
    # Extrinsic matrix [R | T] drops its 3rd column!
    r1 = R[:, 0:1] # 1st column of rotation
    r2 = R[:, 1:2] # 2nd column of rotation
    t  = T.reshape(3, 1)
    
    # Planar extrinsic: 3x3 matrix
    M_planar = np.hstack([r1, r2, t])
    
    # Homography H = K * [r1, r2, t]
    H = K @ M_planar
    return H

def project_pixel_to_ground(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    """Inverse maps a single pixel (u, v) back to physical ground coordinate (X_ego, Y_ego)."""
    H_inv = np.linalg.inv(H)
    pixel_homogeneous = np.array([u, v, 1.0], dtype=np.float32)
    ground_homogeneous = H_inv @ pixel_homogeneous
    
    # Normalize by scale factor (3rd homogeneous component)
    X_ego = ground_homogeneous[0] / ground_homogeneous[2]
    Y_ego = ground_homogeneous[1] / ground_homogeneous[2]
    return float(X_ego), float(Y_ego)
```

---

## 🔴 Tier 3: Mathematical Derivations & Proofs (Thrun / MITx)

### 1. Pinhole Projection from Similar Triangles
- **Starting Point**: A point in 3D camera coordinates $P_c = [X_c, Y_c, Z_c]^T$ emits a light ray through an aperture at $(0, 0, 0)$ hitting the sensor plane at focal distance $f$.
- **Proof**:
  By similar triangles between the optical axis ($Z_c$) and horizontal sensor plane ($X_c$):
  $$\frac{x_{\text{sensor}}}{f} = \frac{X_c}{Z_c} \implies x_{\text{sensor}} = f \frac{X_c}{Z_c}$$
  Converting physical metric sensor coordinates (meters) to image pixel indices $(u, v)$ with pixel pitch $(s_x, s_y)$ pixels/meter and optical center $(c_x, c_y)$:
  $$u = s_x x_{\text{sensor}} + c_x = (s_x f) \frac{X_c}{Z_c} + c_x = f_x \frac{X_c}{Z_c} + c_x$$
  $$v = s_y y_{\text{sensor}} + c_y = (s_y f) \frac{Y_c}{Z_c} + c_y = f_y \frac{Y_c}{Z_c} + c_y$$
  In homogeneous coordinates:
  $$\begin{bmatrix} s \cdot u \\ s \cdot v \\ s \end{bmatrix} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = K \cdot P_c$$

### 2. Planar Homography Invertibility Proof
- **Theorem**: The 3D-to-2D projection of points lying on a plane $\Pi$ into a pinhole camera is a projectivity (homography) $H \in \mathbb{R}^{3 \times 3}$, and is invertible if the camera center does not lie on $\Pi$.
- **Proof**:
  Let the road plane in ego coordinates be defined by $Z_{\text{ego}} = 0$.
  The general camera projection is:
  $$\begin{bmatrix} s \cdot u \\ s \cdot v \\ s \end{bmatrix} = K [R \mid T] \begin{bmatrix} X_e \\ Y_e \\ Z_e \\ 1 \end{bmatrix} = K \begin{bmatrix} r_1 & r_2 & r_3 & T \end{bmatrix} \begin{bmatrix} X_e \\ Y_e \\ 0 \\ 1 \end{bmatrix} = K \begin{bmatrix} r_1 & r_2 & T \end{bmatrix} \begin{bmatrix} X_e \\ Y_e \\ 1 \end{bmatrix}$$
  Let $H = K [r_1 \mid r_2 \mid T]$. Since $K$ is non-singular ($\det(K) = f_x f_y \neq 0$), and $[r_1 \mid r_2]$ are orthonormal columns of a rotation matrix with camera height $T_z = h > 0$, the columns of $[r_1 \mid r_2 \mid T]$ are linearly independent.
  Thus, $\det(H) \neq 0$. Therefore, $H^{-1}$ exists uniquely, allowing exact 2D planar unprojection:
  $$\begin{bmatrix} X_e \\ Y_e \\ 1 \end{bmatrix} = H^{-1} \begin{bmatrix} s \cdot u \\ s \cdot v \\ s \end{bmatrix}$$

---

## 🎓 Tier 4: Cutting-Edge Research & PhD Track

### 1. The Death of Handcrafted Calibration: Online Self-Calibration Networks
In real driving, camera extrinsics drift continuously due to:
1. **Dynamic load changes**: Passengers getting in and out change suspension pitch by $\pm 1.5^\circ$.
2. **Thermal expansion**: Summer heat versus winter cold warps camera mounting brackets on the windshield.
3. **Pavement vibrations**: High-frequency chatter degrades extrinsic alignment.

**Modern Literature Solutions**:
- **Extrinsic Calibration Networks**: Neural networks trained to predict roll, pitch, and yaw perturbations $[\Delta \phi, \Delta \theta, \Delta \psi]$ dynamically from video streams by minimizing photometric reprojection error across overlapping cameras.
- **BARF: Bundle-Adjusting Neural Radiance Fields** (Lin et al., ICCV 2021) and **SC-NeRF**: Jointly optimizes neural scene representations and camera calibration parameters directly from raw video without checkerboard targets.

### 2. Open PhD Research Questions
- *How can multi-camera temporal networks maintain metric scale consistency when driving down steep $15\%$ mountain gradients where the flat road plane assumption completely collapses?*
- *Can we formulate a continuous Lie Algebra $\mathfrak{se}(3)$ Kalman filter that estimates dynamic chassis deflection simultaneously with ego-motion velocity?*

---

## 🟣 Tier 5: Real-World Hardware & Practical Robotics

### Calibrating Your Own $20 Desk Webcam
You do not need industrial LiDAR or a \$100,000 sensor rig. To calibrate any USB webcam:
1. Print a standard $9 \times 6$ checkerboard pattern on regular A4 paper.
2. Mount the pattern on flat cardboard.
3. Capture 15 images from varying angles using OpenCV:
```python
import cv2
import numpy as np

# Find chessboard corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.025  # 25mm square size

# Calibrate camera to obtain K and distortion coefficients [k1, k2, p1, p2, k3]
# ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
```
4. Feed the resulting $K$ matrix directly into `modules/01_camera_geometry/camera_model.py`!
