# Chapter 02: 3D Camera Rig & Inverse Perspective Mapping (IPM)

> **The Big Question**: *A camera sensor is a flat 2D rectangle of pixels, but an autonomous vehicle operates in a 3D physical world of meters and seconds. When photons strike a camera pixel, the depth dimension along that ray is destroyed. How do we mathematically reverse this optical projection to construct a top-down ground map—and why does a tiny $1.5^\circ$ vehicle suspension bounce cause standard geometry to hallucinate that the road has vanished?*

---

## 1. 🚨 The Real-World Dilemma: The Speed Bump Illusion

Imagine driving an autonomous car at 35 mph. You tap the brakes before a speed bump:
- The car's front springs compress, causing the chassis to pitch downward by just **$1.8^\circ$**.
- To human eyes, you barely feel the subtle tilt.
- But to a naive computer vision system assuming a fixed camera:
  - The horizon line in the image drops.
  - A painted crosswalk 30 meters ahead appears to suddenly jump **12 meters closer**!
  - The parallel lane lines violently flare outward into a wide trumpet shape.

```
Vehicle Chassis Pitch (θ = 1.8° tilt):
      Camera tilted down
            \
             \ Ray hits ground MUCH closer than expected!
══════════════\═══════════*───────────────────────── Road Surface
                          ▲
                    Calculated: 18m
                    Actual:     30m (40% Distance Error!)
```

> [!CAUTION]
> **The Sensitivity of Optical Projection**  
> At 60 meters distance, an angular calibration error of just $0.5^\circ$ translates to a **$9.4\text{-meter}$ positioning error** in 3D space! If your car trusts uncalibrated camera geometry to plan braking trajectories, it will either stop 10 meters too early or crash into the stopped vehicle ahead.

---

## 2. 💡 The Mental Model: Pinhole Geometry & The Flat Ground Assumption

### The Pinhole Model: Division by Depth
In optical physics, light passes through an aperture and strikes a sensor plane:

$$u = f_x \frac{X_c}{Z_c} + c_x, \quad v = f_y \frac{Y_c}{Z_c} + c_y$$

Look closely at the denominator: **$Z_c$**.
Every pixel $(u, v)$ is divided by its depth $Z_c$. This means an infinite number of 3D points along the sightline project to the exact same 2D pixel:
- A $0.2\text{ m}$ toy car at $2\text{ m}$ depth.
- A $2.0\text{ m}$ real sedan at $20\text{ m}$ depth.
- A $20.0\text{ m}$ billboard at $200\text{ m}$ depth.

### The "Tabletop" Trick: Inverse Perspective Mapping (IPM)
How do we undo division by $Z_c$ without a LiDAR sensor?
We make one temporary assumption: **The road ahead is a flat planar tabletop ($Z_{\text{ground}} = 0$)**.

If every pixel belongs to the ground plane, the mapping between the 2D image $(u, v)$ and the 2D ground coordinates $(X_w, Y_w)$ becomes a **bijective (1-to-1) projective transformation called a Homography ($H$)**:

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim H \begin{bmatrix} X_w \\ Y_w \\ 1 \end{bmatrix} \iff \begin{bmatrix} X_w \\ Y_w \\ 1 \end{bmatrix} \sim H^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

### ❓ Socratic Challenge: What happens to a 3D box truck under IPM?
What happens if an object violates the flat-ground assumption (e.g., a 3-meter tall delivery truck)?
- The tires touch the road at $d = 15\text{ m}$ (projected accurately).
- But the roof of the truck is 3 meters in the air!
- The IPM ray passes through the roof and keeps traveling until it hits the imaginary flat ground **75 meters behind the truck**!
- Result: IPM stretches the 3D truck into an enormous 60-meter smear across all lanes. This is why planar IPM works for painted lane lines, but fails for 3D obstacles!

---

## 3. 🧪 Lab Mission: Hands-On Simulator Experiments

Scroll to the **Interactive Camera Geometry & IPM Studio** at the top of this chapter:

1. **Experiment 1 (The Pitch Flare Disaster)**:
   - Observe the top-down BEV reconstruction of the two parallel highway lane lines.
   - Adjust the **Camera Pitch Angle** slider from $0.0^\circ$ to $+2.5^\circ$.
   - *Observation*: Notice how the parallel lines in the top-down view violently flare outward into a hyperbolic curve!
   - Now adjust the pitch slider to $-2.0^\circ$. The parallel lines cross each other into a sharp triangle.
2. **Experiment 2 (Dynamic Pitch Compensation)**:
   - Toggle **Dynamic IMU Suspension Compensation** to `ON`.
   - Now move the vehicle speed and braking slider to induce pitch.
   - *Observation*: The real-time transformation matrix dynamically cancels the chassis pitch angle $\Delta \theta(t)$, keeping the lane lines perfectly parallel in BEV space!

---

## 4. 🛠️ The Karpathy Build: Camera Rig & Homography from Scratch

Here is the linear algebra implemented in pure Python and NumPy with zero black box OpenCV functions:

```python
import numpy as np

def make_intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Builds 3x3 camera intrinsic matrix K."""
    return np.array([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Computes 3x3 orthogonal rotation matrix R = Rz(yaw) * Ry(pitch) * Rx(roll).
    Angles in radians.
    """
    # Roll (rotation around X axis)
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(roll), -np.sin(roll)],
        [0.0, np.sin(roll),  np.cos(roll)]
    ])
    # Pitch (rotation around Y axis)
    Ry = np.array([
        [ np.cos(pitch), 0.0, np.sin(pitch)],
        [ 0.0,           1.0, 0.0],
        [-np.sin(pitch), 0.0, np.cos(pitch)]
    ])
    # Yaw (rotation around Z axis)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw),  np.cos(yaw), 0.0],
        [0.0,          0.0,         1.0]
    ])
    return Rz @ Ry @ Rx

def compute_ground_homography(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Computes 3x3 Planar Homography H mapping ground (X, Y, 1) -> image (u, v, 1).
    Assumes ground plane is Z_world = 0.
    """
    # Columns of rotation matrix
    r1 = R[:, 0:1]  # X axis
    r2 = R[:, 1:2]  # Y axis
    # H = K * [r1, r2, t]
    Rt_ground = np.hstack([r1, r2, t.reshape(3, 1)])
    H = K @ Rt_ground
    return H / H[2, 2]  # Normalize scale

def unproject_pixel_to_ground(H_inv: np.ndarray, u: float, v: float) -> tuple[float, float]:
    """Unprojects a 2D image pixel onto the metric 3D ground plane (X, Y in meters)."""
    p_img = np.array([u, v, 1.0], dtype=np.float64)
    p_ground = H_inv @ p_img
    x_metric = p_ground[0] / p_ground[2]
    y_metric = p_ground[1] / p_ground[2]
    return float(x_metric), float(y_metric)
```

---

## 5. 📐 Mathematical Rigor: The Transformation Chain

A metric point $\mathbf{P}_w = [X_w, Y_w, Z_w]^T$ in the global ground frame undergoes four successive transformations to become pixel $(u, v)$:

```
World Coordinates [Xw, Yw, Zw]
       │
       ▼  [Extrinsic Transform: R, t]
Camera 3D Coordinates [Xc, Yc, Zc] = R * Pw + t
       │
       ▼  [Perspective Division: 1 / Zc]
Normalized Ray Coordinates [xc, yc, 1] = [Xc/Zc, Yc/Zc, 1]
       │
       ▼  [Intrinsic Matrix: K]
Image Pixel Coordinates [u, v, 1]^T = K * [xc, yc, 1]^T
```

### The Planar Homography Simplification
When $Z_w = 0$ (road surface):

$$\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = \mathbf{r}_1 X_w + \mathbf{r}_2 Y_w + \mathbf{r}_3 (0) + \mathbf{t} = \begin{bmatrix} \mathbf{r}_1 & \mathbf{r}_2 & \mathbf{t} \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ 1 \end{bmatrix}$$

Multiplying by intrinsic matrix $K$:

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} \mathbf{r}_1 & \mathbf{r}_2 & \mathbf{t} \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ 1 \end{bmatrix} = H \begin{bmatrix} X_w \\ Y_w \\ 1 \end{bmatrix}$$

Because $H \in \mathbb{R}^{3 \times 3}$ is invertible, any point on the ground can be reconstructed via:

$$\mathbf{p}_{\text{ground}} = H^{-1} \mathbf{p}_{\text{image}}$$

---

## 6. 🩺 Andrew Ng's Diagnostic Field Guide

| Observed Symptom | Underlying Mathematical Mechanism | Verification Test | Production Fix |
| :--- | :--- | :--- | :--- |
| **Lane lines flare outwards at distance** | **Camera Pitch Over-Estimation**: The calibrated pitch angle is higher than physical reality, causing rays to intersect the ground too early. | Check if distant lane width measures $> 3.7\text{ m}$ (standard highway width). | Recalibrate extrinsic pitch $\theta_{\text{pitch}}$ downward in increments of $0.1^\circ$. |
| **Lane lines converge into a triangle ahead** | **Camera Pitch Under-Estimation**: The calibrated pitch angle is lower than reality, projecting ground points too far away. | Check if distant lane width measures $< 3.7\text{ m}$. | Recalibrate extrinsic pitch $\theta_{\text{pitch}}$ upward. |
| **Distance errors oscillate during highway driving** | **Chassis Dynamic Pitch / Brake Squat**: Acceleration lifts the front end; braking compresses the front suspension. | Cross-correlate distance estimation error with longitudinal accelerometer $a_x(t)$. | Ingest chassis IMU pitch rate $\dot{\theta}$ and wheel height sensors into dynamic homography matrix $H(t)$. |

---

## 7. 🎯 Self-Check: Test Your Mental Model

<details>
<summary><b>Q1: If a camera has focal length $f_x = 1000\text{ px}$ and optical center $c_x = 960\text{ px}$, what is the horizontal angle of an object detected at pixel $u = 1460$?</b></summary>

<br>

**Answer**: 
From the pinhole projection formula:
$$u - c_x = f_x \tan(\theta) \implies 1460 - 960 = 500 = 1000 \cdot \tan(\theta)$$
$$\tan(\theta) = \frac{500}{1000} = 0.5 \implies \theta = \arctan(0.5) \approx 26.57^\circ$$
The object is located $26.57^\circ$ to the right of the camera's optical centerline.
</details>

<details>
<summary><b>Q2: Why can't planar IPM be used to calculate the 3D bounding box dimensions of an oncoming semi-truck?</b></summary>

<br>

**Answer**: Because planar IPM is strictly a 2D-to-2D projection conditioned on the flat ground assumption ($Z_{\text{road}} = 0$). An oncoming semi-truck has height ($Z_{\text{height}} \approx 3.5\text{ m}$). Rays that hit the top of the truck do not touch the road; unprojecting them with $H^{-1}$ projects them far into the distance behind the truck, catastrophically warping its geometry. 3D bounding boxes require dense depth estimation, Bird's-Eye View (BEV) frustum transforms, or 3D Occupancy Networks.
</details>
