# Module 01: Vision Foundations & Multi-Camera Geometry

> "LiDAR is a crutch. The biological brain perceives 3D velocity and depth from photons falling on 2D retinas. We must build that exact geometric engine." — Elon Musk

---

## 🎯 Purpose & Learning Goals
Tesla FSD relies purely on vision. But neural networks cannot reason about 3D space unless the sensor geometry is mathematically rigorous. In this module, you will:
1. Master the **Pinhole Camera Model** and camera intrinsic matrix $K$.
2. Derive coordinate transformations between the **Ego Vehicle Frame** and **Camera Optical Frames** using extrinsic matrices $[R \mid T]$.
3. Implement **Inverse Perspective Mapping (IPM)** to project road surface pixels onto a flat metric ground plane ($Z = 0$).
4. Build a calibrated **3-Camera Rig** (Front, Left Forward, Right Forward) and stitch their overlapping fields of view into a seamless ground plane.
5. FastAI "Break It & Fix It": Fix road projection smearing caused by dynamic vehicle pitch/roll suspension oscillations.

---

## 📐 Mathematical Formulation

### 1. Pinhole Camera Projection
A 3D point in the vehicle ego frame $P_{\text{ego}} = [X_e, Y_e, Z_e, 1]^T$ transforms to the camera optical frame via extrinsics $[R \mid T] \in \mathbb{R}^{3 \times 4}$:
$$P_{\text{cam}} = \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = R \cdot P_{\text{ego}} + T$$

The point is projected onto the 2D image plane via the intrinsic matrix $K$:
$$\tilde{p} = \begin{bmatrix} s \cdot u \\ s \cdot v \\ s \end{bmatrix} = K \cdot P_{\text{cam}} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}$$
Where $s = Z_c$ is the optical depth, and the pixel coordinates are:
$$u = \frac{f_x X_c}{Z_c} + c_x, \quad v = \frac{f_y Y_c}{Z_c} + c_y$$

### 2. Inverse Perspective Mapping (IPM)
Under the flat ground assumption ($Z_{\text{ego}} = 0$), the 3D-to-2D projection reduces to an invertible $3 \times 3$ planar homography $H$:
$$p = H \cdot \begin{bmatrix} X_{\text{ego}} \\ Y_{\text{ego}} \\ 1 \end{bmatrix} \implies \begin{bmatrix} X_{\text{ego}} \\ Y_{\text{ego}} \\ 1 \end{bmatrix} = H^{-1} \cdot p$$
This allows us to warp perspective camera images directly into a metric top-down Bird's-Eye View (BEV).

---

## 🏎️ Why Tesla Does It This Way
- Tesla vehicles do not use expensive HD LiDAR or radar. Instead, 8 surround cameras with overlapping frustums cover $360^\circ$.
- Vehicle acceleration and braking cause **pitch and roll variations** (up to $\pm 3^\circ$). If uncompensated, a $2^\circ$ pitch error shifts perceived lane distance at 50 meters by over 15 meters!
- Tesla runs real-time online camera self-calibration networks to estimate camera extrinsics $[R \mid T]$ dynamically on every frame.

---

## 🧪 Quick Run
```bash
# 1. Run the multi-camera calibration and IPM stitcher
python modules/01_camera_geometry/calibrate_rig.py

# 2. Run the Break-It & Fix-It drill (pitch perturbation & dynamic compensation)
python modules/01_camera_geometry/break_it_fix_it.py

# 3. Verify unit tests
pytest modules/01_camera_geometry/tests/test_geometry.py
```

🎨 **Interactive Visual Explainer**: Open [`visual_explainers/01_camera_geometry_ipm.html`](../../visual_explainers/01_camera_geometry_ipm.html) in your browser to experiment with 3D camera rays and pitch/roll distortion sliders in real time!
