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

## 📐 How the Formulas Are Obtained

### 1. Pinhole Projection from Similar Triangles
Consider a 3D point $P_c = (X_c, Y_c, Z_c)$ in front of a camera pinhole located at the origin $(0, 0, 0)$. The image sensor is placed at focal distance $f$ along the optical axis $Z_c$.

By drawing a 2D cross-section in the $X_c-Z_c$ plane, a ray connects $(X_c, Y_c, Z_c)$ through the origin to the sensor plane at $(x_{\text{sensor}}, y_{\text{sensor}}, f)$:
By **similar right triangles**:
$$\frac{x_{\text{sensor}}}{f} = \frac{X_c}{Z_c} \implies x_{\text{sensor}} = f \frac{X_c}{Z_c}$$
$$\frac{y_{\text{sensor}}}{f} = \frac{Y_c}{Z_c} \implies y_{\text{sensor}} = f \frac{Y_c}{Z_c}$$

To convert continuous physical metric coordinates on the silicon chip ($x_{\text{sensor}}$ in meters) into discrete pixel grid coordinates $(u, v)$ on a digital image:
$$u = s_x x_{\text{sensor}} + c_x = (s_x f) \frac{X_c}{Z_c} + c_x = f_x \frac{X_c}{Z_c} + c_x$$
$$v = s_y y_{\text{sensor}} + c_y = (s_y f) \frac{Y_c}{Z_c} + c_y = f_y \frac{Y_c}{Z_c} + c_y$$
Where $s_x, s_y$ are pixels per meter, $f_x, f_y$ are focal lengths in pixels, and $(c_x, c_y)$ is the principal point (image center).

Writing in projective homogeneous matrix coordinates:
$$\begin{bmatrix} s \cdot u \\ s \cdot v \\ s \end{bmatrix} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = K \cdot P_c$$
Where $s = Z_c$ is optical depth.

### 2. Derivation of the IPM Planar Homography ($H$)
A point in the vehicle ego frame $P_{\text{ego}} = [X_e, Y_e, Z_e, 1]^T$ projects to the image plane via extrinsics $[R \mid T]$ and intrinsics $K$:
$$\tilde{p} = K \left( R \begin{bmatrix} X_e \\ Y_e \\ Z_e \end{bmatrix} + T \right) = K \begin{bmatrix} r_1 & r_2 & r_3 & T \end{bmatrix} \begin{bmatrix} X_e \\ Y_e \\ Z_e \\ 1 \end{bmatrix}$$
Where $r_1, r_2, r_3$ are the three column vectors of the $3 \times 3$ rotation matrix $R$.

Under the **Flat Road Surface Assumption**, all drivable road points satisfy $Z_e = 0$:
$$\tilde{p} = K \begin{bmatrix} r_1 & r_2 & r_3 & T \end{bmatrix} \begin{bmatrix} X_e \\ Y_e \\ 0 \\ 1 \end{bmatrix} = K \begin{bmatrix} r_1 & r_2 & T \end{bmatrix} \begin{bmatrix} X_e \\ Y_e \\ 1 \end{bmatrix}$$
The third column $r_3$ is multiplied by 0 and drops out!
We define the **$3 \times 3$ Planar Homography Matrix**:
$$H = K \begin{bmatrix} r_1 & r_2 & T \end{bmatrix} \in \mathbb{R}^{3 \times 3}$$
Because $H$ is a non-singular $3 \times 3$ matrix, it can be inverted directly:
$$\begin{bmatrix} X_e \\ Y_e \\ 1 \end{bmatrix} = H^{-1} \begin{bmatrix} s \cdot u \\ s \cdot v \\ s \end{bmatrix}$$
This allows us to warp every pixel in the perspective camera view to an exact metric $(X, Y)$ coordinate in top-down space!

---

## 📺 Recommended Free Video Tutorials to Learn the Math
- **Camera Intrinsics & Extrinsics**: [Prof. Shree Nayar: Pinhole Cameras & Geometry](https://www.youtube.com/watch?v=qByYk6JggQU)
- **Matrix Transformations & Inverses**: [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- **Projections & Homographies**: [Prof. Gilbert Strang: MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)

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
