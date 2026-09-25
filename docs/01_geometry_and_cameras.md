# Deep Dive: Multi-Camera Geometry & Inverse Perspective Mapping (IPM)

> "To understand 3D from 2D, you must master the coordinate transforms between photons, sensors, and the metric physical ground."

---

## 1. The Pinhole Camera Model
Light rays from 3D space pass through a pinhole aperture and project onto the 2D camera sensor plane.

Let a 3D point in the Camera Optical Coordinate Frame be:
$$P_c = [X_c, Y_c, Z_c]^T$$

Under perspective projection, its position on the image plane $(u, v)$ in pixels is given by:
$$u = f_x \frac{X_c}{Z_c} + c_x, \quad v = f_y \frac{Y_c}{Z_c} + c_y$$

In matrix form using homogeneous coordinates:
$$\begin{bmatrix} s \cdot u \\ s \cdot v \\ s \end{bmatrix} = K \cdot P_c = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}$$
Where:
- $K \in \mathbb{R}^{3 \times 3}$ is the **Intrinsic Matrix**.
- $f_x, f_y$ are focal lengths in pixel units: $f_x = \frac{W / 2}{\tan(\text{HFOV} / 2)}$.
- $c_x, c_y$ is the principal point (optical center, typically image center).
- $s = Z_c$ is the optical depth along the camera's Z-axis.

---

## 2. Vehicle Ego Frame vs. Camera Optical Frame
Standard vehicle coordinate frame (ISO 8855 / SAE):
- **$+X$**: Forward (pointing out the front bumper)
- **$+Y$**: Left (driver's left side)
- **$+Z$**: Up (pointing toward the sky)

Standard camera optical frame:
- **$+X_c$**: Right (along camera sensor horizontal)
- **$+Y_c$**: Down (along camera sensor vertical)
- **$+Z_c$**: Forward (optical axis into the scene)

The extrinsic transformation transforms a point from vehicle ego space $P_{\text{ego}}$ to camera space $P_c$:
$$P_c = R \cdot P_{\text{ego}} + T$$
Where $R \in \mathbb{R}^{3 \times 3}$ is an orthonormal rotation matrix ($R^T R = I, \det(R) = 1$) and $T \in \mathbb{R}^{3 \times 1}$ is the translation vector.

---

## 3. Inverse Perspective Mapping (IPM)
When assuming a flat road surface ($Z_{\text{ego}} = 0$):
$$\begin{bmatrix} s \cdot u \\ s \cdot v \\ s \end{bmatrix} = K \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_1 \\ r_{21} & r_{22} & r_{23} & t_2 \\ r_{31} & r_{32} & r_{33} & t_3 \end{bmatrix} \begin{bmatrix} X_e \\ Y_e \\ 0 \\ 1 \end{bmatrix} = K \begin{bmatrix} r_1 & r_2 & t \end{bmatrix} \begin{bmatrix} X_e \\ Y_e \\ 1 \end{bmatrix}$$

This simplifies the 3D-to-2D projection into a planar **Homography Matrix** $H \in \mathbb{R}^{3 \times 3}$:
$$H = K \begin{bmatrix} r_1 & r_2 & t \end{bmatrix}$$
Because $H$ is a square $3 \times 3$ matrix, it is invertible!
$$\begin{bmatrix} X_e \\ Y_e \\ 1 \end{bmatrix} = H^{-1} \begin{bmatrix} s \cdot u \\ s \cdot v \\ s \end{bmatrix}$$

---

## 4. Diagnostic Error Analysis: Suspension Pitch Error
If the vehicle pitches forward by angle $\Delta \theta$ during braking, the ground intersection angle $\alpha$ shifts:
$$\alpha_{\text{actual}} = \alpha_{\text{nominal}} + \Delta \theta$$
$$\text{Estimated Distance } X = \frac{h}{\tan(\alpha_{\text{nominal}})} \neq \frac{h}{\tan(\alpha_{\text{actual}})}$$
At a distance of 30 meters with camera height $h = 1.4\text{m}$, nominal $\alpha \approx 2.67^\circ$.
If braking causes a pitch tilt of $\Delta \theta = +2.0^\circ$, the angle becomes $4.67^\circ$:
$$X_{\text{estimated}} = \frac{1.4}{\tan(4.67^\circ)} = 17.14\text{ meters}$$
A massive **12.86 meter estimation error**! This proves why online self-calibration networks are essential for vision-only autonomy.
