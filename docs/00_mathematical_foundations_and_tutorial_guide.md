# Mathematical Foundations & Self-Study Tutorial Guide

> "Never treat a formula as magic. In this course, every equation is derived from first principles (geometry, probability, or kinematics), paired with physical intuition, and backed by curated free tutorials."

---

## 🧭 Curated Free Math Tutorials by Topic

Before or alongside each module, if you want a visual, intuitive refresher on the underlying math, use these world-class free resources:

| Math Domain | Recommended Free Tutorial | Why It Matters for Autonomous Driving |
|---|---|---|
| **Linear Transformations & Projections** | [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | Camera intrinsics $K$, extrinsics $[R \mid T]$, coordinate transformations, change of basis. |
| **Probability, Random Variables & Bayes' Rule** | [MIT 6.041x: Introduction to Probability (Prof. John Tsitsiklis)](https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/) | Depth categorical distributions, Bernoulli occupancy grids, and Kalman Gaussian conditioning. |
| **Maximum Likelihood Estimation (MLE)** | [StatQuest: Maximum Likelihood Clearly Explained (Josh Starmer)](https://www.youtube.com/watch?v=XepXtl9YKwc) | Homoscedastic uncertainty loss balancing in HydraNet (Module 02). |
| **Camera Geometry & Pinhole Models** | [First Principles of Computer Vision (Prof. Shree Nayar, Columbia)](https://www.youtube.com/playlist?list=PL2zRqk164rLXjiA14hTzbmh7e9uLzFfio) | Pinhole projection, vanishing points, lens distortion, and planar homographies. |
| **Kalman Filtering from First Principles** | [Greg Czerniak: Kalman Filters Explained Simply](https://greg.czerniak.info/guides/kalman1/) & [Brian Douglas: State-Space & Kalman](https://www.youtube.com/watch?v=mwn8xhgNpFY) | Multi-object tracking, Gaussian covariance updates, and Mahalanobis distance. |
| **Calculus, Jerk & Curvature** | [3Blue1Brown: Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) | Quintic spline boundary value problems, analytical road curvature $\kappa$, and jerk $\dddot{s}(t)$. |
| **Control Systems & Stability (Lyapunov)** | [Brian Douglas: Classical & Modern Control Systems](https://www.youtube.com/user/ControlLectures) | Stanley steering stability proof, Pure Pursuit lookahead, and PID speed regulation. |

---

## 📐 Master Step-by-Step Derivations Index

Below is the complete catalog of first-principles derivations included across the course modules:

### 1. Pinhole Projection from Similar Triangles (Module 01)
- **Starting Point**: A point in 3D camera coordinates $(X_c, Y_c, Z_c)$ emits a light ray through a pinhole at origin $(0, 0, 0)$ hitting the sensor plane at distance $f$ (focal length).
- **Geometric Derivation**:
  By similar triangles formed by the optical axis ($Z_c$) and the horizontal sensor axis ($X_c$):
  $$\frac{x_{\text{sensor}}}{f} = \frac{X_c}{Z_c} \implies x_{\text{sensor}} = f \frac{X_c}{Z_c}$$
  Converting metric sensor coordinates (meters) to discrete image pixels $(u, v)$ via pixel pitch $(s_x, s_y)$ pixels/meter and optical center $(c_x, c_y)$:
  $$u = s_x x_{\text{sensor}} + c_x = (s_x f) \frac{X_c}{Z_c} + c_x = f_x \frac{X_c}{Z_c} + c_x$$
  $$v = s_y y_{\text{sensor}} + c_y = (s_y f) \frac{Y_c}{Z_c} + c_y = f_y \frac{Y_c}{Z_c} + c_y$$
- **Tutorial**: [Columbia University: Pinhole Camera Models](https://www.youtube.com/watch?v=qByYk6JggQU)

---

### 2. Planar Homography for IPM by Ground Plane Constraint (Module 01)
- **Starting Point**: General 3D to 2D projection $p = K [R \mid T] P_{\text{ego}}$.
- **Algebraic Derivation**:
  In vehicle coordinates, the road is assumed flat: $Z_{\text{ego}} = 0$.
  $$\begin{bmatrix} s \cdot u \\ s \cdot v \\ s \end{bmatrix} = K \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_1 \\ r_{21} & r_{22} & r_{23} & t_2 \\ r_{31} & r_{32} & r_{33} & t_3 \end{bmatrix} \begin{bmatrix} X_e \\ Y_e \\ 0 \\ 1 \end{bmatrix}$$
  Notice the third column of the rotation matrix $[r_{13}, r_{23}, r_{33}]^T$ is multiplied by $0$ and drops out!
  $$\begin{bmatrix} s \cdot u \\ s \cdot v \\ s \end{bmatrix} = K \begin{bmatrix} r_1 & r_2 & t \end{bmatrix} \begin{bmatrix} X_e \\ Y_e \\ 1 \end{bmatrix} = H \begin{bmatrix} X_e \\ Y_e \\ 1 \end{bmatrix}$$
  Because $H$ is a non-singular $3 \times 3$ matrix, it is directly invertible:
  $$\begin{bmatrix} X_e \\ Y_e \\ 1 \end{bmatrix} = H^{-1} \begin{bmatrix} s \cdot u \\ s \cdot v \\ s \end{bmatrix}$$
- **Tutorial**: [Multiple View Geometry in Computer Vision (Hartley & Zisserman)](https://www.robots.ox.ac.uk/~vgg/hzbook/)

---

### 3. Multi-Task Uncertainty Loss via Maximum Likelihood (Module 02)
- **Starting Point**: Multi-task model with shared weights $W$ and observation noise $\sigma_i^2$ for task $i$.
- **Probabilistic Derivation (MITx Probability Connection)**:
  Assume observation likelihood is Gaussian:
  $$p(y_i \mid f_i(x; W), \sigma_i) = \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left( -\frac{\|y_i - f_i(x; W)\|^2}{2\sigma_i^2} \right)$$
  Taking the negative natural logarithm ($-\ln$):
  $$-\ln p(y_i \mid \dots) = \frac{1}{2\sigma_i^2} \|y_i - f_i(x; W)\|^2 + \ln(\sqrt{2\pi\sigma_i^2})$$
  $$= \frac{1}{2\sigma_i^2} \mathcal{L}_i(W) + \ln \sigma_i + \frac{1}{2}\ln(2\pi)$$
  Dropping the constant $\frac{1}{2}\ln(2\pi)$ and summing over all $M$ tasks:
  $$\mathcal{L}_{\text{total}} = \sum_{i=1}^M \left( \frac{1}{2\sigma_i^2} \mathcal{L}_i(W) + \ln \sigma_i \right)$$
  To prevent $\sigma_i \le 0$ during gradient descent, substitute $s_i = \ln(\sigma_i^2) \iff \sigma_i^2 = \exp(s_i)$ and $\ln \sigma_i = \frac{1}{2} s_i$:
  $$\mathcal{L}_{\text{total}}(W, s_1, \dots, s_M) = \sum_{i=1}^M \left( \frac{1}{2}\exp(-s_i) \mathcal{L}_i(W) + \frac{1}{2} s_i \right)$$
- **Tutorial**: [StatQuest: Maximum Likelihood Clearly Explained](https://www.youtube.com/watch?v=XepXtl9YKwc)

---

### 4. Lift-Splat-Shoot Expectation Weighting (Module 03)
- **Starting Point**: Optical depth along ray is uncertain; model predicts categorical probabilities $P(D = d_k \mid u, v)$.
- **Derivation**:
  By the Law of Total Probability, the expected 3D feature representation at ray depth $d$ is:
  $$\mathbb{E}[F(u, v, d)] = \sum_{k=1}^K P(D = d_k \mid u, v) \cdot c(u, v)$$
  This outer product forms the frustum tensor $F \in \mathbb{R}^{D \times H \times W \times C}$.
- **Tutorial**: [Jonah Philion & Sanja Fidler: Lift, Splat, Shoot ECCV 2020](https://arxiv.org/abs/2008.05711)

---

### 5. Kalman Filter State Update & Covariance Contraction (Module 05)
- **Starting Point**: Prior belief $x \sim \mathcal{N}(\mu^-, P^-)$ and linear measurement $z = Hx + v$ with $v \sim \mathcal{N}(0, R)$.
- **Bayesian Derivation**:
  The posterior distribution is the product of two Gaussian PDFs:
  $$p(x \mid z) \propto p(z \mid x) \cdot p(x) \propto \exp\left( -\frac{1}{2} \left[ (z - Hx)^T R^{-1} (z - Hx) + (x - \mu^-)^T (P^-)^{-1} (x - \mu^-) \right] \right)$$
  Expanding and completing the square for $x$:
  The quadratic coefficient gives the posterior precision matrix:
  $$P^{-1} = (P^-)^{-1} + H^T R^{-1} H$$
  Applying the **Woodbury Matrix Inversion Identity**:
  $$P = (I - K H) P^-, \quad \text{where } K = P^- H^T (H P^- H^T + R)^{-1}$$
  The linear term gives the posterior mean:
  $$\mu = \mu^- + K (z - H \mu^-)$$
  Because $K H P^-$ is positive semi-definite, $\text{Tr}(P) < \text{Tr}(P^-)$: uncertainty strictly decreases!
- **Tutorial**: [MIT OCW 6.041x: Conditioning Normal Random Variables](https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/)

---

### 6. Analytical Road Curvature from Cubic Splines (Module 05)
- **Starting Point**: Vector lane modeled as $y(x) = c_0 + c_1 x + c_2 x^2 + c_3 x^3$.
- **Calculus Derivation**:
  Curvature $\kappa$ is defined as the rate of change of heading angle $\psi$ with respect to arc length $s$:
  $$\kappa = \left| \frac{d\psi}{ds} \right|$$
  By the chain rule:
  $$\frac{d\psi}{ds} = \frac{d\psi/dx}{ds/dx}$$
  From differential geometry:
  $$ds = \sqrt{dx^2 + dy^2} = \sqrt{1 + (y'(x))^2} dx \implies \frac{ds}{dx} = \sqrt{1 + (y'(x))^2}$$
  Since $\psi(x) = \arctan(y'(x))$:
  $$\frac{d\psi}{dx} = \frac{d}{dx}[\arctan(y')] = \frac{y''(x)}{1 + (y'(x))^2}$$
  Dividing the two derivatives:
  $$\kappa(x) = \frac{\frac{y''(x)}{1 + (y'(x))^2}}{\sqrt{1 + (y'(x))^2}} = \frac{|y''(x)|}{(1 + (y'(x))^2)^{3/2}}$$
- **Tutorial**: [Khan Academy: Curvature Formula Derivation](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/curvature)

---

### 7. Quintic Boundary Value Problem (Module 06)
- **Starting Point**: 5th-order polynomial $s(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$.
- **Algebraic Derivation**:
  Initial conditions ($t=0$):
  $$s(0) = a_0 = s_0, \quad \dot{s}(0) = a_1 = v_0, \quad \ddot{s}(0) = 2 a_2 = a_0 \implies a_2 = \frac{a_0}{2}$$
  Terminal conditions ($t=T$):
  $$s(T) = a_0 + a_1 T + a_2 T^2 + a_3 T^3 + a_4 T^4 + a_5 T^5 = s_T$$
  $$\dot{s}(T) = a_1 + 2 a_2 T + 3 a_3 T^2 + 4 a_4 T^3 + 5 a_5 T^4 = v_T$$
  $$\ddot{s}(T) = 2 a_2 + 6 a_3 T + 12 a_4 T^2 + 20 a_5 T^3 = a_T$$
  Subtracting known initial terms to the right-hand side produces the linear system:
  $$\begin{bmatrix} T^3 & T^4 & T^5 \\ 3 T^2 & 4 T^3 & 5 T^4 \\ 6 T & 12 T^2 & 20 T^3 \end{bmatrix} \begin{bmatrix} a_3 \\ a_4 \\ a_5 \end{bmatrix} = \begin{bmatrix} s_T - s_0 - v_0 T - \frac{1}{2} a_0 T^2 \\ v_T - v_0 - a_0 T \\ a_T - a_0 \end{bmatrix}$$
  This $3 \times 3$ system has non-zero determinant $\det(A) = 2 T^9 \neq 0$ for all $T > 0$, guaranteeing a unique, non-oscillatory smooth solution!
- **Tutorial**: [Werling et al. ICRA 2010: Optimal Trajectory Generation in Frenet Frame](https://www.researchgate.net/publication/224155184_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame)

---

### 8. Stanley Controller Exponential Stability Proof (Module 07)
- **Starting Point**: Cross-track error dynamics $\dot{e}(t) = -v(t) \sin(\theta_e - \delta)$.
- **Control Theory Derivation (Sebastian Thrun / Hoffmann IEEE 2007)**:
  Substitute the Stanley steering law $\delta = \theta_e + \arctan\left(\frac{k \cdot e(t)}{v(t)}\right)$:
  $$\theta_e - \delta = -\arctan\left(\frac{k \cdot e(t)}{v(t)}\right)$$
  Using the trigonometric identity $\sin(\arctan(u)) = \frac{u}{\sqrt{1 + u^2}}$:
  $$\sin\left(-\arctan\left(\frac{ke}{v}\right)\right) = -\frac{\frac{ke}{v}}{\sqrt{1 + \left(\frac{ke}{v}\right)^2}} = -\frac{k \cdot e}{\sqrt{v^2 + k^2 e^2}}$$
  Thus, error rate of change is:
  $$\dot{e}(t) = -v \left( -\frac{k \cdot e}{\sqrt{v^2 + k^2 e^2}} \right) = -\frac{k \cdot v(t) \cdot e(t)}{\sqrt{v(t)^2 + k^2 e(t)^2}}$$
  Choose the positive-definite Lyapunov function $V(e) = \frac{1}{2} e^2$:
  $$\dot{V}(e) = e \cdot \dot{e} = -\frac{k \cdot v(t) \cdot e^2}{\sqrt{v(t)^2 + k^2 e^2}}$$
  For all $v > 0$ and $e \neq 0$:
  $$\dot{V}(e) < 0$$
  By Lyapunov's Direct Method, the cross-track error $e(t) \to 0$ exponentially as $t \to \infty$!
- **Tutorial**: [Brian Douglas: Lyapunov Stability Theorem](https://www.youtube.com/watch?v=1Fq-XG198u8)
