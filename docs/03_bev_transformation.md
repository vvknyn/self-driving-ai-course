# Deep Dive: Bird's-Eye View (BEV) Transformation (Lift-Splat-Shoot)

> "In 2D image space, objects shrink with distance and occlusions create disjoint boxes. In 3D Bird's-Eye View space, physics is Euclidean, metrics are meters, and trajectories can be planned directly."

---

## 1. Monocular Depth Ambiguity
Under standard pinhole projection:
$$u = f_x \frac{X}{Z} + c_x, \quad v = f_y \frac{Y}{Z} + c_y$$
A single pixel $(u, v)$ corresponds to an **infinite 1D ray** in 3D space:
$$P(d) = d \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}, \quad d \in (0, \infty)$$
Without active sensors (like LiDAR), the exact scalar depth $d$ cannot be uniquely determined from a single static pixel.

---

## 2. The Lift-Splat-Shoot (LSS) Formulation (Philion & Fidler ECCV 2020)

### Step 1: Lift (Categorical Depth Distribution)
Instead of forcing the network to predict a single deterministic depth $\hat{d}$, we frame depth estimation as a **Categorical Random Variable** $D \in \{d_1, d_2, \dots, d_K\}$ over $K$ discrete depth intervals:
$$d_k = D_{\text{min}} + k \cdot \Delta d$$

For each feature vector $c(u, v) \in \mathbb{R}^C$ at pixel $(u, v)$, the network predicts depth logits $\alpha \in \mathbb{R}^K$.
Applying Softmax yields probabilities:
$$P(D = d_k \mid u, v) = \frac{\exp(\alpha_k)}{\sum_{j=1}^K \exp(\alpha_j)}$$

The 2D feature is **Lifted** into a 3D ray of point features via an outer product:
$$F(d_k, u, v) = P(D = d_k \mid u, v) \otimes c(u, v) \in \mathbb{R}^{K \times C}$$
The lifted tensor for a camera frame of shape $(C_{\text{in}}, H, W)$ has shape:
$$(D, H, W, C_{\text{out}})$$

### Step 2: Splat (Geometric Frustum Unprojection)
Using the camera intrinsic matrix $K$ and extrinsic matrix $[R \mid T]$, each coordinate $(u, v, d_k)$ maps to an exact 3D metric coordinate in the vehicle ego frame:
$$P_{\text{ego}}(d_k, u, v) = R^T \left( d_k \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} - T \right) = \begin{bmatrix} X_{\text{ego}} \\ Y_{\text{ego}} \\ Z_{\text{ego}} \end{bmatrix}$$

We define a 3D voxel grid over the vehicle space bounded by:
$$X \in [X_{\text{min}}, X_{\text{max}}], \quad Y \in [Y_{\text{min}}, Y_{\text{max}}], \quad Z \in [Z_{\text{min}}, Z_{\text{max}}]$$
Voxel indices are assigned via integer division:
$$i_x = \left\lfloor \frac{X_{\text{ego}} - X_{\text{min}}}{\Delta X} \right\rfloor, \quad i_y = \left\lfloor \frac{Y_{\text{ego}} - Y_{\text{min}}}{\Delta Y} \right\rfloor, \quad i_z = \left\lfloor \frac{Z_{\text{ego}} - Z_{\text{min}}}{\Delta Z} \right\rfloor$$

### Step 3: Shoot (Pillar Pooling)
Features mapped to the same horizontal $(i_x, i_y)$ cell are pooled across vertical $Z$:
$$\text{BEV}(i_x, i_y) = \sum_{i_z} \sum_{p \in \text{Voxel}(i_x, i_y, i_z)} F(p)$$

The resulting 2D BEV feature map has shape:
$$(C_{\text{out}}, N_y, N_x)$$
This representation is now completely camera-agnostic: multiple surround cameras (Front, Left, Right, Rear) splat into the **same shared BEV coordinate grid** seamlessly!
