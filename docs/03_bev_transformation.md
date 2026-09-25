# Module 03: Bird's-Eye View (BEV) Transformation (Lift-Splat-Shoot)

> "In 2D image space, objects shrink with distance and perspective creates non-linear distortions. In 3D Bird's-Eye View space, physics is Euclidean, metrics are meters, and trajectories can be planned directly."

---

## 🟢 Tier 1: Intuition & Diagnostics (Andrew Ng Style)

### Why Can't We Plan in 2D Camera Space?
In 2D camera pixel coordinates:
1. **Objects shrink with distance**: A sedan 10 meters away occupies 50,000 pixels. The exact same sedan 60 meters away occupies only 500 pixels.
2. **Occlusions are non-linear**: A truck in front of a car partially blocks pixels in an unstructured way.
3. **Control happens in metric space**: You cannot tell a vehicle's steering rack to "steer 14 pixels to the left". You must command "swerve $1.2$ meters laterally over $15$ meters longitudinally".

We need a unified top-down coordinate system where **1 unit equals 1 physical meter**, regardless of which camera saw the object. This is **Bird's-Eye View (BEV)**.

### The Mental Model of Lift-Splat-Shoot (LSS)
Philion & Fidler (ECCV 2020) introduced the seminal 3-step paradigm:
1. **LIFT**: Take each 2D camera pixel. Since we don't know the exact distance, create a line of discrete points extending outwards along the camera's line of sight (like beads on a string). Assign each bead a probability score $P(D = d_k)$.
2. **SPLAT**: Use calibrated camera geometry matrices ($K, R, T$) to map every 3D bead into a shared metric voxel grid surrounding the vehicle.
3. **SHOOT**: Vertically sum all points falling into each ground column (pillar) to compress the 3D volume into a clean, 2D top-down BEV feature map.

```
          2D CAMERA FEATURE                    3D FRUSTUM RAY (LIFT)               TOP-DOWN BEV (SPLAT & SHOOT)
       ┌─────────────────────┐                   d=30m  ● P=0.05                       ┌─────────────────────────┐
       │      [Vehicle]      │                          │                              │                         │
       │      Feature c      │ ─────────────>    d=20m  ● P=0.85 (Peak!) ──────────>   │         [Car]           │
       │      at (u, v)      │                          │                              │      at (X=20m, Y=0)    │
       └─────────────────────┘                   d=10m  ● P=0.10                       │                         │
                                                        │                              │         [Ego]           │
                                                     (Camera)                          └─────────────────────────┘
```

### Andrew Ng Diagnostic Table: BEV Transform Failure Modes

| Symptom | Root Cause | Diagnostic Test | Solution |
|---|---|---|---|
| Obstacles appear as long radial smears pointing back to the camera | Depth distribution collapsed to uniform (high entropy) | Measure depth softmax entropy $\mathcal{H} = -\sum p \log p$ | Decrease softmax temperature $T$ or add depth supervision |
| Objects jump violently between adjacent BEV cells from frame to frame | Softmax temperature too low ($T < 0.1$), causing argmax snap | Check variance of predicted depth peak | Increase temperature $T$ to soften distribution |
| Voxel pooling kernel causes GPU Out-of-Memory (OOM) | Dense 3D frustum tensor exceeds VRAM | Print `tensor.element_size() * tensor.nelement()` | Use fast cumulative sum pooling or downsample image resolution |
| Vehicles in overlapping camera regions are duplicated twice | Extrinsic calibration mismatch between camera rigs | Check BEV overlap region for double-image ghosting | Refine extrinsic rotation matrices $[R \mid T]$ |

---

## 🟡 Tier 2: Code From Scratch (Andrej Karpathy Style)

Let's inspect the exact PyTorch operations that power Lift-Splat-Shoot. No libraries, just pure tensor manipulation.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LiftSplatShoot(nn.Module):
    def __init__(self, d_min=4.0, d_max=44.0, num_bins=41, c_feat=64):
        super().__init__()
        self.d_min = d_min
        self.d_max = d_max
        self.num_bins = num_bins
        # Discrete depth bin centers
        self.depth_bins = nn.Parameter(
            torch.linspace(d_min, d_max, num_bins), requires_grad=False
        )
        self.c_feat = c_feat

    def lift(self, features: torch.Tensor, depth_logits: torch.Tensor) -> torch.Tensor:
        """
        Features: (B, N, C, H, W)
        Depth Logits: (B, N, D, H, W)
        Returns Frustum Tensor: (B, N, D, H, W, C)
        """
        B, N, C, H, W = features.shape
        D = self.num_bins
        
        # Softmax along depth dimension to produce categorical probabilities
        prob_depth = F.softmax(depth_logits, dim=2) # (B, N, D, H, W)
        
        # Outer product: multiply depth probability by visual feature
        # Karpathy note: unsqueeze(5) onto prob, unsqueeze(2) onto features
        prob_expanded = prob_depth.unsqueeze(5)        # (B, N, D, H, W, 1)
        feat_permuted = features.permute(0, 1, 3, 4, 2).unsqueeze(2) # (B, N, 1, H, W, C)
        
        # Broadcasting produces the complete 3D lifted frustum!
        frustum = prob_expanded * feat_permuted        # (B, N, D, H, W, C)
        return frustum

    def splat_and_shoot_fast(self, frustum: torch.Tensor, x_coords: torch.Tensor, 
                             y_coords: torch.Tensor, nx: int = 100, ny: int = 100) -> torch.Tensor:
        """
        Vectorized BEV Pillar Pooling using 1D index_add_ (zero Python loops).
        Frustum: (N_points, C)
        x_coords, y_coords: (N_points,) integer grid indices
        """
        # Linearize 2D BEV grid indices: flat_idx = y * nx + x
        valid_mask = (x_coords >= 0) & (x_coords < nx) & (y_coords >= 0) & (y_coords < ny)
        valid_flat_idx = y_coords[valid_mask] * nx + x_coords[valid_mask]
        valid_features = frustum[valid_mask] # (M, C)
        
        # Allocate flat BEV accumulator
        flat_bev = torch.zeros((nx * ny, self.c_feat), dtype=frustum.dtype, device=frustum.device)
        
        # Accumulate all features falling into the same pillar
        flat_bev.index_add_(0, valid_flat_idx, valid_features)
        
        # Reshape into 2D BEV map: (C, ny, nx)
        bev_map = flat_bev.view(ny, nx, self.c_feat).permute(2, 0, 1)
        return bev_map
```

---

## 🔴 Tier 3: Mathematical Derivations & Proofs (MITx Probability Connection)

### 1. Depth Discretization as a Categorical Random Variable
- **Starting Point**: Let the true physical depth $D$ along optical ray $(u, v)$ be a continuous random variable with conditional density $f_{D \mid U, V}(d \mid u, v)$.
- **Discretization Proof**:
  We partition the continuous interval $[D_{\text{min}}, D_{\text{max}}]$ into $K$ disjoint bins:
  $$I_k = \left[ d_k - \frac{\Delta d}{2}, d_k + \frac{\Delta d}{2} \right), \quad k \in \{1, \dots, K\}$$
  By the axioms of probability:
  $$P(D \in I_k \mid u, v) = \int_{I_k} f_{D \mid U, V}(t \mid u, v) \, dt$$
  The neural network predicts unnormalized logits $\alpha(u, v) \in \mathbb{R}^K$. Under the **Maximum Entropy Principle** subject to moment constraints, the unique distribution matching logits without inductive bias is the **Categorical Softmax**:
  $$p_k = P(D = d_k \mid u, v) = \frac{\exp(\alpha_k / T)}{\sum_{j=1}^K \exp(\alpha_j / T)}$$

### 2. Expected 3D Feature Representation
By the **Law of Total Probability**, the expected feature vector $F(x, y, z)$ at physical location $(x, y, z)$ in the camera ray's path is:
$$\mathbb{E}[F(x, y, z)] = \sum_{k=1}^K P(D = d_k \mid u, v) \cdot c(u, v)$$
Where $c(u, v) \in \mathbb{R}^C$ is the deterministic 2D visual context extracted by the HydraNet backbone.

### 3. Voxel Pillar Pooling as an Orthogonal Projection
Let $V(i_x, i_y)$ denote the vertical pillar defined by horizontal grid indices $i_x, i_y$ for all vertical elevations $z \in [Z_{\text{min}}, Z_{\text{max}}]$.
The final BEV map value is the linear projection operator:
$$\text{BEV}(i_x, i_y) = \int_{Z_{\text{min}}}^{Z_{\text{max}}} F(x(i_x), y(i_y), z) \, dz \approx \sum_{z_j} F(i_x, i_y, z_j)$$
Because this operation is linear, backpropagation gradients $\frac{\partial \mathcal{L}}{\partial \text{BEV}}$ flow directly and smoothly back to both the depth logits $\alpha_k$ and the 2D feature backbone $c(u, v)$!

---

## 🎓 Tier 4: Cutting-Edge Research & PhD Track

### The Great Architectural Debate: LSS vs BEVFormer vs VAD

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                             BEV PARADIGM COMPARISON TABLE                                   │
├───────────────────┬──────────────────────┬──────────────────────────┬───────────────────────┤
│ Metric            │ LSS (ECCV 2020)      │ BEVFormer (ECCV 2022)    │ VAD (ICCV 2023)       │
├───────────────────┼──────────────────────┼──────────────────────────┼───────────────────────┤
│ Method            │ Categorical Splatting│ Deformable Cross-Attn    │ Sparse Vector Queries │
│ Memory Bandwidth  │ High (Frustum tensor)│ Medium (Query based)     │ Ultra-Low (Vectors)   │
│ Latency (FPS)     │ 25 FPS               │ 15 FPS                   │ 45 FPS                │
│ Small Object mAP  │ Strong               │ Very Strong              │ Balanced              │
│ NuScenes NDS Rank │ Classic Baseline     │ State-of-the-Art         │ State-of-the-Art      │
└───────────────────┴──────────────────────┴──────────────────────────┴───────────────────────┘
```

- **BEVDepth (AAAI 2023)** proved that supervising the depth distribution with projected LiDAR ground truth during training boosts 3D detection mAP by **+18%**, resolving the depth ambiguity of pure vision.
- **MatrixVT (ICCV 2023)** demonstrated that prime factorizing the voxel pooling step via matrix multiplication reduces LSS memory footprint by **$85\%$** without loss of accuracy.

### Open PhD Research Questions
- *Can we formulate a continuous neural field representation (e.g. 3D Gaussian Splatting) for BEV feature maps that eliminates fixed discrete grid resolutions entirely?*
- *How can BEV representations handle dynamic rolling-shutter camera distortions when the ego-vehicle takes a sharp turn at 80 km/h?*

---

## 🟣 Tier 5: Real-World Hardware & Practical Robotics

### GPU VRAM Profiling & Optimization on Edge Devices
If you deploy LSS on an embedded robotic platform (Jetson Orin Nano, Raspberry Pi 5 with AI Hat, or Apple Silicon MPS):
- A naive frustum tensor with $N=6$ cameras, $D=64$ depth bins, $C=64$ channels, and $H=128, W=352$ consumes:
  $$\text{Memory} = 6 \times 64 \times 64 \times 128 \times 352 \times 4 \text{ bytes} \approx \mathbf{4.42 \text{ Gigabytes!}}$$
- On an 8 GB Jetson Orin Nano, this will immediately cause `CUDA Out of Memory` kernel panics!

**The 3 Engineering Fixes**:
1. **Logarithmic Depth Binning**: Instead of linear spacing ($0.5\text{m}$ everywhere), allocate dense bins close to the vehicle ($4\text{m}\text{--}20\text{m}$) and sparse bins far away ($20\text{m}\text{--}60\text{m}$). This cuts $D$ from 64 to 28 bins!
2. **Channel Projection**: Reduce trunk channels from $C=64$ to $C=32$ prior to lifting.
3. **In-place Index Add**: Never materialize the full 6D frustum tensor; use PyTorch `torch.index_add_` directly from flattened camera indices.
