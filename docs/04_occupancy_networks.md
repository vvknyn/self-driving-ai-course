# Module 04: 3D Occupancy Networks & Spatiotemporal Dynamics

> "Bounding boxes fail on ontology cracks—objects that defy standard classification. 3D Occupancy treats the physical world as a continuous voxel field." — Ashok Elluswamy, Tesla AI Day 2022

---

## 🟢 Tier 1: Intuition & Diagnostics (Andrew Ng Style)

### The Breakdown of 3D Bounding Boxes
For two decades, autonomous vehicle perception revolved around **3D Bounding Boxes**. Engineers trained detectors to classify objects into a rigid list of semantic labels:
$$\mathcal{C} \in \{\text{Car}, \text{Truck}, \text{Pedestrian}, \text{Bicycle}, \text{Traffic Cone}\}$$

**The Fatal Failure Mode (Ontology Cracks)**:
What happens when your vehicle encounters an object that does not fit neatly into these neat little rectangular boxes?
- An overturned semi-trailer with a flatbed sticking out across two lanes.
- A flat mattress falling off a luggage rack at 100 km/h.
- An overhanging tree branch hanging down to windshield height ($Z=2.5\text{m}$).
- A pile of construction debris or traffic cones knocked onto their sides.

In every one of these cases, traditional bounding box detectors predict **zero bounding boxes**. To the planner, the road appears 100% empty!

### The 3D Occupancy Solution: The "Minecraft" World Model
Instead of asking *"What is that object?"*, 3D Occupancy asks two simpler, physics-based questions:
1. **Is there physical matter in this voxel volume?** ($P(\text{occupied}) \in [0, 1]$).
2. **If so, in what direction and speed is it moving?** ($\vec{v} = (v_x, v_y, v_z) \in \mathbb{R}^3$).

The car doesn't care whether the obstacle is a fallen refrigerator, a cardboard box, or an alien spacecraft—if a voxel volume is occupied, **do not drive into it!**

```
      BOUNDING BOX DETECTION (BRITTLE)            3D OCCUPANCY VOXEL FIELD (ROBUST)
      ┌─────────────────────────────┐             ┌─────────────────────────────┐
      │   ? No Box Predicted !      │             │   ■ ■ ■ ■ ■ ■ ■ (Occupied)  │
      │   (Unrecognized debris)     │             │   ■ ■ ■ ■ ■ ■ ■             │
      │                             │ ──────────> │   □ □ □ □ □ □ □ (Freespace) │
      │   Result: CRASH into        │             │   □ □ □ □ □ □ □             │
      │   unclassified mattress.    │             │   Planner stops automatically!│
      └─────────────────────────────┘             └─────────────────────────────┘
```

### Andrew Ng Diagnostic Table: Occupancy Network Failures

| Symptom | Root Cause | Diagnostic Test | Solution |
|---|---|---|---|
| Model predicts all voxels are empty (98% accuracy, 0% recall) | Extreme class imbalance (empty air dominates road) | Print mean occupancy probability over validation set | Supervise with Focal Loss ($\gamma=2.0$) or class-weighted Lovász loss |
| Occluded vehicle is forgotten the moment a truck passes in front | Lack of temporal state (single-frame amnesia) | Test detection continuity during 1-second visual blockage | Introduce Spatiotemporal ConvGRU with ego-motion compensation |
| Overhanging bridge triggers false emergency braking | 2D BEV projection collapsed vertical elevation $Z$ | Check vertical slice prediction at $Z > 2.5\text{m}$ | Preserve 3D voxel elevation rather than 2D pillar compression |
| Inference latency exceeds 80 ms per frame | Dense 3D convolutions scale cubically $\mathcal{O}(N_x N_y N_z)$ | Benchmark 3D Conv layer FLOPs with `torch.profiler` | Switch to Sparse 3D Convolutions or Tri-Perspective View (TPV) |

---

## 🟡 Tier 2: Code From Scratch (Andrej Karpathy Style)

Let's implement a clean, vectorized 3D Occupancy Spatiotemporal ConvGRU in raw PyTorch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatiotemporalConvGRU(nn.Module):
    """
    Spatiotemporal Recurrent Fusion Cell for 3D/2D Feature Volumes.
    Carries forward temporal belief state across occlusions.
    """
    def __init__(self, channels: int = 32):
        super().__init__()
        self.channels = channels
        # Gating convolutions for Reset (R) and Update (Z) gates
        self.conv_gates = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1)
        # Candidate state convolution
        self.conv_candidate = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor | None = None) -> torch.Tensor:
        """
        x_t: Current frame observation features (B, C, H, W)
        h_prev: Previous frame temporal memory state (B, C, H, W)
        """
        if h_prev is None:
            h_prev = torch.zeros_like(x_t)

        # Concatenate current observation with prior belief
        combined = torch.cat([x_t, h_prev], dim=1) # (B, 2C, H, W)
        gates = torch.sigmoid(self.conv_gates(combined))
        
        # Split into Reset gate R and Update gate Z
        r_gate, z_gate = torch.chunk(gates, 2, dim=1)
        
        # Candidate memory state
        combined_candidate = torch.cat([x_t, r_gate * h_prev], dim=1)
        h_tilde = torch.tanh(self.conv_candidate(combined_candidate))
        
        # Convex combination: state update
        h_t = (1.0 - z_gate) * h_prev + z_gate * h_tilde
        return h_t

class OccupancyHead(nn.Module):
    def __init__(self, in_channels: int = 32, num_z_layers: int = 8):
        super().__init__()
        # Predicts occupancy logits across discrete elevation layers Z
        self.occ_conv = nn.Conv2d(in_channels, num_z_layers, kernel_size=1)
        # Predicts 2D/3D velocity flow vectors (vx, vy)
        self.flow_conv = nn.Conv2d(in_channels, 2, kernel_size=1)

    def forward(self, bev_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        occ_logits = self.occ_conv(bev_features)   # (B, Z, H, W)
        flow_vectors = self.flow_conv(bev_features) # (B, 2, H, W)
        return occ_logits, flow_vectors
```

---

## 🔴 Tier 3: Mathematical Derivations & Proofs (MITx Probability)

### 1. 3D Occupancy as a Spatial Bernoulli Random Field
- **Starting Point**: Let 3D space be discretized into a lattice of $M = N_x \times N_y \times N_z$ voxel cells $V_i$.
- **Probabilistic Formulation**:
  Each voxel $V_i$ is an independent **Bernoulli Random Variable**:
  $$O_i \in \{0, 1\}, \quad P(O_i = 1) = p_i, \quad P(O_i = 0) = 1 - p_i$$
  The joint probability of an entire 3D scene observation $O = (O_1, \dots, O_M)$ is:
  $$P(O \mid X) = \prod_{i=1}^M p_i^{O_i} (1 - p_i)^{1 - O_i}$$
  Taking the negative log-likelihood gives standard Binary Cross-Entropy:
  $$\mathcal{L}_{\text{BCE}} = -\sum_{i=1}^M \left[ O_i \ln(p_i) + (1 - O_i) \ln(1 - p_i) \right]$$

### 2. Focal Loss Proof for Extreme Class Imbalance
In a driving voxel grid, over $98\%$ of voxels are empty air ($O_i = 0$). With standard BCE, easy negative voxels dominate gradient updates.
Lin et al. (ICCV 2017) introduced **Focal Loss**:
$$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \ln(p_t)$$
Where $p_t = p$ if $O_i=1$, else $p_t = 1-p$.
- **Gradient Derivation**:
  $$\frac{\partial \mathcal{L}_{\text{Focal}}}{\partial z} = \alpha_t (1 - p_t)^\gamma \left[ \gamma p_t \ln(p_t) + p_t - 1 \right]$$
  For an easy empty voxel ($p \approx 0.01 \implies p_t = 0.99$):
  $$(1 - p_t)^2 = (0.01)^2 = 0.0001$$
  The gradient is attenuated by a factor of **$10,000\times$**! This forces the optimizer to focus exclusively on rare, occupied obstacle voxels.

---

## 🎓 Tier 4: Cutting-Edge Research & PhD Track

### 1. Tesla AI Day 2022 Occupancy Architecture Breakdown
At Tesla AI Day 2022, Ashok Elluswamy presented the production Tesla Occupancy Network:
1. **Multi-Camera Inputs**: 8 surround camera streams ($1280 \times 960$ at 36 FPS).
2. **Backbone**: RegNet-style trunk with Feature Pyramid Networks.
3. **Cross-Attention Lifting**: Deformable queries lift 2D image features into a $128 \times 128 \times 16$ 3D voxel volume.
4. **Spatiotemporal ConvGRU**: Aligns historical voxel states using IMU ego-motion odometry (translating and rotating the prior grid $H_{t-1}$ to compensate for vehicle movement).
5. **NeRF Supervised Training**: The network was trained without human labels by using Neural Radiance Field (NeRF) reconstruction over hundreds of millions of customer driving clips!

### 2. Tri-Perspective View (TPVFormer) & Sparse Convolutions
Dense 3D convolution on a $200 \times 200 \times 16$ volume requires over $640,000$ voxels per frame.
- **TPVFormer (CVPR 2023)**: Factorizes the 3D volume into three orthogonal 2D planes: Top-Down (XY), Front (XZ), and Side (YZ). Complexity drops from $\mathcal{O}(N^3)$ to $\mathcal{O}(3 N^2)$.
- **Sparse Convolutions (SpConv)**: Only computes convolutions on non-empty voxels, cutting memory by $90\%$.

---

## 🟣 Tier 5: Real-World Hardware & Practical Robotics

### Edge Memory Scaling & Real-Time Constraints

| Representation | Grid Dimensions | Memory per Frame (FP32) | Latency (Jetson Orin) | Recommended Hardware |
|---|---|---|---|---|
| Dense 3D Voxel | $128 \times 128 \times 16 \times 32$ | 33.5 MB | 28 ms | RTX 4090 / Drive Thor |
| Sparse 3D (SpConv) | $128 \times 128 \times 16$ (5% fill) | 1.8 MB | 9.4 ms | Jetson Orin Nano 8GB |
| Tri-Perspective (TPV) | $3 \times (128 \times 128) \times 32$ | 6.2 MB | 6.8 ms | Apple M-Series / Jetson |

**Robotics Implementation Tip**: If implementing on a mobile robot without a discrete GPU, run the Occupancy Network at 10 Hz and use IMU odometry integration at 100 Hz to extrapolate the prior voxel grid between neural inference ticks.
