# Chapter 05: 3D Occupancy Networks & Temporal Memory

> **The Big Question**: *A pedestrian walks behind a thick concrete pillar at an intersection. In the current camera snapshot, they are completely invisible. A single-frame perception system concludes the crosswalk is empty and steps on the accelerator. How does an autonomous vehicle maintain "object permanence" like a human brain—and how can it detect weird obstacles like overturned furniture or construction rubble that don't fit into clean 3D bounding boxes?*

---

## 1. 🚨 The Real-World Dilemma: The Occlusion & "Unknown Object" Trap

Traditional autonomous perception relies on **3D Bounding Boxes**:
- You train a neural network to fit tight oriented cuboids $[x, y, z, w, l, h, \theta]$ around specific pre-defined classes: `Car`, `Truck`, `Pedestrian`, `Cyclist`.

### Flaw 1: The Long-Tail Rubble Disaster
What happens when a flatbed truck drops a mattress, a ladder, an overturned cement mixer, or tree branches across the freeway?
- The 3D detector searches for cars and pedestrians.
- The mattress has no wheels, no hood, and no legs.
- The detector outputs: **Confidence = 0.0**. The car plows into the ladder at 70 mph!

### Flaw 2: The Instant Amnesia Problem
When an object passes behind a traffic sign, tree, or parked bus:
- At time $t = 0.0\text{ s}$: Pedestrian detected with 95% confidence.
- At time $t = 0.1\text{ s}$: Pedestrian occluded by bus. Confidence drops to **0.0%**.
- The car has zero memory. It assumes the pedestrian vanished into thin air!

```
Frame t = 0 (Visible):             Frame t = 1 (Occluded by Bus):
┌──────────────────────────┐       ┌──────────────────────────┐
│  [Pedestrian]            │       │      ┌──────┐            │
│       🚶                 │       │      │ BUS  │  (🚶 hidden)│
│                          │       │      └──────┘            │
└──────────────────────────┘       └──────────────────────────┘
Detection: YES                     Single-frame AI: "Road is Clear!"
                                   ACCELERATING ──► CRASH HAZARD
```

---

## 2. 💡 The Mental Model: Voxel Fields & Recurrent Memory

### The Voxel Field: A 3D Minecraft World
Instead of trying to categorize *what* an object is, we first ask a much simpler physical question:
**"Is this cubic meter of space empty air, or is it filled with matter?"**

We divide the 3D world surrounding the car into a grid of volumetric pixels (**Voxels**):
- Voxel volume: $X \in [-40, 40]\text{ m}, Y \in [-40, 40]\text{ m}, Z \in [-2, 4]\text{ m}$.
- Voxel resolution: $\Delta x = 0.4\text{ m}$. Total voxels: $200 \times 200 \times 16 = 640,000$ cells.
- Every voxel stores a probability $P(\text{occupied}) \in [0, 1]$ and an occupancy semantic class.
- If a ladder falls onto the road, the voxels are occupied. The car doesn't need to know it's a ladder—it knows it cannot drive through solid matter!

### Temporal Memory: The ConvGRU State Bank
How do we remember objects behind walls?
We introduce a **Recurrent Neural Network** in Bird's-Eye View:
1. When the car moves from $t-1$ to $t$, the ego-vehicle translates and rotates.
2. We warp the previous memory bank $H_{t-1}$ using the car's odometry $\Delta \mathbf{x}, \Delta \theta$.
3. We pass the warped memory and the new camera observations into a **Convolutional Gated Recurrent Unit (ConvGRU)**.
4. The ConvGRU updates the belief state, maintaining occupied voxels behind occluders!

---

## 3. 🧪 Lab Mission: Hands-On Simulator Experiments

Scroll to the **Interactive 3D Occupancy Studio** at the top of this chapter:

1. **Experiment 1 (Occlusion Memory Loss)**:
   - In the simulator, toggle **Temporal Memory (ConvGRU)** to `OFF`.
   - Watch the animated pedestrian walk behind the parked bus.
   - *Observation*: The moment the pedestrian steps behind the bus, their red voxel signature vanishes instantly from the vehicle's perception grid.
2. **Experiment 2 (Persistent Object Permanence)**:
   - Toggle **Temporal Memory (ConvGRU)** to `ON`.
   - Let the pedestrian pass behind the bus again.
   - *Observation*: The red occupied voxel cluster persists and moves forward along the predicted walking vector even when 100% occluded by the bus!
3. **Experiment 3 (Voxel Resolution vs GPU Memory)**:
   - Toggle voxel grid resolution between $0.8\text{ m}$ (coarse) and $0.2\text{ m}$ (fine).
   - *Observation*: Notice the VRAM memory footprint display jump from 12 MB to 768 MB.

---

## 4. 🛠️ The Karpathy Build: Temporal ConvGRU from Scratch

Here is the exact spatiotemporal recurrence engine implemented in PyTorch, tracing the flow of hidden state tensors:

```python
import torch
import torch.nn as nn

class SpatialConvGRUCell(nn.Module):
    """
    Spatially-aware Convolutional GRU cell for BEV and Occupancy Memory.
    Maintains persistent volumetric hidden states across temporal frames.
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        # Convolutions for update gate (z) and reset gate (r)
        self.conv_gates = nn.Conv2d(
            in_channels + hidden_channels, 
            2 * hidden_channels, 
            kernel_size, 
            padding=padding
        )
        
        # Convolution for candidate hidden state (h_tilde)
        self.conv_candidate = nn.Conv2d(
            in_channels + hidden_channels, 
            hidden_channels, 
            kernel_size, 
            padding=padding
        )

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: (B, in_channels, H, W) current frame BEV observation
            h_prev: (B, hidden_channels, H, W) previous temporal state
        Returns:
            h_next: (B, hidden_channels, H, W) updated temporal state
        """
        combined = torch.cat([x, h_prev], dim=1)  # -> (B, in + hidden, H, W)
        gates = self.conv_gates(combined)         # -> (B, 2 * hidden, H, W)
        
        # Split into update gate z and reset gate r
        z_gate, r_gate = torch.split(gates, self.hidden_channels, dim=1)
        z = torch.sigmoid(z_gate)
        r = torch.sigmoid(r_gate)
        
        # Candidate state with reset gate applied
        combined_candidate = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.conv_candidate(combined_candidate))
        
        # Convex combination of previous and candidate state
        h_next = (1.0 - z) * h_prev + z * h_tilde
        return h_next
```

---

## 5. 📐 Mathematical Rigor: Ego-Motion Memory Warping

Before passing the previous memory state $H_{t-1}$ to the ConvGRU at time $t$, we must compensate for the vehicle's own physical movement:

$$\mathbf{p}_t = T_{t-1 \to t} \mathbf{p}_{t-1} = \begin{bmatrix} \cos(\Delta \theta) & -\sin(\Delta \theta) & \Delta X \\ \sin(\Delta \theta) & \cos(\Delta \theta) & \Delta Y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_{t-1} \\ Y_{t-1} \\ 1 \end{bmatrix}$$

We sample the warped tensor $H_{t-1}^{\text{warped}}$ using bilinear grid sampling:

$$H_{t-1}^{\text{warped}} = \operatorname{GridSample}\left(H_{t-1}, T_{t-1 \to t}^{-1}\right)$$

### 3D Occupancy Binary Cross-Entropy with Affinity Loss
To train voxel probabilities $p_v = \sigma(z_v)$ on ground truth $y_v \in \{0, 1\}$:

$$\mathcal{L}_{\text{occ}} = -\sum_{v \in \text{voxels}} \left[ w_1 y_v \log p_v + w_0 (1 - y_v) \log(1 - p_v) \right] + \lambda \mathcal{L}_{\text{affinity}}$$

Where $w_1 / w_0 \approx 20.0$ because $95\%$ of outdoor space is empty air.

---

## 6. 🩺 Andrew Ng's Diagnostic Field Guide

| Observed Symptom | Underlying Mathematical Mechanism | Verification Test | Production Fix |
| :--- | :--- | :--- | :--- |
| **Moving vehicles leave long red "ghost trails" in memory** | **Update Gate Stagnation ($z \approx 0$)**: The recurrent network is refusing to overwrite past occupied states with new empty air observations. | Inspect mean magnitude of update gate tensor `z.mean()`. | Add motion velocity flow vectors to the input tensor so the GRU can dynamically reset vacated voxels. |
| **The whole occupancy grid blurs when turning corners** | **Uncompensated Ego-Motion**: The temporal memory is being blended across frames without applying the rotation matrix $T_{t-1 \to t}^{-1}$. | Turn the vehicle sharply in place and check if stationary poles blur into arcs. | Apply bilinear `grid_sample` ego-motion warping before the ConvGRU recurrent step. |
| **GPU runs out of memory during backward pass** | **3D Tensor Explosion**: Storing all intermediate 3D pre-activation tensors across $T$ frames consumes $\mathcal{O}(B \cdot T \cdot C \cdot X \cdot Y \cdot Z)$ VRAM. | Profile peak memory allocation during training. | Use Gradient Checkpointing on ConvGRU cells, or collapse height dimension $Z$ before recurrent processing. |

---

## 7. 🎯 Self-Check: Test Your Mental Model

<details>
<summary><b>Q1: Why is 3D Occupancy fundamentally safer than 3D Bounding Boxes for an autonomous emergency braking system?</b></summary>

<br>

**Answer**: Bounding box detectors rely on supervised classification: if an object does not resemble the bounding boxes in the training distribution (e.g. an overturned boat, a mattress on the highway, fallen tree limbs), the detector assigns it a low confidence score and ignores it. 3D Occupancy is **class-agnostic geometry**: it measures whether a 3D physical volume is solid matter or empty drivable air. A vehicle will stop for solid matter regardless of whether it can identify what the object is called.
</details>

<details>
<summary><b>Q2: What happens if an autonomous vehicle relies on temporal memory without compensating for its own ego-motion?</b></summary>

<br>

**Answer**: If the car travels forward 5 meters between frames and blends past memory directly without coordinate warping, stationary obstacles will appear to drift toward the vehicle or stretch into elongated smeared corridors. A stationary light pole will be remembered at both its previous and current coordinates simultaneously, corrupting the free-space map.
</details>
