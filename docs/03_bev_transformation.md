# Chapter 04: Bird's-Eye View (BEV) Transform (Lift, Splat, Shoot)

> **The Big Question**: *Your car has 8 surrounding cameras taking flat 2D perspective pictures. But your path planner needs a flat, top-down 2D map to steer without crashing into curbs. How do you lift pixels off flat camera photos and drop them onto a bird's-eye view ground plane when you have no LiDAR and don't know the exact depth of anything?*

---

## 1. 🚨 The Real-World Dilemma: The Vanishing Distance Problem

Imagine driving toward a pedestrian standing 50 meters away:
- In your windshield camera, that pedestrian occupies a 20-pixel tall sliver near the image horizon.
- If the pedestrian takes 3 steps toward you (moving to 40 meters), their height in pixels barely changes by **2 pixels**.
- But if a pedestrian 3 meters in front of your bumper takes 3 steps toward you, their image size **triples**, exploding across 400 pixels!

```
Perspective View (Camera):               Bird's-Eye View (Ego Ground Plane):
┌──────────────────────────────┐         ┌──────────────────────────────┐
│  [Pedestrian: 20px @ 50m]    │         │          [Pedestrian @ 50m]  │
│                              │         │              ▲               │
│                              │         │              │ 10m           │
│                              │         │              ▼               │
│                              │         │          [Pedestrian @ 40m]  │
│  [Pedestrian: 400px @ 3m]    │         │                              │
│                              │         │              ▲ 37m           │
│                              │         │              ▼               │
└──────────────────────────────┘         │          [Car Bumper (0,0)]  │
Non-linear, distorted distances!         └──────────────────────────────┘
                                         Linear, Euclidean metric space!
```

> [!CAUTION]
> **Why Autonomous Cars Cannot Plan in Perspective Space**  
> In a camera image, Euclidean geometry is broken. A 10-meter distance near the horizon corresponds to 2 pixels, while a 10-meter distance near the hood corresponds to 800 pixels. **You cannot compute braking distances, collision trajectories, or steering curvature in pixel coordinates.** You must transform everything into a top-down metric coordinate system: **Bird's-Eye View (BEV)**.

---

## 2. 💡 The Mental Model: Why Simple Geometry Fails

### The Naive Idea: Inverse Perspective Mapping (IPM)
In Chapter 02, we learned that if we assume the entire world is a flat tabletop, we can multiply pixel coordinates by a homography matrix $H^{-1}$ to stretch the image onto the ground.

### ❓ Socratic Challenge: Why does IPM fail catastrophically in the city?
Think about what happens to a 4-meter tall box truck when you project it with IPM:
- IPM assumes **every pixel touches the asphalt**.
- The wheels of the truck touch the asphalt at distance $d = 15\text{ m}$.
- The top of the truck is 4 meters up in the air. The ray connecting the camera through the roof hits the ground **80 meters behind the truck**!
- Result: IPM smears the truck into a terrifying 70-meter long blur across three lanes, causing the car to slam on phantom brakes!

### The Breakthrough: Lift-Splat-Shoot (Philion & Fidler, ECCV 2020)
Instead of pretending the world is flat, we decompose the problem into three transparent steps:
1. **LIFT**: For every pixel, ask the neural network: *"What is the probability this pixel is at distance 2m, 3m, 4m, ... 50m?"* Create a cloud of visual features along the camera's sightline.
2. **SPLAT**: Use camera calibration matrices to project all 3D points down onto a flat top-down grid (like tossing sand grains onto a tabletop).
3. **SHOOT**: Sum all features that fall into the same $(X, Y)$ ground grid cell using pooling, creating a clean BEV feature tensor.

---

## 3. 🧪 Lab Mission: Hands-On Simulator Experiments

Scroll up to the **Interactive BEV Studio** at the top of this chapter and complete these 3 experiments:

1. **Experiment 1 (The Depth Uncertainty Smear)**:
   - Select the `Front Camera` feed.
   - Adjust the **Depth Softmax Temperature ($\tau$)** slider from `0.1` to `2.5`.
   - *Observation*: Notice how at $\tau = 2.5$, the vehicle representation in the BEV grid smears out like an elongated cigar along the line of sight. At $\tau = 0.2$, the depth distribution collapses to a crisp point.
2. **Experiment 2 (Multi-Camera Overlap Splatting)**:
   - Toggle on `Front Left Camera` and `Front Right Camera`.
   - Observe the overlapping purple and cyan frustum cones.
   - *Observation*: Where the cameras overlap, the BEV pooling combines features from two completely separate vantage points into a single cohesive obstacle representation!

---

## 4. 🛠️ The Karpathy Build: Lift-Splat from Raw Scratch

Here is the entire mathematical core of Lift-Splat-Shoot implemented in readable, standalone PyTorch. Notice the explicit tracking of tensor shapes at every single line:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LiftSplat(nn.Module):
    """
    Transforms multi-view camera feature maps into a unified 3D Bird's-Eye View.
    Reference: Philion & Fidler (ECCV 2020)
    """
    def __init__(self, D=40, d_min=1.0, d_max=50.0, C=64):
        super().__init__()
        self.D = D  # Number of discrete depth bins along the ray
        self.C = C  # Feature channels per pixel
        
        # Discrete depth bins: e.g. [1.0m, 2.25m, ..., 50.0m]
        self.depth_bins = torch.linspace(d_min, d_max, D)
        
        # Depth prediction sub-network (predicts categorical distribution + context)
        self.depth_net = nn.Conv2d(in_channels=128, out_channels=D + C, kernel_size=1)

    def lift(self, img_features):
        """
        LIFTS 2D features into 3D frustum rays using outer-product depth distribution.
        
        Input:
            img_features: (B * N_cam, 128, H, W)
        Returns:
            frustum_features: (B * N_cam, D, H, W, C)
        """
        B_N, _, H, W = img_features.shape
        
        # 1. Forward pass through 1x1 conv
        logits = self.depth_net(img_features)            # -> (B*N, D + C, H, W)
        
        # 2. Split into depth distribution and semantic context
        depth_logits = logits[:, :self.D, :, :]           # -> (B*N, D, H, W)
        context = logits[:, self.D:, :, :]                # -> (B*N, C, H, W)
        
        # 3. Softmax along depth dimension (probabilities sum to 1.0 along the ray)
        depth_prob = F.softmax(depth_logits, dim=1)       # -> (B*N, D, H, W)
        
        # 4. Outer Product: lift features onto the depth ray
        # (B*N, D, 1, H, W) * (B*N, 1, C, H, W) -> (B*N, D, C, H, W)
        frustum = depth_prob.unsqueeze(2) * context.unsqueeze(1)
        
        # Permute to (B*N, D, H, W, C)
        return frustum.permute(0, 1, 3, 4, 2)
```

> [!TIP]
> **Tensor Tracing Rule of Thumb**  
> Notice step 4: `depth_prob.unsqueeze(2) * context.unsqueeze(1)`. This is an **outer product**. We are broadcasting the scalar depth probability across all 64 feature channels. If depth bin $d=12\text{ m}$ has probability $0.85$, that entire 64-dimensional feature vector gets weighted by $0.85$.

---

## 5. 📐 Mathematical Rigor: The Coordinate Geometry

Let a point in camera image space be $(u, v)$ with discrete depth candidate $d$.

### Step 1: Unprojecting from Image Pixels to Camera 3D
Using the intrinsic matrix $K \in \mathbb{R}^{3 \times 3}$:

$$\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = d \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

Where $K$ is defined by focal lengths $(f_x, f_y)$ and optical center $(c_x, c_y)$:

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}, \quad K^{-1} = \begin{bmatrix} \frac{1}{f_x} & 0 & -\frac{c_x}{f_x} \\ 0 & \frac{1}{f_y} & -\frac{c_y}{f_y} \\ 0 & 0 & 1 \end{bmatrix}$$

### Step 2: Transforming Camera 3D to Ego-Vehicle Metric Coordinates
Using extrinsic rotation $R_{\text{ext}} \in SO(3)$ and translation $\mathbf{t}_{\text{ext}} \in \mathbb{R}^3$:

$$\mathbf{p}_{\text{ego}} = R_{\text{ext}} \mathbf{p}_c + \mathbf{t}_{\text{ext}}$$

### Step 3: Quantizing into Metric BEV Voxel Bins
Given grid bounds $X \in [-50, 50]\text{ m}$ and resolution $\Delta x = 0.5\text{ m/pixel}$:

$$x_{\text{grid}} = \left\lfloor \frac{X_{\text{ego}} - X_{\min}}{\Delta x} \right\rfloor$$

---

## 6. 🩺 Andrew Ng's Diagnostic Field Guide

When debugging BEV perception in a production autonomous vehicle stack, consult this table:

| Observed Symptom | Underlying Mathematical Cause | Diagnostic Verification Test | Engineering Fix |
| :--- | :--- | :--- | :--- |
| **Objects appear stretched into radials pointing at camera** | Depth distribution entropy is too high; model is guessing uniform depth probabilities. | Plot $-\sum p_i \log p_i$ across depth bins. If entropy $> 3.0$, model has zero depth confidence. | Add explicit auxiliary depth supervision with sparse LiDAR or stereo pseudo-ground truth. |
| **All obstacles shift 1.5m to the right during acceleration** | Pitch/Squat dynamic misalignment. Hard acceleration tilts the chassis up $2^\circ$, rotating $R_{\text{ext}}$. | Log pitch angle from vehicle IMU vs estimated horizon in image feed. | Feed live vehicle suspension IMU pitch/roll into camera extrinsics $R_{\text{ext}}(t)$. |
| **BEV pooling runs at 4 FPS (too slow for real-time)** | Naive `torch.unique` or slow scatter operations on GPU memory. | Profile kernel latency with PyTorch Profiler (`nsys nvprof`). | Use GPU Cumulative Sum trick (`cumsum` trick from Philion & Fidler) or custom Triton kernel. |

---

## 7. 🎯 Self-Check: Test Your Mental Model

<details>
<summary><b>Q1: Why does Lift-Splat-Shoot predict a probability distribution over depths instead of a single scalar depth value (e.g. depth = 14.2m)?</b></summary>

<br>

**Answer**: Because monocular depth estimation is mathematically ill-posed. A small dark car at 20 meters and a large dark SUV at 30 meters can project identical pixel shapes. Predicting a single scalar depth forces the network to make an overconfident, potentially catastrophic error. A categorical probability distribution allows the network to express **epistemic uncertainty** (e.g. 40% chance at 20m, 60% chance at 30m). Downstream temporal tracking and BEV fusion can then resolve this ambiguity over successive video frames.
</details>

<details>
<summary><b>Q2: If an autonomous car drives up a steep $15^\circ$ hill while the camera calibration matrix assumes flat ground, where will an obstacle appear on the BEV map?</b></summary>

<br>

**Answer**: The obstacle will appear **much closer than it actually is**. Because the camera is tilted upward relative to the road surface, rays that strike the inclined road hit earlier in 3D camera space. Without dynamic pitch correction from the chassis IMU, the perception system will think the road is an obstacle directly in front of the bumper, causing a false emergency stop.
</details>
