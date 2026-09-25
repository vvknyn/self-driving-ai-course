# Deep Dive: 3D Occupancy Networks & Spatiotemporal Dynamics

> "Bounding boxes fail on ontology cracks—objects that defy standard classification. 3D Occupancy treats the physical world as a continuous voxel field." — Ashok Elluswamy, Tesla AI Day 2022

---

## 1. The Breakdown of 3D Bounding Boxes
Traditional 3D object detection assumes every obstacle belongs to a predefined list of classes:
$$\mathcal{C} \in \{\text{Car}, \text{Truck}, \text{Pedestrian}, \text{Cyclist}\}$$

When a car encounters:
- An overturned semi-trailer with a flatbed sticking out into the lane,
- A large fallen tree branch hanging at windshield height,
- A road construction barricade with irregular geometry,
- A mattress falling off a roof rack,
Bounding box detectors predict **zero bounding boxes** because none of their predefined class anchors match!

---

## 2. The 3D Occupancy Field Representation
Occupancy Networks discretize 3D physical volume around the vehicle into a dense grid of voxels:
$$V(i_x, i_y, i_z), \quad i_x \in [0, N_x-1], \quad i_y \in [0, N_y-1], \quad i_z \in [0, N_z-1]$$
Typical resolution is $\Delta s = 0.4\text{m} \times 0.4\text{m} \times 0.4\text{m}$.

For every voxel, the network outputs:
1. **Bernoulli Occupancy Probability**:
   $$P(\text{occupied} = 1) = \sigma(z_{\text{occ}}) \in [0, 1]$$
2. **Dynamic 3D Velocity Flow Vector**:
   $$\vec{v} = (\Delta X, \Delta Y, \Delta Z) \in \mathbb{R}^3$$
   Describing how fast that voxel is moving relative to the ego car.

---

## 3. Spatiotemporal Fusion via ConvGRU
Driving occurs over continuous video time steps $t=0, 1, 2, \dots$.
Single-frame perception suffers from **Occlusion Amnesia**: if a pedestrian walks behind a bus or a truck blocks a camera view, single-frame models drop the detection immediately.

A **Convolutional Gated Recurrent Unit (ConvGRU)** preserves state over time:
$$R_t = \sigma(W_r * X_t + U_r * H_{t-1}) \quad \text{(Reset Gate)}$$
$$Z_t = \sigma(W_z * X_t + U_z * H_{t-1}) \quad \text{(Update Gate)}$$
$$\tilde{H}_t = \tanh(W_h * X_t + U_h * (R_t \odot H_{t-1})) \quad \text{(Candidate Memory)}$$
$$H_t = (1 - Z_t) \odot H_{t-1} + Z_t \odot \tilde{H}_t \quad \text{(Updated State)}$$

### Bayesian Filtering Interpretation (MITx Probability)
- $H_{t-1}$ acts as the **Prior Distribution** over world geometry from past video history.
- $X_t$ acts as the **Observation Likelihood** from the current camera frames.
- $H_t$ acts as the **Posterior State Update**, carrying forward occluded objects and projecting them along their estimated velocity vectors $\vec{v}$.
