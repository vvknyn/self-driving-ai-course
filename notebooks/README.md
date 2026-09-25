# 📓 Interactive Jupyter Notebooks for Autonomous Driving Course

Every chapter in the Self-Driving AI course is available as a **standalone, self-contained Jupyter Notebook** runnable in **Google Colab with 1-click** (no GPU required, zero local installation).

| Chapter | Notebook Title | Google Colab | Source File |
| :--- | :--- | :---: | :--- |
| **00** | Neural Networks, Autograd & Loss Landscapes | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/notebooks/00_neural_networks_and_autograd.ipynb) | [`00_neural_networks_and_autograd.ipynb`](./00_neural_networks_and_autograd.ipynb) |
| **01** | Driving Perception Gym & Class Imbalance (Focal Loss) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/notebooks/01_driving_perception_gym.ipynb) | [`01_driving_perception_gym.ipynb`](./01_driving_perception_gym.ipynb) |
| **02** | 3D Camera Rig & Inverse Perspective Mapping (IPM) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/notebooks/02_camera_geometry_and_ipm.ipynb) | [`02_camera_geometry_and_ipm.ipynb`](./02_camera_geometry_and_ipm.ipynb) |
| **03** | Multi-Task HydraNet Perception & Uncertainty Loss | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/notebooks/03_hydranet_multitask_learning.ipynb) | [`03_hydranet_multitask_learning.ipynb`](./03_hydranet_multitask_learning.ipynb) |
| **04** | Bird's-Eye View (BEV) Transform (Lift-Splat-Shoot) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/notebooks/04_bev_lift_splat_shoot.ipynb) | [`04_bev_lift_splat_shoot.ipynb`](./04_bev_lift_splat_shoot.ipynb) |
| **05** | 3D Occupancy Networks & Temporal Memory (ConvGRU) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/notebooks/05_3d_occupancy_and_temporal_memory.ipynb) | [`05_3d_occupancy_and_temporal_memory.ipynb`](./05_3d_occupancy_and_temporal_memory.ipynb) |
| **06** | Vector Space Tracking & Multi-Object Association | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/notebooks/06_vector_space_tracking_kalman.ipynb) | [`06_vector_space_tracking_kalman.ipynb`](./06_vector_space_tracking_kalman.ipynb) |
| **07** | Lattice Trajectory Planning & Quintic Splines | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/notebooks/07_lattice_trajectory_planning.ipynb) | [`07_lattice_trajectory_planning.ipynb`](./07_lattice_trajectory_planning.ipynb) |
| **08** | Closed-Loop Control & Vehicle Kinematics (Stanley) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/notebooks/08_closed_loop_control_stanley.ipynb) | [`08_closed_loop_control_stanley.ipynb`](./08_closed_loop_control_stanley.ipynb) |
| **09** | Full FSD Capstone & Real-Time System Architecture | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/notebooks/09_full_fsd_system_architecture.ipynb) | [`09_full_fsd_system_architecture.ipynb`](./09_full_fsd_system_architecture.ipynb) |

---

### Local Execution
```bash
# Clone the repository
git clone https://github.com/vvknyn/self-driving-ai-course.git
cd self-driving-ai-course

# Install dependencies
pip install torch numpy matplotlib jupyterlab

# Launch JupyterLab
jupyter lab notebooks/
```
