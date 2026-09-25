"""
Driving Scenario Generator for Capstone Evaluation
Synthesizes dynamic multi-camera video streams and ground truth obstacles:
  - Lead car in ego lane slowing down
  - Overtaking car in right lane
"""

import torch
import numpy as np

class DrivingScenario:
    def __init__(self, num_frames: int = 30, dt: float = 0.1):
        self.num_frames = num_frames
        self.dt = dt
        self.current_step = 0
        
        # Initial positions
        self.lead_x = 22.0
        self.lead_y = 0.0
        self.lead_v = 13.0
        
        self.side_x = 10.0
        self.side_y = -3.5
        self.side_v = 16.0

    def step(self):
        t = self.current_step * self.dt
        
        # Lead vehicle begins braking after step 10
        if self.current_step > 10:
            self.lead_v = max(6.0, self.lead_v - 0.4)
            
        self.lead_x += self.lead_v * self.dt
        self.side_x += self.side_v * self.dt
        
        obstacles = [
            {"name": "Lead Car", "x": self.lead_x, "y": self.lead_y, "v": self.lead_v},
            {"name": "Right Lane Car", "x": self.side_x, "y": self.side_y, "v": self.side_v}
        ]
        
        # Simulate 3-camera synthetic video batch (1, N_cams=3, C=3, H=128, W=256)
        # In a physical rig or dataset, this reads from cameras or nuScenes logs!
        cam_batch = torch.randn(1, 3, 3, 128, 256) * 0.1
        
        self.current_step += 1
        return cam_batch, obstacles
