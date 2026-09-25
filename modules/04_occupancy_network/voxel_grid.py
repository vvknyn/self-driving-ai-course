"""
3D Voxel Coordinate Space & Synthetic Scene Occupancy Generator
Represents a dense 3D volume around the vehicle ego frame:
  X (forward): [0.0, 32.0] meters (resolution: 0.5m -> 64 voxels)
  Y (lateral): [-12.0, 12.0] meters (resolution: 0.5m -> 48 voxels)
  Z (height):  [-1.0, 3.0] meters (resolution: 0.5m -> 8 voxels)
"""

import torch
import numpy as np

class VoxelGridConfig:
    def __init__(
        self,
        x_range=(0.0, 32.0),
        y_range=(-12.0, 12.0),
        z_range=(-1.0, 3.0),
        voxel_size=0.5
    ):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
        self.voxel_size = voxel_size
        
        self.nx = int(round((self.x_max - self.x_min) / voxel_size)) # 64
        self.ny = int(round((self.y_max - self.y_min) / voxel_size)) # 48
        self.nz = int(round((self.z_max - self.z_min) / voxel_size)) # 8

    def point_to_voxel_index(self, x, y, z):
        ix = int((x - self.x_min) / self.voxel_size)
        iy = int((y - self.y_min) / self.voxel_size)
        iz = int((z - self.z_min) / self.voxel_size)
        return ix, iy, iz

    def is_inside(self, ix, iy, iz):
        return 0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz

def generate_synthetic_3d_occupancy(
    config: VoxelGridConfig,
    lead_x: float = 16.0,
    lead_y: float = 0.0,
    lead_vx: float = 0.0,
    include_barrier: bool = True
) -> dict:
    """
    Synthesizes ground truth 3D binary occupancy and 3D velocity vectors.
    Returns:
      occupancy: Tensor of shape (1, 1, NX, NY, NZ) with values 0 (free) or 1 (occupied)
      velocity: Tensor of shape (1, 3, NX, NY, NZ) with 3D velocities [vx, vy, vz]
    """
    occ = np.zeros((config.nx, config.ny, config.nz), dtype=np.float32)
    vel = np.zeros((3, config.nx, config.ny, config.nz), dtype=np.float32)
    
    # 1. Lead Vehicle: bounding box [lead_x - 2, lead_x + 2], [lead_y - 1, lead_y + 1], [0.0, 1.5]
    for x in np.arange(lead_x - 2.0, lead_x + 2.0, config.voxel_size):
        for y in np.arange(lead_y - 1.0, lead_y + 1.0, config.voxel_size):
            for z in np.arange(0.0, 1.5, config.voxel_size):
                ix, iy, iz = config.point_to_voxel_index(x, y, z)
                if config.is_inside(ix, iy, iz):
                    occ[ix, iy, iz] = 1.0
                    vel[0, ix, iy, iz] = lead_vx # Moving along X axis
                    
    # 2. Side Highway Concrete Barrier at Y = -5.0m
    if include_barrier:
        for x in np.arange(0.0, 30.0, config.voxel_size):
            for y in np.arange(-5.2, -4.8, config.voxel_size):
                for z in np.arange(0.0, 0.8, config.voxel_size):
                    ix, iy, iz = config.point_to_voxel_index(x, y, z)
                    if config.is_inside(ix, iy, iz):
                        occ[ix, iy, iz] = 1.0
                        # Barrier is static (velocity = 0.0)

    # 3. Overhanging Tree Branch / Sign at X = 12m, Y = 2m, Z = 2.2m to 2.8m (LiDAR often misses this!)
    for x in np.arange(11.0, 13.0, config.voxel_size):
        for y in np.arange(1.0, 3.5, config.voxel_size):
            for z in np.arange(2.0, 2.8, config.voxel_size):
                ix, iy, iz = config.point_to_voxel_index(x, y, z)
                if config.is_inside(ix, iy, iz):
                    occ[ix, iy, iz] = 1.0

    occ_tensor = torch.tensor(occ).unsqueeze(0).unsqueeze(0) # (1, 1, NX, NY, NZ)
    vel_tensor = torch.tensor(vel).unsqueeze(0)             # (1, 3, NX, NY, NZ)
    return {"occupancy": occ_tensor, "velocity": vel_tensor}
