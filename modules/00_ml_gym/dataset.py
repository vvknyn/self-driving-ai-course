"""
Driving Patch Dataset Generator
Synthesizes realistic driving image crops across 4 critical classes:
  0: Clear Road (asphalt surface with texture)
  1: Lead Vehicle (rear profile with taillights and shadow)
  2: Pedestrian (vertical profile on road edge)
  3: Lane Marking (high-contrast yellow/white stripe on road)

Shape conventions:
  Images: Tensor of shape (B, 3, 64, 64) normalized to [0, 1]
  Labels: Tensor of shape (B,) with integer class IDs 0..3
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

CLASS_NAMES = ["Clear Road", "Lead Vehicle", "Pedestrian", "Lane Marking"]

class DrivingPatchDataset(Dataset):
    def __init__(self, num_samples: int = 1200, seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Realistic class distribution: road is dominant (60%), vehicle (20%), lane (15%), pedestrian (5%)
        probs = [0.60, 0.20, 0.05, 0.15]
        self.labels = np.random.choice(4, size=num_samples, p=probs)
        self.images = []
        
        for label in self.labels:
            img = self._generate_sample(label)
            self.images.append(img)
            
        self.images = torch.stack(self.images, dim=0) # (N, 3, 64, 64)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def _generate_sample(self, label: int) -> torch.Tensor:
        # Base asphalt background: dark gray with micro-texture noise
        base = np.random.normal(loc=0.25, scale=0.04, size=(64, 64, 3)).clip(0.15, 0.35)
        
        if label == 0:
            # Clear Road: pure asphalt texture, subtle lighting gradient
            grad = np.linspace(0.9, 1.1, 64)[:, None, None]
            img = base * grad
        elif label == 1:
            # Lead Vehicle: rear silhouette
            img = base.copy()
            # Vehicle body rectangle
            img[20:50, 16:48, :] = [0.1, 0.15, 0.4] # Metallic blue/dark body
            # Roof / rear window
            img[22:34, 20:44, :] = [0.05, 0.05, 0.08]
            # Red taillights
            img[36:42, 18:24, :] = [0.85, 0.1, 0.1]
            img[36:42, 40:46, :] = [0.85, 0.1, 0.1]
            # Ground contact shadow
            img[50:54, 14:50, :] = [0.05, 0.05, 0.05]
        elif label == 2:
            # Pedestrian: vertical silhouette
            img = base.copy()
            # Head (circle/oval)
            img[16:22, 30:34, :] = [0.75, 0.6, 0.5]
            # Torso (jacket)
            img[22:42, 28:36, :] = [0.15, 0.4, 0.2]
            # Legs
            img[42:58, 29:32, :] = [0.1, 0.1, 0.15]
            img[42:58, 33:36, :] = [0.1, 0.1, 0.15]
        elif label == 3:
            # Lane Marking: high-contrast white or yellow painted stripe
            img = base.copy()
            color = [0.95, 0.95, 0.95] if np.random.rand() > 0.5 else [0.95, 0.85, 0.1]
            # Angled or straight line
            x_start = np.random.randint(24, 36)
            for y in range(64):
                x = int(x_start + (y - 32) * 0.15)
                if 0 <= x < 60:
                    img[y, x:x+4, :] = color

        # Add Gaussian sensor noise
        noise = np.random.normal(0, 0.02, size=img.shape)
        img = (img + noise).clip(0.0, 1.0)
        # Convert HWC -> CHW float32 tensor
        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def get_dataloaders(batch_size: int = 32, train_split: float = 0.8):
    full_ds = DrivingPatchDataset(num_samples=1000)
    train_size = int(len(full_ds) * train_split)
    val_size = len(full_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(batch_size=16)
    x, y = next(iter(train_loader))
    print(f"✅ DrivingPatchDataset initialized successfully!")
    print(f"Batch image tensor shape: {x.shape} (B, C, H, W)")
    print(f"Batch label tensor shape: {y.shape} (B,)")
    print(f"Sample classes in batch: {[CLASS_NAMES[i] for i in y[:5].tolist()]}")
