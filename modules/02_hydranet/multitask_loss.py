"""
Multi-Task Loss Balancing via Homoscedastic Uncertainty
Reference: Kendall, Gal, & Cipolla (CVPR 2018)
"Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"

PROBABILISTIC DERIVATION (MITx Probability Connection):
  Suppose task i has model prediction f_i(x) and ground truth y_i with Gaussian noise ~ N(0, sigma_i^2).
  The conditional likelihood of observing y_i is:
    P(y_i | f_i(x), sigma_i) = (1 / sqrt(2*pi*sigma_i^2)) * exp( - ||y_i - f_i(x)||^2 / (2 * sigma_i^2) )
    
  Taking the negative log-likelihood:
    -log P(y_i | ...) = (1 / (2 * sigma_i^2)) * ||y_i - f_i(x)||^2 + log(sigma_i) + const
    
  Let s_i = log(sigma_i^2) for numerical stability.
  Then:
    1 / (sigma_i^2) = exp(-s_i)
    log(sigma_i) = 0.5 * s_i
    
  Total Multi-Task Loss:
    L_total = sum_i [ 0.5 * exp(-s_i) * L_i + 0.5 * s_i ]
    
  Notice the automatic trade-off:
    - If task i has a large loss L_i, the optimizer increases s_i to damp its gradient.
    - But it cannot increase s_i indefinitely because of the + 0.5 * s_i penalty!
"""

import torch
import torch.nn as nn

class UncertaintyMultiTaskLoss(nn.Module):
    def __init__(self, num_tasks: int = 4):
        super().__init__()
        # Learnable log variances s_i = log(sigma_i^2), initialized to 0.0 (sigma = 1.0)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks, dtype=torch.float32))

    def forward(self, task_losses: list) -> torch.Tensor:
        """
        task_losses: list of scalar tensors [L_lane, L_freespace, L_vehicle, L_traffic_light]
        """
        total_loss = 0.0
        for i, loss in enumerate(task_losses):
            s = self.log_vars[i]
            # 0.5 * exp(-s) * Loss + 0.5 * s
            precision = torch.exp(-s)
            weighted_task_loss = 0.5 * precision * loss + 0.5 * s
            total_loss = total_loss + weighted_task_loss
            
        return total_loss

    def get_task_weights(self) -> list:
        """Returns the effective weighting factor 0.5 * exp(-s_i) for each task."""
        with torch.no_grad():
            return (0.5 * torch.exp(-self.log_vars)).tolist()
