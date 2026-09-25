"""
Module 03: FastAI "Break It & Fix It" Drill

EXPERIMENT:
  What happens when depth estimation fails and collapses into a uniform distribution?
  In 2D-to-BEV lifting, if the network outputs equal probability for all depths:
  P(D = d) = 1 / num_depth_bins.

You will see:
  1. THE BROKEN RUN: Uniform depth collapse.
     A single car at 20 meters is smeared into a 40-meter radial streak across BEV space!
     Downstream collision planners hallucinate phantom obstacles everywhere along the ray.
  2. THE FIXED RUN: High-confidence categorical depth distribution.
     Entropy drops, and the vehicle features concentrate cleanly in a tight 2m x 4m box.
"""

import torch
import torch.nn.functional as F
import numpy as np

def run_drill():
    num_bins = 20
    depth_bins = torch.linspace(2.0, 42.0, num_bins)
    
    print("\n" + "="*70)
    print("💥 1. THE BROKEN RUN: Uniform Depth Collapse (High Entropy Smearing)")
    print("="*70)
    
    # Broken: Flat logits -> uniform softmax probabilities (Entropy = log(20) = 2.99)
    flat_logits = torch.zeros(1, num_bins, 1, 1)
    broken_probs = F.softmax(flat_logits, dim=1).squeeze()
    
    entropy_broken = -(broken_probs * torch.log(broken_probs + 1e-9)).sum().item()
    print(f"Depth Probability Distribution Entropy: {entropy_broken:.2f} (Max possible entropy!)")
    print(f"Probabilities along ray: {broken_probs[:5].tolist()}... (all {1/num_bins:.3f})")
    
    # Feature spread along ray
    spread_meters = depth_bins[-1] - depth_bins[0]
    print(f"🚨 RAY SMEARING: Obstacle is smeared across {spread_meters:.1f} meters of space!")
    print("   The car cannot tell whether the obstacle is 5 meters ahead or 40 meters ahead!")

    print("\n" + "="*70)
    print("✅ 2. THE FIXED RUN: Peaked Categorical Distribution (MITx Softmax Temperature)")
    print("="*70)
    
    # Fixed: Distinct peaked logits around true depth = 20 meters (bin index 9)
    true_bin = 9
    peaked_logits = torch.full((1, num_bins, 1, 1), -5.0)
    peaked_logits[0, true_bin, 0, 0] = 8.0 # Strong peak
    
    fixed_probs = F.softmax(peaked_logits, dim=1).squeeze()
    entropy_fixed = -(fixed_probs * torch.log(fixed_probs + 1e-9)).sum().item()
    
    print(f"Depth Probability Distribution Entropy: {entropy_fixed:.4f} (Near-zero uncertainty)")
    print(f"Peak probability at {depth_bins[true_bin]:.1f}m: {fixed_probs[true_bin].item()*100:.1f}%")
    print(f"🎉 SUCCESS: Obstacle is localized within ±{(depth_bins[1]-depth_bins[0])/2:.2f} meters!")
    print("\n💡 Tesla AI Lesson:")
    print("This is why Tesla uses self-supervised video multi-view depth consistency")
    print("to train sharp depth predictions without needing expensive LiDAR ground truth.")

if __name__ == "__main__":
    run_drill()
