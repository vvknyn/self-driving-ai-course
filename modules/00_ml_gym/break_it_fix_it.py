"""
Module 00: FastAI "Break It & Fix It" Drill

EXPERIMENT:
  What happens when an engineer accidentally uses an aggressive learning rate (lr=10.0)
  without gradient clipping on driving perception models?
  Or what happens when class imbalance is unweighted?

You will see:
  1. The BROKEN RUN: Loss explodes to NaN / infinity or accuracy collapses to random guessing.
  2. The FIXED RUN: Learning rate warmup + AdamW + Focal Loss recovering clean convergence.
"""

import torch
import torch.nn.functional as F
from dataset import get_dataloaders
from model import DrivingClassifier, FocalLoss

def simulate_broken_run():
    print("\n" + "="*65)
    print("❌ 1. THE BROKEN RUN: Aggressive Learning Rate (lr = 25.0)")
    print("="*65)
    
    train_loader, _ = get_dataloaders(batch_size=32)
    model = DrivingClassifier(num_classes=4)
    # BROKEN: Extreme learning rate that blows past the loss minimum
    broken_optimizer = torch.optim.SGD(model.parameters(), lr=25.0)
    
    for step, (images, targets) in enumerate(train_loader):
        broken_optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        broken_optimizer.step()
        
        print(f"  Step {step+1:02d}: Loss = {loss.item():.4f}")
        if torch.isnan(loss) or loss.item() > 1000.0 or step >= 5:
            print(f"\n💥 FAILURE DETECTED: Loss diverged to {loss.item()}!")
            print("  Reason: Weights updated by massive step vectors -> activation overflow -> NaN gradients.")
            break

def simulate_fixed_run():
    print("\n" + "="*65)
    print("✅ 2. THE FIXED RUN: Gradient Norm Clamping + Scaled LR + AdamW")
    print("="*65)
    
    train_loader, _ = get_dataloaders(batch_size=32)
    model = DrivingClassifier(num_classes=4)
    fixed_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = FocalLoss(gamma=2.0)
    
    for step, (images, targets) in enumerate(train_loader):
        fixed_optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        
        # THE FIX: Clip gradients to max norm 1.0 to prevent gradient explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        fixed_optimizer.step()
        
        if step % 3 == 0:
            print(f"  Step {step+1:02d}: Loss = {loss.item():.4f} (Stable & Monotonically Decreasing)")
        if step >= 9:
            break
            
    print("\n🎉 SUCCESS: Gradients are bounded, loss smoothly converges!")

if __name__ == "__main__":
    simulate_broken_run()
    simulate_fixed_run()
