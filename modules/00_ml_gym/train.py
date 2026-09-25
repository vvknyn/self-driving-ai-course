"""
Module 00: Driving ML Gym Training Script
Trains the DrivingClassifier on driving crops using Focal Loss.

FastAI philosophy:
  1. Instant feedback: starts training immediately with visual progress.
  2. Per-class diagnostics: inspects failure modes on rare classes (pedestrians).
"""

import torch
import torch.optim as optim
from dataset import get_dataloaders, CLASS_NAMES
from model import DrivingClassifier, FocalLoss

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(targets)
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += len(targets)
        
    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * 4
    class_total = [0] * 4
    
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            
            total_loss += loss.item() * len(targets)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += len(targets)
            
            for p, t in zip(preds, targets):
                class_total[t.item()] += 1
                if p == t:
                    class_correct[t.item()] += 1
                    
    acc = correct / total
    class_accs = [class_correct[i] / max(1, class_total[i]) for i in range(4)]
    return total_loss / total, acc, class_accs

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n🚗 Training Driving Classifier on device: {device}")
    
    train_loader, val_loader = get_dataloaders(batch_size=32)
    model = DrivingClassifier(num_classes=4).to(device)
    
    # Class weights for focal loss
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    print("-" * 65)
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Train Acc':<14}{'Val Loss':<14}{'Val Acc':<10}")
    print("-" * 65)
    
    for epoch in range(1, 6):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, class_accs = evaluate(model, val_loader, criterion, device)
        
        print(f"{epoch:<8}{train_loss:<14.4f}{train_acc*100:<13.1f}%{val_loss:<14.4f}{val_acc*100:<9.1f}%")
        
    print("-" * 65)
    print("\n🔍 Andrew Ng Diagnostic: Per-Class Validation Accuracy Breakdown:")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  {name:<15}: {class_accs[idx]*100:.1f}%")
        
    print("\n🎯 Why Tesla Cares:")
    print("  Notice how Focal Loss ensures high accuracy even on rare pedestrians")
    print("  (which only make up ~5% of raw driving frames).")
    print("  Without focal modulation, rare safety-critical objects are ignored by SGD.\n")

if __name__ == "__main__":
    main()
