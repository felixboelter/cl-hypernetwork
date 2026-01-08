"""
baseline_vit_all.py
Upper-bound baseline: train on all 100 CIFAR-100 classes at once.
"""

from dataclasses import dataclass
from typing import Dict
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from torchvision import datasets
from torchvision import transforms
from avalanche.benchmarks.classic import SplitCIFAR100
from tqdm import tqdm


@dataclass
class BaselineConfig:
    backbone_name: str = "vit_base_patch16_224"
    num_classes: int = 100
    epochs: int = 10
    batch_size: int = 64
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    n_experiences: int = 10


def evaluate_overall(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_per_task(model: nn.Module, benchmark, device: torch.device) -> Dict[int, float]:
    model.eval()
    results = {}
    with torch.no_grad():
        for exp_id, exp in enumerate(benchmark.test_stream):
            dataloader = DataLoader(
                exp.dataset,
                batch_size=64,
                num_workers=4,
                pin_memory=True,
            )
            correct = 0
            total = 0
            for x, y, _ in dataloader:
                x, y = x.to(device), y.to(device)
                x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            results[exp_id] = correct / total if total > 0 else 0.0
    return results


def main():
    config = BaselineConfig()
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    device = torch.device(config.device)
    print(f"Using device: {device}")

    model = timm.create_model(
        config.backbone_name,
        pretrained=True,
        num_classes=config.num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            x, y = x.to(device), y.to(device)
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_loss = epoch_loss / len(train_loader)
        acc = correct / total if total > 0 else 0.0
        print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}, Acc = {acc:.2%}")

    overall_acc = evaluate_overall(model, test_loader, device)
    print(f"\nOverall Test Accuracy: {overall_acc:.2%}")

    benchmark = SplitCIFAR100(
        n_experiences=config.n_experiences,
        seed=config.seed,
        return_task_id=False,
    )
    per_task = evaluate_per_task(model, benchmark, device)
    print("\nPer-task Accuracy (Split-CIFAR-100):")
    for task_id, acc in per_task.items():
        print(f"  Task {task_id}: {acc:.2%}")

    csv_path = "results/baseline_vit_all_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["overall_acc"])
        writer.writerow([overall_acc])
        writer.writerow([])
        writer.writerow(["task_id", "acc"])
        for task_id, acc in per_task.items():
            writer.writerow([task_id, acc])


if __name__ == "__main__":
    main()
