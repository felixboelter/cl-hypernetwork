"""
baseline_vit_joint_split.py
Upper-bound baseline: train on the union of all Split-CIFAR-100 tasks.
"""

from dataclasses import dataclass
from typing import Dict
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import timm
from avalanche.benchmarks.classic import SplitCIFAR100
from tqdm import tqdm


@dataclass
class BaselineConfig:
    backbone_name: str = "vit_base_patch16_224"
    num_classes: int = 100
    n_experiences: int = 10
    epochs_per_task: int = 3
    batch_size: int = 64
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


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

    benchmark = SplitCIFAR100(
        n_experiences=config.n_experiences,
        seed=config.seed,
        return_task_id=False,
    )
    train_datasets = [exp.dataset for exp in benchmark.train_stream]
    joint_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(
        joint_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    total_epochs = config.epochs_per_task * config.n_experiences
    for epoch in range(total_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}"):
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

    per_task = evaluate_per_task(model, benchmark, device)
    print("\nPer-task Accuracy (Split-CIFAR-100):")
    for task_id, acc in per_task.items():
        print(f"  Task {task_id}: {acc:.2%}")

    csv_path = "results/baseline_vit_joint_split_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "acc"])
        for task_id, acc in per_task.items():
            writer.writerow([task_id, acc])


if __name__ == "__main__":
    main()
