"""
hyper_vit_cifar100.py
Hypernetwork distillation of LoRA QKV weights for Split-CIFAR-100.

Stage 1: Train per-task LoRA adapters on QKV for a lightweight attention block.
Stage 2: Distill LoRA weights into a hypernetwork conditioned on input CLS features.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from avalanche.benchmarks.classic import SplitCIFAR100
from tqdm import tqdm


@dataclass
class HyperConfig:
    backbone_name: str = "vit_base_patch16_224"
    embed_dim: int = 768
    num_classes: int = 100
    n_experiences: int = 10
    epochs_per_task: int = 10
    distill_epochs: int = 20
    batch_size: int = 64
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    num_heads: int = 12
    rank: int = 16
    hyper_hidden: int = 256
    mlp_hidden: int = 2048
    distill_samples_per_task: int = 1024


def compute_cl_metrics(results_matrix: List[Dict[int, float]]) -> Dict[str, float]:
    n_tasks = len(results_matrix)
    if n_tasks == 0:
        return {"avg_acc": 0.0, "learning_acc": 0.0, "bwt": 0.0, "forgetting": 0.0}

    final_accs = [results_matrix[-1].get(j, 0.0) for j in range(n_tasks)]
    avg_acc = float(sum(final_accs) / len(final_accs))

    learning_accs = [results_matrix[j].get(j, 0.0) for j in range(n_tasks)]
    learning_acc = float(sum(learning_accs) / len(learning_accs))

    bwt_vals = []
    forgetting_vals = []
    for j in range(n_tasks - 1):
        best_after_training = results_matrix[j].get(j, 0.0)
        final_acc = results_matrix[-1].get(j, 0.0)
        bwt_vals.append(final_acc - best_after_training)
        forgetting_vals.append(best_after_training - final_acc)

    bwt = float(sum(bwt_vals) / len(bwt_vals)) if bwt_vals else 0.0
    forgetting = float(sum(forgetting_vals) / len(forgetting_vals)) if forgetting_vals else 0.0

    return {
        "avg_acc": avg_acc,
        "learning_acc": learning_acc,
        "bwt": bwt,
        "forgetting": forgetting,
    }


def pack_lora_weights(a_q: torch.Tensor, b_q: torch.Tensor,
                      a_k: torch.Tensor, b_k: torch.Tensor,
                      a_v: torch.Tensor, b_v: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        a_q.flatten(), b_q.flatten(),
        a_k.flatten(), b_k.flatten(),
        a_v.flatten(), b_v.flatten(),
    ])


def unpack_lora_weights(vec: torch.Tensor, dim: int, rank: int) -> Tuple[torch.Tensor, ...]:
    a_size = dim * rank
    b_size = rank * dim
    idx = 0
    a_q = vec[idx:idx + a_size].view(dim, rank)
    idx += a_size
    b_q = vec[idx:idx + b_size].view(rank, dim)
    idx += b_size
    a_k = vec[idx:idx + a_size].view(dim, rank)
    idx += a_size
    b_k = vec[idx:idx + b_size].view(rank, dim)
    idx += b_size
    a_v = vec[idx:idx + a_size].view(dim, rank)
    idx += a_size
    b_v = vec[idx:idx + b_size].view(rank, dim)
    return a_q, b_q, a_k, b_k, a_v, b_v


class LoRAQKV(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.a_q = nn.Parameter(torch.randn(dim, rank) * 0.02)
        self.b_q = nn.Parameter(torch.randn(rank, dim) * 0.02)
        self.a_k = nn.Parameter(torch.randn(dim, rank) * 0.02)
        self.b_k = nn.Parameter(torch.randn(rank, dim) * 0.02)
        self.a_v = nn.Parameter(torch.randn(dim, rank) * 0.02)
        self.b_v = nn.Parameter(torch.randn(rank, dim) * 0.02)

    def as_vector(self) -> torch.Tensor:
        return pack_lora_weights(self.a_q, self.b_q, self.a_k, self.b_k, self.a_v, self.b_v)


class HyperNet(nn.Module):
    def __init__(self, cond_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)


class HyperLoRAAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, rank: int, mlp_hidden: int):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rank = rank

        self.q_base = nn.Linear(dim, dim, bias=True)
        self.k_base = nn.Linear(dim, dim, bias=True)
        self.v_base = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        for p in self.q_base.parameters():
            p.requires_grad = False
        for p in self.k_base.parameters():
            p.requires_grad = False
        for p in self.v_base.parameters():
            p.requires_grad = False
        for p in self.out_proj.parameters():
            p.requires_grad = False

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def _apply_lora(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(torch.matmul(x, a), b)

    def forward(self, x: torch.Tensor, lora_weights: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        a_q, b_q, a_k, b_k, a_v, b_v = lora_weights
        x_norm = self.norm1(x)
        q = self.q_base(x_norm) + self._apply_lora(x_norm, a_q, b_q)
        k = self.k_base(x_norm) + self._apply_lora(x_norm, a_k, b_k)
        v = self.v_base(x_norm) + self._apply_lora(x_norm, a_v, b_v)

        bsz, n_tokens, _ = x.shape
        q = q.view(bsz, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, n_tokens, self.dim)

        x = x + self.out_proj(out)
        x = x + self.mlp(self.norm2(x))
        return x


class HyperViT(nn.Module):
    def __init__(self, config: HyperConfig):
        super().__init__()
        self.backbone = timm.create_model(
            config.backbone_name,
            pretrained=True,
            num_classes=0,
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.dim = config.embed_dim
        self.attn_block = HyperLoRAAttention(
            dim=config.embed_dim,
            num_heads=config.num_heads,
            rank=config.rank,
            mlp_hidden=config.mlp_hidden,
        )
        self.classifier = nn.Linear(config.embed_dim, config.num_classes)

    def forward_with_lora(self, x: torch.Tensor, lora_weights: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.backbone.forward_features(x)
        if len(tokens.shape) == 2:
            tokens = tokens.unsqueeze(1)
        tokens = self.attn_block(tokens, lora_weights)
        cls_out = tokens[:, 0]
        return self.classifier(cls_out)


def train_task_lora(model: HyperViT, lora: LoRAQKV, experience, config: HyperConfig, device: torch.device):
    model.train()
    params = list(lora.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    dataloader = DataLoader(
        experience.dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    total_loss = 0.0
    cls_feats = []

    for epoch in range(config.epochs_per_task):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for x, y, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs_per_task}"):
            x, y = x.to(device), y.to(device)
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

            optimizer.zero_grad()
            logits = model.forward_with_lora(x, (
                lora.a_q, lora.b_q, lora.a_k, lora.b_k, lora.a_v, lora.b_v
            ))
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            with torch.no_grad():
                tokens = model.backbone.forward_features(x)
                if len(tokens.shape) == 2:
                    tokens = tokens.unsqueeze(1)
                cls_feats.append(tokens[:, 0].detach().cpu())

        avg_loss = epoch_loss / len(dataloader)
        acc = correct / total
        print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}, Acc = {acc:.2%}")
        total_loss += avg_loss

    if cls_feats:
        cls_all = torch.cat(cls_feats, dim=0)
    else:
        cls_all = torch.empty((0, config.embed_dim))
    return total_loss / config.epochs_per_task, cls_all


def distill_hypernet(hypernet: HyperNet,
                     cond_vectors: List[torch.Tensor],
                     target_weights: List[torch.Tensor],
                     config: HyperConfig,
                     device: torch.device):
    hypernet.train()
    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    cond_stack = torch.cat(cond_vectors, dim=0).to(device)
    target_stack = torch.cat(target_weights, dim=0).to(device)

    for epoch in range(config.distill_epochs):
        optimizer.zero_grad()
        pred = hypernet(cond_stack)
        loss = F.mse_loss(pred, target_stack)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"  Distill Epoch {epoch+1}: Loss = {loss.item():.6f}")


def evaluate_with_hypernet(model: HyperViT,
                           hypernet: HyperNet,
                           benchmark,
                           up_to_exp: int,
                           config: HyperConfig,
                           device: torch.device):
    model.eval()
    hypernet.eval()
    results = {}
    with torch.no_grad():
        for exp_id in range(up_to_exp + 1):
            exp = benchmark.test_stream[exp_id]
            dataloader = DataLoader(
                exp.dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                pin_memory=True,
            )
            correct = 0
            total = 0
            for x, y, _ in dataloader:
                x, y = x.to(device), y.to(device)
                x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                tokens = model.backbone.forward_features(x)
                if len(tokens.shape) == 2:
                    tokens = tokens.unsqueeze(1)
                cls = tokens[:, 0]
                weight_vec = hypernet(cls)
                a_q, b_q, a_k, b_k, a_v, b_v = unpack_lora_weights(
                    weight_vec, config.embed_dim, config.rank
                )
                lora_weights = (
                    a_q.mean(dim=0),
                    b_q.mean(dim=0),
                    a_k.mean(dim=0),
                    b_k.mean(dim=0),
                    a_v.mean(dim=0),
                    b_v.mean(dim=0),
                )
                logits = model.forward_with_lora(x, lora_weights)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            results[exp_id] = correct / total
    return results


def main():
    config = HyperConfig()
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    device = torch.device(config.device)
    print(f"Using device: {device}")

    model = HyperViT(config).to(device)
    benchmark = SplitCIFAR100(
        n_experiences=config.n_experiences,
        seed=config.seed,
        return_task_id=False,
    )

    lora_modules = [LoRAQKV(config.embed_dim, config.rank).to(device) for _ in range(config.n_experiences)]
    cond_vectors: List[torch.Tensor] = []
    target_weights: List[torch.Tensor] = []

    # Train task-specific LoRA adapters
    for exp_id, experience in enumerate(benchmark.train_stream):
        print(f"\\n{'='*60}")
        print(f"Experience {exp_id}: Classes {experience.classes_in_this_experience}")
        print(f"{'='*60}")

        loss, cls_all = train_task_lora(model, lora_modules[exp_id], experience, config, device)
        if cls_all.numel() == 0:
            continue
        if cls_all.size(0) > config.distill_samples_per_task:
            idx = torch.randperm(cls_all.size(0))[:config.distill_samples_per_task]
            cls_all = cls_all[idx]
        cond_vectors.append(cls_all)
        lora_vec = lora_modules[exp_id].as_vector().detach().cpu()
        target_weights.append(lora_vec.repeat(cls_all.size(0), 1))
        print(f"Task {exp_id} training loss: {loss:.4f}")

    # Distill hypernetwork to reproduce LoRA weights
    weight_dim = target_weights[0].numel()
    hypernet = HyperNet(config.embed_dim, config.hyper_hidden, weight_dim).to(device)
    print("\\nDistilling hypernetwork...")
    distill_hypernet(hypernet, cond_vectors, target_weights, config, device)

    # Evaluate with hypernetwork-generated LoRA weights
    results_matrix = []
    csv_path = "results/hyper_vit_cifar100_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "exp_id",
            "task_id",
            "task_acc",
            "avg_acc",
            "learning_acc",
            "bwt",
            "forgetting",
        ])
    for exp_id in range(config.n_experiences):
        print("\\nEvaluating on all tasks...")
        task_results = evaluate_with_hypernet(model, hypernet, benchmark, exp_id, config, device)
        results_matrix.append(task_results)

        cl_metrics = compute_cl_metrics(results_matrix)
        print(f"\\nResults after task {exp_id}:")
        for task_id, acc in task_results.items():
            print(f"  Task {task_id}: {acc:.2%}")
        print("\\nMetrics:")
        print(f"  Average Accuracy:  {cl_metrics['avg_acc']:.2%}")
        print(f"  Learning Accuracy: {cl_metrics['learning_acc']:.2%}")
        if exp_id > 0:
            print(f"  Backward Transfer: {cl_metrics['bwt']:.2%}")
            print(f"  Forgetting:        {cl_metrics['forgetting']:.2%}")

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for task_id, acc in task_results.items():
                writer.writerow([
                    exp_id,
                    task_id,
                    acc,
                    cl_metrics["avg_acc"],
                    cl_metrics["learning_acc"],
                    cl_metrics["bwt"],
                    cl_metrics["forgetting"],
                ])


if __name__ == "__main__":
    main()
