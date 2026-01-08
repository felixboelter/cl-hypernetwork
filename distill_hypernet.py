"""
distill_hypernet.py
Stage 2: Distill a hypernetwork to reproduce LoRA weights from CLS features.
"""

from dataclasses import dataclass
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import csv
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import timm
from avalanche.benchmarks.classic import SplitCIFAR100
from tqdm import tqdm


@dataclass
class DistillConfig:
    embed_dim: int = 768
    rank: int = 16
    hyper_hidden: int = 256
    distill_epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lora_dir: str = "results/lora"
    backbone_name: str = "vit_base_patch16_224"
    num_classes: int = 100
    num_heads: int = 12
    mlp_hidden: int = 2048
    n_experiences: int = 10
    router_hidden: int = 256
    router_epochs: int = 20
    continual_mgda: bool = True
    replay_prev: bool = True
    replay_batch_size: int = 256
    eval_after_each: bool = False
    continual_eval_csv: str = "results/hypernet_eval_results_continual.csv"


def unpack_lora_weights(vec: torch.Tensor, dim: int, rank: int):
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


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class HyperNet(nn.Module):
    def __init__(self, cond_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.cls_norm = nn.LayerNorm(cond_dim)
        self.project = nn.Linear(cond_dim, hidden)
        self.blocks = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        cond = self.cls_norm(cond)
        x = F.relu(self.project(cond))
        x = self.blocks(x)
        return self.head(x)


class Router(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_tasks: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_tasks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HyperLoRAAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, rank: int, mlp_hidden: int):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

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
    def __init__(self, config: DistillConfig):
        super().__init__()
        self.backbone = timm.create_model(
            config.backbone_name,
            pretrained=True,
            num_classes=0,
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

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


def mgda_combine(loss_old: torch.Tensor, loss_new: torch.Tensor, params: List[torch.Tensor]):
    grads_new = torch.autograd.grad(loss_new, params, retain_graph=True, allow_unused=True)
    if loss_old is None:
        loss_new.backward()
        return
    grads_old = torch.autograd.grad(loss_old, params, allow_unused=True)
    g_new_flat = torch.cat([g.flatten() for g in grads_new if g is not None])
    g_old_flat = torch.cat([g.flatten() for g in grads_old if g is not None])
    if g_new_flat.numel() == 0 or g_old_flat.numel() == 0:
        loss_new.backward()
        return
    dot_old_old = (g_old_flat * g_old_flat).sum()
    dot_new_new = (g_new_flat * g_new_flat).sum()
    dot_old_new = (g_old_flat * g_new_flat).sum()
    denom = dot_old_old + dot_new_new - 2 * dot_old_new
    alpha = (dot_old_old - dot_old_new) / (denom + 1e-8)
    alpha = torch.clamp(alpha, 0.0, 1.0)
    for param, g_old, g_new in zip(params, grads_old, grads_new):
        if g_old is None and g_new is None:
            continue
        if g_old is None:
            param.grad = g_new
            continue
        if g_new is None:
            param.grad = g_old
            continue
        param.grad = (1.0 - alpha) * g_old + alpha * g_new


def evaluate_hypernet(
    hypernet: HyperNet,
    shared_head: nn.Linear,
    benchmark: SplitCIFAR100,
    config: DistillConfig,
    device: torch.device,
    max_task_id: int,
    lora_dim: int,
) -> Dict[int, float]:
    model = HyperViT(config).to(device)
    model.classifier = shared_head
    results: Dict[int, float] = {}
    for exp_id, exp in enumerate(benchmark.test_stream):
        if exp_id > max_task_id:
            break
        dataloader = DataLoader(
            exp.dataset,
            batch_size=64,
            num_workers=4,
            pin_memory=True,
        )
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y, _ in dataloader:
                x, y = x.to(device), y.to(device)
                x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                tokens = model.backbone.forward_features(x)
                if len(tokens.shape) == 2:
                    tokens = tokens.unsqueeze(1)
                cls = tokens[:, 0]
                cls_mean = cls.mean(dim=0, keepdim=True)
                lora_vec = hypernet(cls_mean).squeeze(0)
                lora_weights = unpack_lora_weights(lora_vec, config.embed_dim, config.rank)
                tokens = model.attn_block(tokens, lora_weights)
                cls_out = tokens[:, 0]
                logits = model.classifier(cls_out)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        results[exp_id] = correct / total if total > 0 else 0.0
    return results


def evaluate_saved_heads(
    classifier_states: Dict[str, Dict[str, torch.Tensor]],
    router: Router,
    benchmark: SplitCIFAR100,
    config: DistillConfig,
    device: torch.device,
) -> Dict[int, float]:
    model = HyperViT(config).to(device)
    results: Dict[int, float] = {}
    for exp_id, exp in enumerate(benchmark.test_stream):
        key = f"task_{exp_id}"
        state = classifier_states.get(key)
        if state is None:
            continue
        model.classifier.load_state_dict(state)
        dataloader = DataLoader(
            exp.dataset,
            batch_size=64,
            num_workers=4,
            pin_memory=True,
        )
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y, _ in dataloader:
                x, y = x.to(device), y.to(device)
                x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                tokens = model.backbone.forward_features(x)
                if len(tokens.shape) == 2:
                    tokens = tokens.unsqueeze(1)
                cls = tokens[:, 0]
                if router is not None:
                    task_logits = router(cls)
                    task_pred = task_logits.argmax(dim=1).mode().values.item()
                    state = classifier_states.get(f"task_{task_pred}")
                    if state is not None:
                        model.classifier.load_state_dict(state)
                logits = model.classifier(cls)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        results[exp_id] = correct / total if total > 0 else 0.0
    return results


def run_diagnostics(
    hypernet: HyperNet,
    shared_head: nn.Linear,
    task_files: List[Path],
    benchmark: SplitCIFAR100,
    config: DistillConfig,
    device: torch.device,
    samples_per_task: int,
    out_path: Path,
) -> None:
    hypernet.eval()
    shared_head.eval()

    weights_per_task: List[torch.Tensor] = []
    for path in task_files:
        data = torch.load(path, map_location="cpu")
        cls_samples = data["cls_samples"][:samples_per_task].to(device)
        with torch.no_grad():
            cls_mean = cls_samples.mean(dim=0, keepdim=True)
            weights = hypernet(cls_mean).squeeze(0)
        weights_per_task.append(weights.detach().cpu().flatten())

    pairwise_path = out_path.with_name(out_path.stem + "_pairwise.csv")
    with open(pairwise_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_i", "task_j", "mean_abs_diff"])
        for i in range(len(weights_per_task)):
            for j in range(i + 1, len(weights_per_task)):
                diff = (weights_per_task[i] - weights_per_task[j]).abs().mean().item()
                writer.writerow([i, j, diff])

    pred_path = out_path.with_name(out_path.stem + "_pred_tasks.csv")
    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "pred_task", "count", "total"])
        model = HyperViT(config).to(device)
        model.classifier = shared_head
        for exp_id, exp in enumerate(benchmark.test_stream):
            dataloader = DataLoader(exp.dataset, batch_size=64, num_workers=4, pin_memory=True)
            counts = torch.zeros(config.n_experiences, dtype=torch.long)
            total = 0
            with torch.no_grad():
                for x, y, _ in dataloader:
                    x = x.to(device)
                    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                    tokens = model.backbone.forward_features(x)
                    if len(tokens.shape) == 2:
                        tokens = tokens.unsqueeze(1)
                    cls = tokens[:, 0]
                    cls_mean = cls.mean(dim=0, keepdim=True)
                    lora_vec = hypernet(cls_mean).squeeze(0)
                    lora_weights = unpack_lora_weights(lora_vec, config.embed_dim, config.rank)
                    tokens = model.attn_block(tokens, lora_weights)
                    cls_out = tokens[:, 0]
                    logits = model.classifier(cls_out)
                    preds = logits.argmax(dim=1)
                    pred_tasks = (preds // 10).clamp(min=0, max=config.n_experiences - 1)
                    counts += torch.bincount(pred_tasks.cpu(), minlength=config.n_experiences)
                    total += preds.size(0)
                    if total >= samples_per_task:
                        break
            for pred_task, count in enumerate(counts.tolist()):
                writer.writerow([exp_id, pred_task, count, total])


def continual_distill_mgda(
    config: DistillConfig,
    task_files: List[Path],
    task_classes: List[List[int]],
    device: torch.device,
) -> Tuple[HyperNet, nn.Linear, Dict[str, Dict[str, torch.Tensor]]]:
    cond_dim = config.embed_dim
    first_data = torch.load(task_files[0], map_location="cpu")
    first_vec = first_data["lora_vector"]
    lora_dim = int(first_vec.numel())
    classifier_state = first_data.get("classifier_state")
    if classifier_state is None:
        raise ValueError("Missing classifier_state in task files.")
    hypernet = HyperNet(cond_dim, config.hyper_hidden, lora_dim).to(device)
    shared_head = nn.Linear(config.embed_dim, config.num_classes).to(device)
    optimizer = torch.optim.AdamW(
        list(hypernet.parameters()) + list(shared_head.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    classifier_states: Dict[str, Dict[str, torch.Tensor]] = {}
    prev_replay: List[Tuple[torch.Tensor, torch.Tensor]] = []
    prev_head_targets: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for task_idx, path in enumerate(task_files):
        data = torch.load(path, map_location="cpu")
        cls_samples = data["cls_samples"]
        lora_vec = data["lora_vector"].view(1, -1)
        classifier_state = data.get("classifier_state")
        if classifier_state is None:
            raise ValueError(f"Missing classifier_state in {path}")
        head_w = classifier_state["weight"].contiguous()
        head_b = classifier_state["bias"].contiguous()
        classifier_states[path.stem] = data.get("classifier_state")
        class_ids = torch.tensor(task_classes[task_idx], dtype=torch.long, device=device)

        cond_task = cls_samples
        target_lora_full = lora_vec.repeat(cond_task.size(0), 1)
        dataset = TensorDataset(cond_task, target_lora_full)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        for epoch in range(config.distill_epochs):
            hypernet.train()
            epoch_loss = 0.0
            batch_iter = tqdm(
                dataloader,
                desc=f"[task {task_idx}] distill {epoch+1}/{config.distill_epochs}",
                leave=False,
            )
            for batch in batch_iter:
                cond, target_lora = batch
                cond = cond.to(device)
                target_lora = target_lora.to(device)
                optimizer.zero_grad()
                pred_new = hypernet(cond)
                loss_new_lora = F.mse_loss(pred_new, target_lora)
                if head_w.size(0) == class_ids.numel():
                    head_w_task = head_w.to(device)
                    head_b_task = head_b.to(device)
                else:
                    head_w_task = head_w.to(device)[class_ids]
                    head_b_task = head_b.to(device)[class_ids]
                loss_new_head = (
                    F.mse_loss(shared_head.weight[class_ids], head_w_task)
                    + F.mse_loss(shared_head.bias[class_ids], head_b_task)
                )
                loss_new = loss_new_lora + loss_new_head

                loss_old = None
                if config.replay_prev and prev_replay:
                    prev_cls = torch.cat([r[0] for r in prev_replay], dim=0)
                    prev_targets = torch.cat([r[1] for r in prev_replay], dim=0)
                    if prev_cls.size(0) > config.replay_batch_size:
                        idx = torch.tensor(
                            random.sample(range(prev_cls.size(0)), config.replay_batch_size)
                        )
                        prev_cls = prev_cls[idx]
                        prev_targets = prev_targets[idx]
                    prev_cls = prev_cls.to(device)
                    prev_targets = prev_targets.to(device)
                    pred_old = hypernet(prev_cls)
                    loss_old_lora = F.mse_loss(pred_old, prev_targets)
                    loss_old_head = 0.0
                    for old_task_id, (old_w, old_b) in prev_head_targets.items():
                        old_class_ids = torch.tensor(
                            task_classes[old_task_id],
                            dtype=torch.long,
                            device=device,
                        )
                        loss_old_head += (
                            F.mse_loss(shared_head.weight[old_class_ids], old_w.to(device))
                            + F.mse_loss(shared_head.bias[old_class_ids], old_b.to(device))
                        )
                    loss_old = loss_old_lora + loss_old_head
                    batch_iter.set_postfix(
                        loss_new_lora=f"{loss_new_lora.item():.4f}",
                        loss_new_head=f"{loss_new_head.item():.4f}",
                        loss_old_lora=f"{loss_old_lora.item():.4f}",
                        loss_old_head=f"{float(loss_old_head):.4f}",
                        replay_batch=prev_cls.size(0),
                    )

                mgda_combine(
                    loss_old,
                    loss_new,
                    list(hypernet.parameters()) + list(shared_head.parameters()),
                )
                optimizer.step()
                epoch_loss += loss_new.item()
            avg_loss = epoch_loss / len(dataloader)
            print(f"[task {task_idx}] Epoch {epoch+1}: Distill Loss = {avg_loss:.6f}")

        prev_replay.append((cond_task.detach().cpu(), target_lora_full.detach().cpu()))
        prev_head_targets[task_idx] = (
            head_w_task.detach().cpu(),
            head_b_task.detach().cpu(),
        )

        if config.eval_after_each:
            benchmark = SplitCIFAR100(
                n_experiences=config.n_experiences,
                seed=42,
                return_task_id=False,
            )
            results = evaluate_hypernet(
                hypernet,
                shared_head,
                benchmark,
                config,
                device,
                task_idx,
                lora_dim,
            )
            out_path = Path(config.continual_eval_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "a", newline="") as f:
                writer = csv.writer(f)
                if out_path.stat().st_size == 0:
                    writer.writerow(["experience", "task_id", "acc"])
                for task_id, acc in results.items():
                    writer.writerow([task_idx, task_id, acc])

    return hypernet, shared_head, classifier_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--diagnose-samples", type=int, default=256)
    parser.add_argument("--diagnose-out", default="results/hypernet_diagnostics.csv")
    args, _ = parser.parse_known_args()

    config = DistillConfig()
    device = torch.device(config.device)
    print(f"Using device: {device}")

    lora_dir = Path(config.lora_dir)
    task_files = sorted(lora_dir.glob("task_*.pt"))
    if not task_files:
        raise FileNotFoundError(f"No task files found in {lora_dir}")

    router_features: List[torch.Tensor] = []
    router_labels: List[torch.Tensor] = []
    benchmark = SplitCIFAR100(
        n_experiences=config.n_experiences,
        seed=42,
        return_task_id=False,
    )
    task_classes = [list(exp.classes_in_this_experience) for exp in benchmark.train_stream]

    if config.continual_mgda:
        hypernet, shared_head, classifier_states = continual_distill_mgda(
            config, task_files, task_classes, device
        )
        avg_loss = 0.0
    else:
        cond_list: List[torch.Tensor] = []
        target_list: List[torch.Tensor] = []

        classifier_states = {}
        for path in task_files:
            data = torch.load(path, map_location="cpu")
            cls_samples = data["cls_samples"]
            lora_vec = data["lora_vector"]
            classifier_states[path.stem] = data.get("classifier_state")
            task_id = data.get("task_id")
            if task_id is not None:
                router_features.append(cls_samples)
                router_labels.append(torch.full((cls_samples.size(0),), int(task_id), dtype=torch.long))
            cond_list.append(cls_samples)
            target_list.append(lora_vec.repeat(cls_samples.size(0), 1))

        cond_stack = torch.cat(cond_list, dim=0)
        target_stack = torch.cat(target_list, dim=0)

        dataset = TensorDataset(cond_stack, target_stack)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        weight_dim = target_stack.size(1)
        hypernet = HyperNet(config.embed_dim, config.hyper_hidden, weight_dim).to(device)
        shared_head = nn.Linear(config.embed_dim, config.num_classes).to(device)
        optimizer = torch.optim.AdamW(hypernet.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        for epoch in range(config.distill_epochs):
            hypernet.train()
            epoch_loss = 0.0
            for cond, target in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.distill_epochs}"):
                cond = cond.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                pred = hypernet(cond)
                loss = F.mse_loss(pred, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            print(f"  Epoch {epoch+1}: Distill Loss = {avg_loss:.6f}")

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(hypernet.state_dict(), out_dir / "hypernet.pt")
    torch.save(shared_head.state_dict(), out_dir / "shared_head.pt")

    csv_path = out_dir / "hypernet_distill_loss.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["final_loss"])
        writer.writerow([avg_loss])

    print(f"Saved hypernet to {out_dir / 'hypernet.pt'}")

    # Train router on CLS features
    router = Router(config.embed_dim, config.router_hidden, config.n_experiences).to(device)
    if not config.continual_mgda and router_features and router_labels:
        router_x = torch.cat(router_features, dim=0)
        router_y = torch.cat(router_labels, dim=0)
        router_ds = TensorDataset(router_x, router_y)
        router_loader = DataLoader(router_ds, batch_size=config.batch_size, shuffle=True)
        router_opt = torch.optim.AdamW(router.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        for epoch in range(config.router_epochs):
            router.train()
            epoch_loss = 0.0
            for x, y in router_loader:
                x, y = x.to(device), y.to(device)
                router_opt.zero_grad()
                logits = router(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                router_opt.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(router_loader)
            print(f"  Router Epoch {epoch+1}: Loss = {avg_loss:.6f}")

    # Evaluate hypernet with router-selected classifiers
    device = torch.device(config.device)
    model = HyperViT(config).to(device)
    results = {}
    first_vec = torch.load(task_files[0], map_location="cpu")["lora_vector"]
    lora_dim = int(first_vec.numel())
    results = evaluate_hypernet(
        hypernet,
        shared_head,
        benchmark,
        config,
        device,
        config.n_experiences - 1,
        lora_dim,
    )
    saved_head_results = evaluate_saved_heads(
        classifier_states,
        router if router_features else None,
        benchmark,
        config,
        device,
    )
    compare_path = out_dir / "hypernet_eval_results_compare.csv"
    with open(compare_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "task_id", "acc"])
        for task_id, acc in results.items():
            writer.writerow(["shared_head", task_id, acc])
        for task_id, acc in saved_head_results.items():
            writer.writerow(["saved_head", task_id, acc])

    csv_path = out_dir / "hypernet_eval_results_continual.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "acc"])
        for task_id, acc in results.items():
            writer.writerow([task_id, acc])

    if args.diagnose:
        run_diagnostics(
            hypernet,
            shared_head,
            task_files,
            benchmark,
            config,
            device,
            args.diagnose_samples,
            Path(args.diagnose_out),
        )


if __name__ == "__main__":
    main()
