"""
train_lora_save.py
Stage 1: Train per-task LoRA QKV adapters and save weights + CLS samples.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from avalanche.benchmarks.classic import SplitCIFAR100
from tqdm import tqdm


@dataclass
class TrainConfig:
    backbone_name: str = "vit_base_patch16_224"
    embed_dim: int = 768
    num_classes: int = 100
    n_experiences: int = 10
    epochs_per_task: int = 10
    batch_size: int = 64
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    num_heads: int = 12
    rank: int = 16
    mlp_hidden: int = 2048
    cls_samples_per_task: int = 2048
    output_dir: str = "results/lora"


def pack_lora_weights(a_q, b_q, a_k, b_k, a_v, b_v) -> torch.Tensor:
    return torch.cat([
        a_q.flatten(), b_q.flatten(),
        a_k.flatten(), b_k.flatten(),
        a_v.flatten(), b_v.flatten(),
    ])


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

    def forward(self, x: torch.Tensor, lora: LoRAQKV) -> torch.Tensor:
        x_norm = self.norm1(x)
        q = self.q_base(x_norm) + self._apply_lora(x_norm, lora.a_q, lora.b_q)
        k = self.k_base(x_norm) + self._apply_lora(x_norm, lora.a_k, lora.b_k)
        v = self.v_base(x_norm) + self._apply_lora(x_norm, lora.a_v, lora.b_v)

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


class LoRAViT(nn.Module):
    def __init__(self, config: TrainConfig):
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

    def forward(self, x: torch.Tensor, lora: LoRAQKV) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.backbone.forward_features(x)
        if len(tokens.shape) == 2:
            tokens = tokens.unsqueeze(1)
        tokens = self.attn_block(tokens, lora)
        cls_out = tokens[:, 0]
        return self.classifier(cls_out)


def main():
    config = TrainConfig()
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    device = torch.device(config.device)
    print(f"Using device: {device}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = LoRAViT(config).to(device)
    benchmark = SplitCIFAR100(
        n_experiences=config.n_experiences,
        seed=config.seed,
        return_task_id=False,
    )

    for exp_id, experience in enumerate(benchmark.train_stream):
        print(f"\n{'='*60}")
        print(f"Experience {exp_id}: Classes {experience.classes_in_this_experience}")
        print(f"{'='*60}")

        lora = LoRAQKV(config.embed_dim, config.rank).to(device)
        params = list(lora.parameters()) + list(model.classifier.parameters())
        optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

        dataloader = DataLoader(
            experience.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        cls_feats: List[torch.Tensor] = []
        for epoch in range(config.epochs_per_task):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            for x, y, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs_per_task}"):
                x, y = x.to(device), y.to(device)
                x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

                optimizer.zero_grad()
                logits = model(x, lora)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

                if len(cls_feats) * config.batch_size < config.cls_samples_per_task:
                    with torch.no_grad():
                        tokens = model.backbone.forward_features(x)
                        if len(tokens.shape) == 2:
                            tokens = tokens.unsqueeze(1)
                        cls_feats.append(tokens[:, 0].detach().cpu())

            avg_loss = epoch_loss / len(dataloader)
            acc = correct / total if total > 0 else 0.0
            print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}, Acc = {acc:.2%}")

        cls_all = torch.cat(cls_feats, dim=0)
        if cls_all.size(0) > config.cls_samples_per_task:
            cls_all = cls_all[:config.cls_samples_per_task]

        artifact = {
            "lora_vector": lora.as_vector().detach().cpu(),
            "cls_samples": cls_all,
            "classifier_state": model.classifier.state_dict(),
            "task_id": exp_id,
        }
        torch.save(artifact, output_dir / f"task_{exp_id}.pt")
        print(f"Saved {output_dir / f'task_{exp_id}.pt'}")


if __name__ == "__main__":
    main()
