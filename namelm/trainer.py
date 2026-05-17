import math
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from namelm.config import LLMConfig
from namelm.dataloader import create_train_val_dataloaders
from namelm.model import NameModel

_SAVE_DIR = Path(__file__).parent.parent / "models"


def _param_label(model: NameModel) -> str:
    n = sum(p.numel() for p in model.parameters())
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}b"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}m"
    return f"{n / 1_000:.1f}k"


def _save_path(model: NameModel) -> Path:
    _SAVE_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _SAVE_DIR / f"name-{_param_label(model)}_{timestamp}.pt"


def _val_loss(model, loader, loss_fn, config, device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            targets = batch["target_ids"].to(device)
            logits = model(inputs)
            total += loss_fn(logits.view(-1, config.vocab_size), targets.view(-1)).item()
    return total / len(loader)


def _plot(metrics: dict, save_path: Path):
    epochs = range(1, len(metrics["train_loss"]) + 1)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("Training Metrics", fontsize=14, fontweight="bold")

    # 1. Loss
    ax = axes[0, 0]
    ax.plot(epochs, metrics["train_loss"], marker="o", label="Train")
    ax.plot(epochs, metrics["val_loss"],   marker="s", label="Val")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Perplexity
    ax = axes[0, 1]
    ax.plot(epochs, [math.exp(l) for l in metrics["train_loss"]], marker="o", label="Train")
    ax.plot(epochs, [math.exp(l) for l in metrics["val_loss"]],   marker="s", label="Val")
    ax.set_title("Perplexity")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Learning rate
    ax = axes[1, 0]
    ax.plot(epochs, metrics["lr"], marker="o", color="green")
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(True, alpha=0.3)

    # 4. Gradient norm
    ax = axes[1, 1]
    ax.plot(epochs, metrics["grad_norm"], marker="o", color="orange")
    ax.set_title("Gradient Norm (avg per epoch)")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    # 5. Tokens per second
    ax = axes[2, 0]
    ax.plot(epochs, metrics["tokens_per_sec"], marker="o", color="purple")
    ax.set_title("Tokens / Second")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    # 6. Train/Val loss gap (overfitting indicator)
    ax = axes[2, 1]
    gap = [v - t for t, v in zip(metrics["train_loss"], metrics["val_loss"])]
    ax.bar(epochs, gap, color=["tomato" if g > 0 else "steelblue" for g in gap])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Val − Train Loss (overfitting gap)")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    plot_path = save_path.with_suffix(".png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


def train(config: LLMConfig, epochs: int = 10, batch_size: int = 64, lr: float = 3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")

    train_loader, val_loader = create_train_val_dataloaders(config, batch_size=batch_size)
    tokens_per_batch = batch_size * config.context_length
    print(f"Train size  : {len(train_loader.dataset):,} samples ({len(train_loader)} batches/epoch)")
    print(f"Val size    : {len(val_loader.dataset):,} samples")

    model = NameModel(config).to(device)
    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}\n")

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    metrics = {"train_loss": [], "val_loss": [], "lr": [], "grad_norm": [], "tokens_per_sec": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_norm, t0 = 0.0, 0.0, time.perf_counter()

        for batch in train_loader:
            inputs  = batch["input_ids"].to(device)
            targets = batch["target_ids"].to(device)

            logits = model(inputs)
            loss = loss_fn(logits.view(-1, config.vocab_size), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            total_norm += nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optimizer.step()
            total_loss += loss.item()

        elapsed = time.perf_counter() - t0
        scheduler.step()

        avg_train = total_loss / len(train_loader)
        avg_val   = _val_loss(model, val_loader, loss_fn, config, device)
        avg_norm  = total_norm / len(train_loader)
        tps       = (len(train_loader) * tokens_per_batch) / elapsed
        cur_lr    = scheduler.get_last_lr()[0]

        metrics["train_loss"].append(avg_train)
        metrics["val_loss"].append(avg_val)
        metrics["grad_norm"].append(avg_norm)
        metrics["tokens_per_sec"].append(tps)
        metrics["lr"].append(cur_lr)

        print(
            f"Epoch {epoch:>3}/{epochs}"
            f"  train: {avg_train:.4f}"
            f"  val: {avg_val:.4f}"
            f"  ppl: {math.exp(avg_val):.2f}"
            f"  grad: {avg_norm:.3f}"
            f"  tok/s: {tps:,.0f}"
            f"  lr: {cur_lr:.2e}"
        )

    save = _save_path(model)
    torch.save(model.state_dict(), save)
    print(f"\nSaved to {save}")

    _plot(metrics, save)

    return model


if __name__ == "__main__":
    config = LLMConfig()
    train(config, epochs=10, batch_size=64, lr=3e-4)
