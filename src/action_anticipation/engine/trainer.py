"""Small training loop kept independent from any experiment tracker."""

from collections.abc import Iterable

import torch
from torch import Tensor, nn


def compute_loss(logits: Tensor, target: Tensor, loss_fn: nn.Module, supervise_all_horizons: bool) -> Tensor:
    if not supervise_all_horizons:
        return loss_fn(logits[:, -1], target)
    targets = target.unsqueeze(1).expand(-1, logits.shape[1]).reshape(-1)
    return loss_fn(logits.reshape(-1, logits.shape[-1]), targets)


def train_epoch(
    model: nn.Module,
    batches: Iterable[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
    supervise_all_horizons: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in batches:
        features = batch["features"].to(device)
        target = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = compute_loss(logits, target, loss_fn, supervise_all_horizons)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * target.shape[0]
        total_examples += target.shape[0]
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    batches: Iterable[dict[str, Tensor]],
    device: torch.device,
    loss_fn: nn.Module,
    supervise_all_horizons: bool,
    metric_horizon: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total_examples = 0
    for batch in batches:
        features = batch["features"].to(device)
        target = batch["target"].to(device)
        logits = model(features)
        loss = compute_loss(logits, target, loss_fn, supervise_all_horizons)
        total_loss += loss.item() * target.shape[0]
        correct += (logits[:, metric_horizon].argmax(dim=-1) == target).sum().item()
        total_examples += target.shape[0]
    return {"loss": total_loss / max(total_examples, 1), "top1": correct / max(total_examples, 1)}
