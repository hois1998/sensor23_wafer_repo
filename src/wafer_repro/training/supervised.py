from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wafer_repro.training.registry import TRAINER_REGISTRY


@TRAINER_REGISTRY.register("supervised_torch")
class SupervisedTorchTrainer:
    def __init__(self, task, device, use_amp: bool) -> None:
        self.task = task
        self.device = device
        self.use_amp = use_amp

    def train_epoch(self, model, loader, criterion, optimizer) -> dict[str, float]:
        return run_epoch(
            model,
            loader,
            criterion,
            optimizer,
            self.device,
            self.use_amp,
            train=True,
            task=self.task,
        )

    def validate(self, model, loader, criterion) -> dict[str, float]:
        return run_epoch(
            model,
            loader,
            criterion,
            None,
            self.device,
            False,
            train=False,
            task=self.task,
        )


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    use_amp: bool,
    train: bool,
    task,
) -> dict[str, float]:
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    desc = "train" if train else "val"

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((preds == labels).sum().detach().cpu())
        total_count += batch_size
        y_true.append(labels.detach().cpu().numpy())
        y_pred.append(preds.detach().cpu().numpy())

    true = np.concatenate(y_true)
    pred = np.concatenate(y_pred)
    return task.summarize_epoch(true, pred, total_loss, total_correct, total_count)


def cpu_state_dict(model) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_optimizer(model, config: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(config.get("name", "adam")).lower()
    lr = float(config.get("lr", 1e-4))
    weight_decay = float(config.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(config.get("momentum", 0.0))
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f"Unknown optimizer: {name}")
