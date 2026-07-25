"""Train a configured anticipation model on one feature modality."""

import argparse
import tomllib
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from action_anticipation.data import AnticipationDataset
from action_anticipation.engine import evaluate, start_wandb_run, train_epoch
from action_anticipation.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text())
    data_config = config["data"]
    train_set = AnticipationDataset(args.train_csv, args.feature_store, **data_config)
    val_set = AnticipationDataset(args.val_csv, args.feature_store, **data_config)
    training = config["training"]
    loader_config = training["dataloader"]
    loader_args = {
        "batch_size": loader_config["batch_size"],
        "num_workers": loader_config["num_workers"],
        "pin_memory": loader_config["pin_memory"],
    }
    train_loader = DataLoader(train_set, shuffle=loader_config["train_shuffle"], **loader_args)
    val_loader = DataLoader(val_set, shuffle=loader_config["validation_shuffle"], **loader_args)
    model = build_model(config["model"]["name"], **config["model"]["kwargs"])
    device_name = training["device"]
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    model.to(device)

    # Use DataParallel to leverage all available GPUs on platforms like Kaggle
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = torch.nn.DataParallel(model)

    optimizer_config = training["optimizer"]
    optimizers = {"adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
    optimizer = optimizers[optimizer_config["name"]](model.parameters(), **optimizer_config["kwargs"])
    if training["loss"]["name"] != "cross_entropy":
        raise ValueError("Only cross_entropy loss is currently supported")
    loss_fn = torch.nn.CrossEntropyLoss(**training["loss"]["kwargs"])
    args.output.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf")
    run = start_wandb_run(config)
    try:
        for epoch in range(1, training["epochs"] + 1):
            train_loss = train_epoch(model, train_loader, optimizer, device, loss_fn, training["supervise_all_horizons"])
            metrics = evaluate(model, val_loader, device, loss_fn, training["supervise_all_horizons"], training["metric_horizon"])
            epoch_metrics = {"epoch": epoch, "train/loss": train_loss, **{f"validation/{name}": value for name, value in metrics.items()}}
            print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={metrics['loss']:.4f} val_top1={metrics['top1']:.4f}")
            if run is not None:
                run.log(epoch_metrics)
            if metrics[training["checkpoint_metric"]] > best_score:
                best_score = metrics[training["checkpoint_metric"]]
                model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
                torch.save({"model": model_to_save.state_dict(), "config": config, "epoch": epoch, "metrics": metrics}, args.output / "best.pt")
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
