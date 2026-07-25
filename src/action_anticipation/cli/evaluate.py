"""Evaluate a saved model checkpoint."""

import argparse
import tomllib
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from action_anticipation.data import AnticipationDataset
from action_anticipation.engine import evaluate
from action_anticipation.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--feature-store", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    dataset = AnticipationDataset(args.csv, args.feature_store, **config["data"])
    training = config["training"]
    loader_config = training["dataloader"]
    loader = DataLoader(dataset, batch_size=loader_config["batch_size"],
                        shuffle=loader_config["validation_shuffle"],
                        num_workers=loader_config["num_workers"], pin_memory=loader_config["pin_memory"])
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
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if training["loss"]["name"] != "cross_entropy":
        raise ValueError("Only cross_entropy loss is currently supported")
    loss_fn = torch.nn.CrossEntropyLoss(**training["loss"]["kwargs"])
    print(evaluate(model.to(device), loader, device, loss_fn, training["supervise_all_horizons"], training["metric_horizon"]))


if __name__ == "__main__":
    main()
