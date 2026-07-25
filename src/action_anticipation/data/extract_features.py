"""Extract pretrained RGB features from frames and store them in LMDB."""

import argparse
import tomllib
from pathlib import Path

import lmdb
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from action_anticipation.models.backbones import build_image_backbone


def _image_paths(directory: Path, pattern: str) -> list[Path]:
    paths = sorted(path for path in directory.rglob(pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} under {directory}")
    return paths


def _key(path: Path, root: Path, key_mode: str) -> str:
    if key_mode == "basename":
        return path.name
    return path.relative_to(root).as_posix()


def extract_features(
    input_dir: Path,
    output_store: Path,
    backbone_name: str,
    batch_size: int,
    pattern: str,
    key_mode: str,
    pretrained: bool,
    lmdb_map_size: int,
) -> int:
    """Write one float32 feature vector per image and return the feature size."""
    backbone = build_image_backbone(backbone_name, pretrained=pretrained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = backbone.model.to(device).eval()
    paths = _image_paths(input_dir, pattern)
    output_store.mkdir(parents=True, exist_ok=True)
    environment = lmdb.open(str(output_store), map_size=lmdb_map_size)
    with torch.inference_mode(), environment.begin(write=True) as transaction:
        for start in tqdm(range(0, len(paths), batch_size), desc="Extracting features"):
            group = paths[start : start + batch_size]
            images = [Image.open(path).convert("RGB") for path in group]
            batch: Tensor = torch.stack([backbone.transform(image) for image in images]).to(device)
            features = model(batch).float().cpu().numpy()
            for path, feature in zip(group, features, strict=True):
                transaction.put(_key(path, input_dir, key_mode).encode("utf-8"), np.asarray(feature, dtype=np.float32).tobytes())
    environment.close()
    return backbone.feature_dim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())["extraction"]
    input_dir, output_store = Path(config["input"]), Path(config["output"])
    feature_dim = extract_features(input_dir, output_store, config["backbone"], config["batch_size"],
                                   config["pattern"], config["key_mode"], config["pretrained"], config["lmdb_map_size"])
    print(f"Wrote {feature_dim}-D {config['backbone']} features to {output_store}")


if __name__ == "__main__":
    main()
