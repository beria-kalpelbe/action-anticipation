"""Convert per-frame detector outputs into fixed-size object-score features."""

import argparse
import tomllib
from pathlib import Path

import lmdb
import numpy as np


def aggregate_detections(detections: np.ndarray, num_classes: int) -> np.ndarray:
    """Sum detection confidence for each class; expected rows are [class, ..., score]."""
    feature = np.zeros(num_classes, dtype=np.float32)
    for detection in detections:
        class_id = int(detection[0])
        if 0 <= class_id < num_classes:
            feature[class_id] += float(detection[-1])
    return feature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())["object_aggregation"]
    detections = np.load(config["detections"], allow_pickle=True)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    environment = lmdb.open(str(output), map_size=config["lmdb_map_size"])
    with environment.begin(write=True) as transaction:
        for offset, frame_detections in enumerate(detections):
            frame = config["start_frame"] + offset
            key = f"{config['video_id']}_frame_{frame:010d}.jpg"
            transaction.put(key.encode("utf-8"), aggregate_detections(frame_detections, config["num_classes"]).tobytes())
    environment.close()


if __name__ == "__main__":
    main()
