"""Dataset for temporal features stored in one LMDB per modality."""

import bisect
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset


# Module-level cache for opened LMDB environments to prevent "already open in this process" errors
_environments: dict[str, lmdb.Environment] = {}
# Module-level cache for frame lists per video in each feature store environment
_video_to_frames: dict[str, dict[str, list[int]]] = {}


class AnticipationDataset(Dataset[dict[str, Tensor | int]]):
    """Read a fixed history of features before each annotated action.

    The annotation CSV uses the legacy RU-LSTM columns: video, start, end,
    verb, noun, action (without a header). Feature keys are
    ``{video}_frame_{frame:010d}.jpg`` by default.
    """

    def __init__(
        self,
        annotations: str | Path,
        feature_store: str | Path,
        sequence_length: int,
        step_seconds: float,
        fps: int,
        label: str,
        frame_template: str,
        lmdb_readahead: bool,
        subset_size: int | str | None = None,
    ) -> None:
        self.annotations = pd.read_csv(
            annotations, header=None,
            names=["video", "start", "end", "verb", "noun", "action"],
            skipinitialspace=True,
        )
        if label not in {"verb", "noun", "action"}:
            raise ValueError("label must be one of: verb, noun, action")
        self.label = label
        self.sequence_length = sequence_length
        self.step_frames = round(step_seconds * fps)
        self.frame_template = frame_template
        
        feature_store_path = str(Path(feature_store).resolve())
        if feature_store_path not in _environments:
            _environments[feature_store_path] = lmdb.open(
                feature_store_path, readonly=True, lock=False, readahead=lmdb_readahead
            )
        self.environment = _environments[feature_store_path]
        self.feature_store_path = feature_store_path

        # Populate video frame caches for any unique video in annotations if not already cached
        if feature_store_path not in _video_to_frames:
            _video_to_frames[feature_store_path] = {}
        
        unique_videos = self.annotations["video"].unique()
        uncached_videos = [v for v in unique_videos if v not in _video_to_frames[feature_store_path]]
        if uncached_videos:
            with self.environment.begin() as txn:
                cursor = txn.cursor()
                for video in uncached_videos:
                    frames = []
                    start_key = f"{video}_frame_0000000000.jpg".encode("utf-8")
                    if cursor.set_range(start_key):
                        prefix = f"{video}_frame_".encode("utf-8")
                        for key, _ in cursor:
                            if not key.startswith(prefix):
                                break
                            k = key.decode("utf-8")
                            frame_part = k.split("_frame_")[1].split(".jpg")[0]
                            frames.append(int(frame_part))
                    if frames:
                        _video_to_frames[feature_store_path][video] = sorted(frames)

        # Filter annotations to handle subset feature stores (where some videos may not be present in LMDB)
        available_videos = set(_video_to_frames[feature_store_path].keys())
        original_len = len(self.annotations)
        self.annotations = self.annotations[self.annotations["video"].isin(available_videos)].reset_index(drop=True)
        if len(self.annotations) < original_len:
            print(f"[{Path(feature_store).name}] Filtered annotations based on available videos in LMDB: "
                  f"{original_len} -> {len(self.annotations)}")

        # Select a fixed part of the dataset if subset_size is specified
        if subset_size is not None and subset_size != "" and subset_size != 0:
            subset_size = int(subset_size)
            self.annotations = self.annotations.iloc[:subset_size].reset_index(drop=True)
            print(f"[{Path(feature_store).name}] Selected a fixed subset of the dataset: first {subset_size} samples.")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> dict[str, Tensor | int]:
        row = self.annotations.iloc[index]
        frames = np.arange(self.sequence_length, 0, -1) * self.step_frames
        frames = np.maximum(int(row.start) - frames, 1)

        # Map each target frame to the closest available frame in the video
        video_frames = _video_to_frames[self.feature_store_path].get(row.video, [])
        if not video_frames:
            raise KeyError(f"No frames available in LMDB for video {row.video}")

        mapped_frames = []
        for frame in frames:
            idx = bisect.bisect_left(video_frames, frame)
            if idx == 0:
                closest = video_frames[0]
            elif idx == len(video_frames):
                closest = video_frames[-1]
            else:
                before = video_frames[idx - 1]
                after = video_frames[idx]
                closest = after if (after - frame < frame - before) else before
            mapped_frames.append(closest)

        keys = [f"{row.video}_{self.frame_template.format(int(frame))}" for frame in mapped_frames]
        with self.environment.begin() as transaction:
            values = [transaction.get(key.encode("utf-8")) for key in keys]
        missing = [key for key, value in zip(keys, values, strict=True) if value is None]
        if missing:
            raise KeyError(f"Missing {len(missing)} features for annotation {index}: {missing[0]}")
        features = np.stack([np.frombuffer(value, dtype=np.float32) for value in values])
        return {"features": Tensor(features), "target": int(row[self.label]), "id": index}
