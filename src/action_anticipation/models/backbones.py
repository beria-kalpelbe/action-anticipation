"""Pretrained image encoders used by feature-extraction jobs.

Keep vision-model construction here, separate from dataset/LMDB concerns.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ImageBackbone:
    model: nn.Module
    transform: object
    feature_dim: int


def build_image_backbone(name: str, pretrained: bool) -> ImageBackbone:
    """Build a torchvision ResNet encoder with its matching preprocessing."""
    try:
        from torchvision.models import get_model, get_model_weights
    except ImportError as error:
        raise ImportError("Install the vision extra: uv sync --extra vision") from error
    if name not in {"resnet18", "resnet50", "resnet101"}:
        raise ValueError("Supported image backbones: resnet18, resnet50, resnet101")
    weights = get_model_weights(name).DEFAULT if pretrained else None
    model = get_model(name, weights=weights)
    if not isinstance(model.fc, nn.Linear):
        raise TypeError(f"Expected a ResNet classifier for {name}")
    feature_dim = model.fc.in_features
    model.fc = nn.Identity()
    transform = weights.transforms() if weights is not None else _default_transform()
    return ImageBackbone(model=model, transform=transform, feature_dim=feature_dim)


def _default_transform() -> object:
    from torchvision.transforms import v2

    return v2.Compose([
        v2.Resize(256), v2.CenterCrop(224), v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
    ])
