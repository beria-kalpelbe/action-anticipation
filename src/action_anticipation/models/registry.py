"""Central model registry used by experiment configurations."""

from collections.abc import Callable
from typing import Any

from torch import nn

from action_anticipation.models.rulstm import RULSTMAnticipator
from action_anticipation.models.transformer import TemporalTransformerAnticipator

ModelFactory = Callable[..., nn.Module]

_MODELS: dict[str, ModelFactory] = {
    "rulstm": RULSTMAnticipator,
    "temporal_transformer": TemporalTransformerAnticipator,
}


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_MODELS))


def build_model(name: str, **kwargs: Any) -> nn.Module:
    """Build a registered model from a config name and keyword arguments."""
    try:
        factory = _MODELS[name]
    except KeyError as error:
        choices = ", ".join(available_models())
        raise ValueError(f"Unknown model {name!r}. Available models: {choices}") from error
    return factory(**kwargs)
