import pytest
import torch

from action_anticipation.models import available_models, build_model


@pytest.mark.parametrize("name,kwargs", [
    ("rulstm", {"feature_dim": 16, "num_classes": 5, "hidden_dim": 8, "num_layers": 1, "dropout": 0.0}),
    ("temporal_transformer", {"feature_dim": 16, "num_classes": 5, "hidden_dim": 8, "num_heads": 2, "num_layers": 1, "feedforward_dim": 32, "dropout": 0.0}),
])
def test_registered_models_return_logits_per_timestep(name, kwargs):
    model = build_model(name, **kwargs)
    logits = model(torch.randn(3, 4, 16))
    assert logits.shape == (3, 4, 5)


def test_model_names_are_discoverable():
    assert set(available_models()) == {"rulstm", "temporal_transformer"}
