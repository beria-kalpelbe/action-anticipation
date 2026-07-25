"""Optional experiment tracking with a safe local fallback."""

import warnings
from typing import Any


def start_wandb_run(config: dict[str, Any]):
    """Start W&B from ``[tracking.wandb]`` or warn and return ``None``."""
    try:
        wandb_config = config["tracking"]["wandb"]
    except KeyError:
        warnings.warn(
            "W&B tracking is not configured ([tracking.wandb] is missing); continuing without experiment tracking.",
            stacklevel=2,
        )
        return None
    if not wandb_config["enabled"]:
        warnings.warn("W&B tracking is disabled in the experiment config; continuing without experiment tracking.", stacklevel=2)
        return None
    try:
        import wandb
    except ImportError:
        warnings.warn("W&B tracking is enabled but wandb is not installed; run `uv sync --extra tracking`. Continuing locally.", stacklevel=2)
        return None
    try:
        return wandb.init(
            project=wandb_config["project"],
            entity=wandb_config["entity"] or None,
            name=wandb_config["run_name"] or None,
            tags=wandb_config["tags"],
            mode=wandb_config["mode"],
            config=config,
        )
    except Exception as error:
        warnings.warn(f"W&B initialization failed ({error}); continuing without experiment tracking.", stacklevel=2)
        return None
