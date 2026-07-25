# Action anticipation research framework

This is a Python 3.12, config-first framework for action-anticipation research.
The source package contains reusable datasets, models, training/evaluation code,
and feature extraction. Local annotations and LMDB stores belong in `data/`;
experiment configurations belong in `configs/`.

```
src/action_anticipation/   framework code
  data/                    LMDB dataset and feature extraction
  models/                  anticipation models and pretrained backbones
  engine/                  training/evaluation loops
configs/                   complete dataset/model/training configurations
tests/                     model contract tests
data/                      annotations and local feature stores
scripts/, notebooks/       archived legacy material
```

Every dataset, model, and training hyperparameter is explicit in the experiment
TOML. Python modules do not supply fallback hyperparameters. Start from
`configs/rulstm.toml` or `configs/temporal_transformer.toml` and version a new
config whenever you change an experiment.

`scripts/train.sh` is a thin launcher for the same config-driven command. It
does not accept hyperparameters; see `scripts/README.md` for its arguments and
the hardened data-download helpers.

```bash
uv sync --extra dev --extra tracking
uv run pytest

uv run aa-train \
  --config configs/temporal_transformer.toml \
  --train-csv data/ek55/training.csv \
  --val-csv data/ek55/validation.csv \
  --feature-store data/ek55/rgb \
  --output artifacts/transformer-rgb
```

For RGB feature extraction, install the vision extra and select a pretrained
backbone:

```bash
uv sync --extra vision
uv run aa-extract-features --config configs/extract_resnet50.toml
```

Copy and edit the extraction config; it contains the input/output paths,
backbone, preprocessing choice, and batch size.

The extractor writes LMDB keys using image basenames by default, matching the
`{video}_frame_{frame:010d}.jpg` convention used by the dataset. The archived
`FEATEXT/` material is only for historical feature/checkpoint reproduction.

### Experiment tracking

Training logs the complete experiment config and epoch metrics to Weights &
Biases when `[tracking.wandb]` is enabled. Set `entity`, `project`, `run_name`,
`tags`, and `mode` in the TOML config. If the section is absent or disabled—or
the optional `wandb` package is not installed—the trainer emits a warning and
continues without remote tracking.
