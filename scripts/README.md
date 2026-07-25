# Scripts

`train.sh` is the only training launcher. It accepts no model or training
hyperparameters; those belong exclusively in the TOML passed as its first
argument.

```bash
scripts/train.sh \
  configs/rulstm.toml \
  data/ek55/training.csv \
  data/ek55/validation.csv \
  data/ek55/rgb \
  artifacts/rulstm-ek55-rgb
```

The `download_data_*` scripts download archived LMDB features into `data/`.
The historical model-download scripts were removed: the new Python 3.12 model
implementations are not compatible with the archived RU-LSTM checkpoints.
