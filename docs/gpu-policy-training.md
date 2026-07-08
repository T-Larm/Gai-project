# GPU Policy Training

Use this on the school A40 machine. The training script defaults to CUDA and
refuses CPU unless `--allow-cpu` is explicitly passed for local debugging.

## Check Environment

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected: CUDA is `True`, device name contains `A40`.

## Train

```powershell
python -m evaluation.train_policy `
  --data-dir data\behavior_policy\stateful_rpg_v2 `
  --out-dir data\behavior_policy\checkpoints\stateful_rpg_v2_a40 `
  --epochs 30 `
  --batch-size 1024 `
  --hidden-dim 256 `
  --device cuda `
  --max-vocab-size 512
```

Outputs:

- `model.pt`
- `metadata.json`
- `metrics.json`
- `confusion_matrices.json`

The model trains the native `action_id` head (11 effective simulator actions —
the behavior-cloning target recomputed with the generator's deterministic
rule) plus an auxiliary `mood` head. Metrics report accuracy and macro-F1 per
head; macro-F1 matters because `heal`/`pray` are rare. Pass `--no-mood-head`
to train the action head alone.
