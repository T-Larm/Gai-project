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
  --data-dir data\behavior_policy\stateful_rpg `
  --out-dir data\behavior_policy\checkpoints\stateful_rpg_a40 `
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

The model trains `dialogue_act`, `emotion`, `disclosure_level`, `gesture`,
`quest_update`, and `source_action_id`. The `source_action_id` head is the most
faithful label from the Stateful RPG dataset; the other heads are the current
dialogue-policy projection used by the project pipeline.
