# Medical-Transformers Benchmark

Supervised horizon-risk benchmarks for MIMIC-IV-shaped clinical trajectory data.
The repository currently supports two discriminative baselines:

- `strats`: sparse irregular time-series transformer.
- `grud`: GRU-D recurrent baseline for irregular time series with missingness.

Both models answer the same task:

```text
first 48 hours of temporal data + static context -> outcome risk over hours 48-336
```

They do not generate trajectories. AUROC/AUPRC are the primary comparable
metrics. The exported MAE file is a simplified peak-risk timing proxy, not a
true autoregressive onset-time metric.

## Authors and Credit

Benchmark adaptation, data interface, multi-label outcome evaluation, and README:
Shahar Oded (`@shaharoded`).

This repository is adapted from the official PyTorch reimplementation of
STraTS by Sindhu Tipirneni and Chandan K. Reddy:
https://github.com/sindhura97/STraTS

Original paper:

```bibtex
@article{tipirneni2022self,
  title={Self-supervised transformer for sparse and irregularly sampled multivariate clinical time-series},
  author={Tipirneni, Sindhu and Reddy, Chandan K},
  journal={ACM Transactions on Knowledge Discovery from Data (TKDD)},
  volume={16},
  number={6},
  pages={1--17},
  year={2022},
  publisher={ACM New York, NY}
}
```

## Setup

Run all commands from the repository root.

Create/use a venv and install dependencies:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install pandas tqdm scikit-learn transformers==4.35.2 torch
```

Expected input files:

```text
data/mimic-iv-input-data.csv
data/context_data.csv
```

## Preprocess

Build the processed benchmark pickle:

```powershell
& .\.venv\Scripts\python.exe src\preprocess_user_mimic_iv.py
```

This writes:

```text
data/processed/user_mimic_iv.pkl
```

The configured targets live in `src/config.py`:

- `DEATH`
- `DISGLYCEMIA_Hyperglycemia`
- `DISGLYCEMIA_Hypoglycemia`
- `KIDNEY_COMPLICATION`
- `CARDIO-VASCULAR_DISORDER`
- `HYPEROSMOLALITY`

Dysglycemia outcomes are synthesized during preprocessing:

- Hypoglycemia: one glucose `<= 54`, or the second and later glucose values `<= 70`.
- Hyperglycemia: one glucose `>= 250`, or the second and later glucose values `>= 180`.

Low-support input concepts and configured outcomes are filtered using
`CONCEPT_SUPPORT_THRESHOLD` and `OUTCOME_SUPPORT_THRESHOLD` in `src/config.py`.

## Train STRATS

On a GPU machine:

```powershell
& .\.venv\Scripts\python.exe src\main.py --dataset user_mimic_iv --model_type strats --run 1o1 --train_frac 1.0 --device cuda --output_dir outputs/user_mimic_iv/strats
```

If CUDA is available, `--device cuda` can be omitted because the script auto-selects
GPU. Keep it explicit when launching jobs so failures are obvious.

## Train GRU-D

On a GPU machine:

```powershell
& .\.venv\Scripts\python.exe src\main.py --dataset user_mimic_iv --model_type grud --run 1o1 --train_frac 1.0 --device cuda --output_dir outputs/user_mimic_iv/grud
```

GRU-D uses the same labels, splits, static context, and evaluator as STRATS, but
converts sparse events into value/mask/delta tensors internally.

## Results

Each model output directory contains:

- `checkpoint_best.bin`: best validation checkpoint.
- `log.txt`: training and evaluation log.
- `test_per_outcome_metrics.csv`: AUROC/AUPRC/minRP/support by outcome.
- `test_predictions.csv`: patient-level labels and predicted probabilities.
- `test_risk_df.csv`: daily repeated risk rows for compatibility with the evaluation shape.
- `test_peak_mae_hours.csv`: simplified peak-risk timing proxy.

Recommended comparison files:

```text
outputs/user_mimic_iv/strats/test_per_outcome_metrics.csv
outputs/user_mimic_iv/grud/test_per_outcome_metrics.csv
```

## Smoke Tests

These small CPU commands verify that preprocessing, model construction, batching,
loss computation, and the training loop work. They are not meaningful model runs.
They intentionally skip validation, so use the full training commands above to
produce result CSVs.

STRATS smoke test:

```powershell
& .\.venv\Scripts\python.exe src\main.py --dataset user_mimic_iv --model_type strats --run 1o1 --train_frac 0.01 --device cpu --max_epochs 1 --train_batch_size 64 --eval_batch_size 128 --validate_after 999999 --max_obs 64 --hid_dim 16 --num_layers 1 --num_heads 2 --output_dir outputs/smoke/strats
```

GRU-D smoke test:

```powershell
& .\.venv\Scripts\python.exe src\main.py --dataset user_mimic_iv --model_type grud --run 1o1 --train_frac 0.01 --device cpu --max_epochs 1 --train_batch_size 64 --eval_batch_size 128 --validate_after 999999 --max_timesteps 64 --hid_dim 16 --output_dir outputs/smoke/grud
```
