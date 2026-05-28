# STRATS Benchmark for MIMIC-IV Shaped Thesis Data

This copy is trimmed to the supervised STRATS comparison path used in this
workspace. The original auxiliary baselines, pretraining scripts, and
PhysioNet/MIMIC-III preprocessors were removed to keep the benchmark focused.

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

## Environment

Use the workspace venv:

```powershell
& .\.venv\Scripts\python.exe -m pip install pandas tqdm scikit-learn transformers==4.35.2 torch
```

## Preprocess

From the repository root:

```powershell
& .\.venv\Scripts\python.exe src\preprocess_user_mimic_iv.py
```

This creates `data/processed/user_mimic_iv.pkl` from:

- `data/temporal_data.csv`
- `data/context_data.csv`

The default comparable outcomes are:

- `DEATH`
- `DISGLYCEMIA_Hyperglycemia` from glucose measurements `>= 180`
- `DISGLYCEMIA_Hypoglycemia` from glucose measurements `<= 70`
- `KIDNEY_COMPLICATION`
- `CARDIO-VASCULAR_DISORDER`
- `NERVOUS_SYSTEM_DISORDER`
- `NEUROVASCULAR_COMPLICATION`
- `SKIN_ULCER`
- `RETINOPATHY`
- `KETOACIDOSIS`

Preprocessing defaults to `--label_mode future`: first `--input_days` days are
used as STRATS input, and labels are positive when the outcome occurs in the
following `--horizon_days` days. In the current CSV, the non-glucose
complication concepts are timestamped at admission, so they have zero positives
under this future-horizon definition. `DEATH` and glucose-threshold outcomes do
have future positives.

## Train

```powershell
& .\.venv\Scripts\python.exe src\main.py --dataset user_mimic_iv --model_type strats --run 1o1 --train_frac 1.0 --device cpu
```

Outputs are written under `outputs/user_mimic_iv/...`, including:

- `test_per_outcome_metrics.csv`
- `test_predictions.csv`
- `test_risk_df.csv`
- `test_peak_mae_hours.csv`
