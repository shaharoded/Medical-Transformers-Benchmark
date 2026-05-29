# Medical-Transformers Benchmark Results

This report documents the full-data RunPod benchmark run for the MIMIC-IV-shaped
clinical trajectory task.

Task:

```text
first 48 hours of temporal data + static context -> outcome risk over hours 48-336
```

Models:

- `strats`: sparse irregular time-series transformer.
- `grud`: GRU-D recurrent baseline with values, masks, and time-delta tensors.

Outputs were written on the RunPod machine under:

```text
/workspace/benchmark/outputs/user_mimic_iv/strats
/workspace/benchmark/outputs/user_mimic_iv/grud
```

## Dataset Snapshot

Preprocessing output:

| Item | Value |
|---|---:|
| Processed file | `data/processed/user_mimic_iv.pkl` |
| Temporal rows | 10,586,188 |
| Patients | 57,078 |
| Input window | first 48 hours |
| Prediction horizon | hours 48-336 |

Configured prediction targets:

- `DEATH`
- `DISGLYCEMIA_Hyperglycemia`
- `DISGLYCEMIA_Hypoglycemia`
- `KIDNEY_COMPLICATION`
- `CARDIO-VASCULAR_DISORDER`
- `HYPEROSMOLALITY`

Dropped low-support configured outcomes: none.

Dropped low-support input concepts:

- `ANTIDIABETIC_HIGH_HYPO_HOSPITAL_BITZUA`
- `METFORMIN_HOSPITAL_BITZUA`

## Overall Test Results

| Model | AUROC | AUPRC | F1@0.5 | Best F1 | minRP | MAE hours |
|---|---:|---:|---:|---:|---:|---:|
| STRATS | 0.902644 | 0.602773 | 0.526159 | 0.601032 | 0.589140 | 43.469718 |
| GRU-D | 0.897490 | 0.585537 | 0.558765 | 0.585612 | 0.577346 | 43.469718 |

The primary comparison metrics for these discriminative baselines are
AUROC/AUPRC/minRP. The MAE file is retained for compatibility with the thesis
evaluation shape, but these baselines do not generate trajectories or true onset
times. `F1@0.5` uses a fixed probability threshold of 0.5. `Best F1` is the
maximum F1 over the precision-recall threshold sweep on the test predictions.

## Per-Outcome Test Metrics

### STRATS

| Outcome | AUROC | AUPRC | F1@0.5 | Best F1 | minRP | Positives | Negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| `DEATH` | 0.894743 | 0.534561 | 0.467157 | 0.544124 | 0.527652 | 1,101 | 10,315 |
| `DISGLYCEMIA_Hyperglycemia` | 0.898491 | 0.776084 | 0.698201 | 0.712684 | 0.704785 | 2,946 | 8,470 |
| `DISGLYCEMIA_Hypoglycemia` | 0.846296 | 0.295564 | 0.311140 | 0.371171 | 0.347769 | 675 | 10,741 |
| `KIDNEY_COMPLICATION` | 0.965344 | 0.870695 | 0.698291 | 0.824324 | 0.820574 | 1,672 | 9,744 |
| `CARDIO-VASCULAR_DISORDER` | 0.936205 | 0.375760 | 0.299732 | 0.460967 | 0.441964 | 220 | 11,196 |
| `HYPEROSMOLALITY` | 0.874785 | 0.763974 | 0.682433 | 0.692923 | 0.692098 | 3,301 | 8,115 |

### GRU-D

| Outcome | AUROC | AUPRC | F1@0.5 | Best F1 | minRP | Positives | Negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| `DEATH` | 0.903303 | 0.534434 | 0.506457 | 0.541157 | 0.529519 | 1,101 | 10,315 |
| `DISGLYCEMIA_Hyperglycemia` | 0.891190 | 0.757936 | 0.694669 | 0.700592 | 0.694162 | 2,946 | 8,470 |
| `DISGLYCEMIA_Hypoglycemia` | 0.818328 | 0.240356 | 0.312931 | 0.324713 | 0.305355 | 675 | 10,741 |
| `KIDNEY_COMPLICATION` | 0.966186 | 0.872039 | 0.776790 | 0.819219 | 0.818888 | 1,672 | 9,744 |
| `CARDIO-VASCULAR_DISORDER` | 0.922761 | 0.333856 | 0.365123 | 0.423841 | 0.418182 | 220 | 11,196 |
| `HYPEROSMOLALITY` | 0.883169 | 0.774603 | 0.696620 | 0.704148 | 0.697970 | 3,301 | 8,115 |

## Per-Outcome MAE Proxy

The same per-outcome MAE proxy was exported for both models:

| Outcome | MAE hours | Positive patients |
|---|---:|---:|
| `DEATH` | 102.142628 | 1,101 |
| `DISGLYCEMIA_Hyperglycemia` | 26.759125 | 2,946 |
| `DISGLYCEMIA_Hypoglycemia` | 46.826370 | 675 |
| `KIDNEY_COMPLICATION` | 23.761154 | 1,672 |
| `CARDIO-VASCULAR_DISORDER` | 34.942273 | 220 |
| `HYPEROSMOLALITY` | 26.386757 | 3,301 |

Because STRATS and GRU-D emit one horizon-risk score per outcome, this MAE is a
compatibility proxy. It should not be interpreted as evidence that the models
learned the same event-time trajectory.

## Result Files

Each model directory contains:

- `checkpoint_best.bin`: best validation checkpoint.
- `log.txt`: training and final evaluation log.
- `test_per_outcome_metrics.csv`: AUROC/AUPRC/F1/minRP/support by outcome.
- `test_predictions.csv`: patient-level labels and predicted probabilities.
- `test_risk_df.csv`: repeated daily risk rows for evaluator compatibility.
- `test_peak_mae_hours.csv`: simplified peak-risk timing proxy.

## Commands Used

Preprocess:

```bash
.venv/bin/python scripts/preprocess_user_mimic_iv.py
```

STRATS:

```bash
.venv/bin/python main.py --dataset user_mimic_iv --model_type strats --run 1o1 --train_frac 1.0 --device cuda --output_dir outputs/user_mimic_iv/strats
```

GRU-D:

```bash
.venv/bin/python main.py --dataset user_mimic_iv --model_type grud --run 1o1 --train_frac 1.0 --device cuda --output_dir outputs/user_mimic_iv/grud
```
