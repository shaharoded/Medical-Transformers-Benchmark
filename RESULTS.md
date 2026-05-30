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

| Model | Average | AUROC | AUPRC | F1@0.5 | Best F1 | minRP |
|---|---|---:|---:|---:|---:|---:|
| STRATS | Macro | 0.903010 | 0.602967 | 0.547001 | 0.601667 | 0.589573 |
| STRATS | Weighted | 0.899522 | 0.719333 | 0.649222 | 0.678485 | 0.671475 |
| GRU-D | Macro | 0.898261 | 0.586806 | 0.557213 | 0.585229 | 0.575922 |
| GRU-D | Weighted | 0.899019 | 0.714643 | 0.655029 | 0.673536 | 0.668379 |

The primary comparison metrics for these discriminative baselines are
AUROC/AUPRC/F1/minRP. These baselines do not generate trajectories or true onset
times, so onset-time MAE is intentionally not reported. `F1@0.5` uses a fixed
probability threshold of 0.5. `Best F1` is the maximum F1 over the
precision-recall threshold sweep on the test predictions. Macro is an
unweighted mean across outcomes. Weighted uses each outcome's positive support
(`n_pos`) as the weight. Values above are means over paired seeds `2023-2025`.

## Per-Outcome Test Metrics

### STRATS

| Outcome | AUROC | AUPRC | F1@0.5 | Best F1 | minRP | Positives | Negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| `DEATH` | 0.895034 | 0.533791 | 0.506082 | 0.538716 | 0.525994 | 1,101 | 10,315 |
| `DISGLYCEMIA_Hyperglycemia` | 0.898643 | 0.776720 | 0.706631 | 0.712981 | 0.708565 | 2,946 | 8,470 |
| `DISGLYCEMIA_Hypoglycemia` | 0.850503 | 0.297016 | 0.341119 | 0.378081 | 0.348828 | 675 | 10,741 |
| `KIDNEY_COMPLICATION` | 0.966257 | 0.869701 | 0.741919 | 0.826582 | 0.822474 | 1,672 | 9,744 |
| `CARDIO-VASCULAR_DISORDER` | 0.931745 | 0.377600 | 0.301284 | 0.458233 | 0.439746 | 220 | 11,196 |
| `HYPEROSMOLALITY` | 0.875878 | 0.762972 | 0.684968 | 0.695411 | 0.691834 | 3,301 | 8,115 |

### GRU-D

| Outcome | AUROC | AUPRC | F1@0.5 | Best F1 | minRP | Positives | Negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| `DEATH` | 0.900886 | 0.525798 | 0.498121 | 0.533675 | 0.528127 | 1,101 | 10,315 |
| `DISGLYCEMIA_Hyperglycemia` | 0.894190 | 0.762703 | 0.699384 | 0.707077 | 0.702828 | 2,946 | 8,470 |
| `DISGLYCEMIA_Hypoglycemia` | 0.819213 | 0.243249 | 0.302996 | 0.321625 | 0.298822 | 675 | 10,741 |
| `KIDNEY_COMPLICATION` | 0.966798 | 0.875662 | 0.778027 | 0.820935 | 0.819015 | 1,672 | 9,744 |
| `CARDIO-VASCULAR_DISORDER` | 0.925552 | 0.338806 | 0.368166 | 0.423876 | 0.405538 | 220 | 11,196 |
| `HYPEROSMOLALITY` | 0.882924 | 0.774620 | 0.696583 | 0.704189 | 0.701202 | 3,301 | 8,115 |

## Result Files

Each model directory contains:

- `checkpoint_best.bin`: best validation checkpoint.
- `log.txt`: training and final evaluation log.
- `test_per_outcome_metrics.csv`: AUROC/AUPRC/F1/minRP/support by outcome.
- `test_predictions.csv`: patient-level labels and predicted probabilities.
- `test_risk_df.csv`: repeated daily risk rows for evaluator compatibility.

No MAE file is produced. STRATS and GRU-D are reported as fixed-horizon
discriminative baselines; missing onset-time MAE should be reported as an
unsupported output rather than approximated.

## Commands Used

Preprocess:

```bash
.venv/bin/python scripts/preprocess_mimic_iv.py
```

STRATS:

```bash
.venv/bin/python main.py --dataset user_mimic_iv --model_type strats --run 1o1 --train_frac 1.0 --device cuda --output_dir outputs/user_mimic_iv/strats
```

GRU-D:

```bash
.venv/bin/python main.py --dataset user_mimic_iv --model_type grud --run 1o1 --train_frac 1.0 --device cuda --output_dir outputs/user_mimic_iv/grud
```
