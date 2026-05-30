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
| STRATS | Macro | 0.9030 +/- 0.0020 | 0.6030 +/- 0.0015 | 0.5470 +/- 0.0209 | 0.6017 +/- 0.0065 | 0.5896 +/- 0.0058 |
| STRATS | Weighted | 0.8995 +/- 0.0013 | 0.7193 +/- 0.0007 | 0.6492 +/- 0.0173 | 0.6785 +/- 0.0019 | 0.6715 +/- 0.0015 |
| GRU-D | Macro | 0.8983 +/- 0.0010 | 0.5868 +/- 0.0020 | 0.5572 +/- 0.0015 | 0.5852 +/- 0.0046 | 0.5759 +/- 0.0029 |
| GRU-D | Weighted | 0.8990 +/- 0.0008 | 0.7146 +/- 0.0014 | 0.6550 +/- 0.0002 | 0.6735 +/- 0.0030 | 0.6684 +/- 0.0030 |

The primary comparison metrics for these discriminative baselines are
AUROC/AUPRC/F1/minRP. These baselines do not generate trajectories or true onset
times, so onset-time MAE is intentionally not reported. `F1@0.5` uses a fixed
probability threshold of 0.5. `Best F1` is the maximum F1 over the
precision-recall threshold sweep on the test predictions. Macro is an
unweighted mean across outcomes. Weighted uses each outcome's positive support
(`n_pos`) as the weight. Values above are means +/- 95% CI over paired seeds
`2023-2025`.

## Per-Outcome Test Metrics

### STRATS

| Outcome | AUROC | AUPRC | F1@0.5 | Best F1 | minRP | Positives | Negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| `DEATH` | 0.8950 +/- 0.0025 | 0.5338 +/- 0.0063 | 0.5061 +/- 0.0390 | 0.5387 +/- 0.0054 | 0.5260 +/- 0.0046 | 1,101 | 10,315 |
| `DISGLYCEMIA_Hyperglycemia` | 0.8986 +/- 0.0004 | 0.7767 +/- 0.0007 | 0.7066 +/- 0.0094 | 0.7130 +/- 0.0026 | 0.7086 +/- 0.0056 | 2,946 | 8,470 |
| `DISGLYCEMIA_Hypoglycemia` | 0.8505 +/- 0.0108 | 0.2970 +/- 0.0084 | 0.3411 +/- 0.0308 | 0.3781 +/- 0.0265 | 0.3488 +/- 0.0215 | 675 | 10,741 |
| `KIDNEY_COMPLICATION` | 0.9663 +/- 0.0011 | 0.8697 +/- 0.0050 | 0.7419 +/- 0.0428 | 0.8266 +/- 0.0036 | 0.8225 +/- 0.0021 | 1,672 | 9,744 |
| `CARDIO-VASCULAR_DISORDER` | 0.9317 +/- 0.0046 | 0.3776 +/- 0.0068 | 0.3013 +/- 0.0406 | 0.4582 +/- 0.0128 | 0.4397 +/- 0.0130 | 220 | 11,196 |
| `HYPEROSMOLALITY` | 0.8759 +/- 0.0011 | 0.7630 +/- 0.0036 | 0.6850 +/- 0.0108 | 0.6954 +/- 0.0026 | 0.6918 +/- 0.0005 | 3,301 | 8,115 |

### GRU-D

| Outcome | AUROC | AUPRC | F1@0.5 | Best F1 | minRP | Positives | Negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| `DEATH` | 0.9009 +/- 0.0026 | 0.5258 +/- 0.0085 | 0.4981 +/- 0.0082 | 0.5337 +/- 0.0087 | 0.5281 +/- 0.0050 | 1,101 | 10,315 |
| `DISGLYCEMIA_Hyperglycemia` | 0.8942 +/- 0.0030 | 0.7627 +/- 0.0051 | 0.6994 +/- 0.0052 | 0.7071 +/- 0.0076 | 0.7028 +/- 0.0094 | 2,946 | 8,470 |
| `DISGLYCEMIA_Hypoglycemia` | 0.8192 +/- 0.0041 | 0.2432 +/- 0.0075 | 0.3030 +/- 0.0109 | 0.3216 +/- 0.0038 | 0.2988 +/- 0.0156 | 675 | 10,741 |
| `KIDNEY_COMPLICATION` | 0.9668 +/- 0.0006 | 0.8757 +/- 0.0045 | 0.7780 +/- 0.0029 | 0.8209 +/- 0.0037 | 0.8190 +/- 0.0030 | 1,672 | 9,744 |
| `CARDIO-VASCULAR_DISORDER` | 0.9256 +/- 0.0027 | 0.3388 +/- 0.0178 | 0.3682 +/- 0.0039 | 0.4239 +/- 0.0127 | 0.4055 +/- 0.0188 | 220 | 11,196 |
| `HYPEROSMOLALITY` | 0.8829 +/- 0.0006 | 0.7746 +/- 0.0002 | 0.6966 +/- 0.0004 | 0.7042 +/- 0.0001 | 0.7012 +/- 0.0033 | 3,301 | 8,115 |

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
