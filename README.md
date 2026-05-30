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
metrics. Onset-time MAE is intentionally not reported because these baselines do
not generate event-time trajectories.

## Model Context

`strats` is a transformer baseline for sparse and irregularly sampled clinical
time series. Instead of resampling the first 48 hours into a dense grid, each
observation is represented through its value, timestamp, and variable identity.
The transformer then attends over the observed event set, and this benchmark
combines the temporal embedding with static context before a multi-label risk
head.

`grud` is a recurrent baseline designed for multivariate time series where
missingness is informative. The sparse trajectory is converted into values,
observation masks, and per-variable time deltas. GRU-D uses learned decay terms
so stale measurements and hidden states can decay toward defaults. Che et al.
describe GRU-D as using "`masking and time interval`" representations of
missing patterns, which is exactly why it is a relevant clinical benchmark for
irregular ICU data.

Both baselines are discriminative horizon-risk models here. They estimate
whether each configured outcome occurs during hours 48-336; they do not generate
a full future trajectory (decoder style).

## TL;DR Results

Full benchmark results are in [RESULTS.md](RESULTS.md).

RunPod full-data test results, reported as mean over paired seeds `2023-2025`:

| Model | AUROC | AUPRC | F1@0.5 | Best F1 | minRP |
|---|---:|---:|---:|---:|---:|
| STRATS | 0.9030 | 0.6030 | 0.5470 | 0.6017 | 0.5896 |
| GRU-D | 0.8983 | 0.5868 | 0.5572 | 0.5852 | 0.5759 |

STRATS is slightly stronger overall on AUROC/AUPRC/minRP in this run. GRU-D is
competitive and slightly stronger on `DEATH`, `KIDNEY_COMPLICATION`, and
`HYPEROSMOLALITY` AUROC. `F1@0.5` uses a fixed probability threshold of 0.5;
`Best F1` is the best threshold sweep value on the test predictions.

## Authors and Credit

Benchmark adaptation, data interface, multi-label outcome evaluation, and README:
Shahar Oded (`@shaharoded`).

This repository (both STraTS and GRU-D) is adapted from the official PyTorch reimplementation by Sindhu Tipirneni and Chandan K. Reddy:
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

Paper link: https://dl.acm.org/doi/pdf/10.1145/3516367

GRU-D reference:

```bibtex
@article{che2018recurrent,
  title={Recurrent Neural Networks for Multivariate Time Series with Missing Values},
  author={Che, Zhengping and Purushotham, Sanjay and Cho, Kyunghyun and Sontag, David and Liu, Yan},
  journal={Scientific Reports},
  volume={8},
  number={1},
  pages={6085},
  year={2018}
}
```

Paper link: https://www.nature.com/articles/s41598-018-24271-9

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

## Data Shape

Raw temporal input is expected as one row per patient event or measurement:

| Column | Meaning |
|---|---|
| `PatientId` | Admission/patient identifier used as the time-series id. |
| `ConceptName` | Event, measurement, medication, or terminal concept name. |
| `StartDateTime` | Event timestamp. Times are anchored to each patient's `ADMISSION`. |
| `EndDateTime` | Event end timestamp, retained in raw data but not used by the benchmark loader. |
| `Value` | Numeric value, boolean value, or categorical value. |

Raw static context is expected as one row per patient:

| Column | Meaning |
|---|---|
| `PatientId` | Same identifier as the temporal file. |
| Other columns | Static context features, currently age, gender, admission type, and comorbidity flags. |

Preprocessing writes `data/processed/user_mimic_iv.pkl` as:

```python
[data, labels, train_ids, valid_ids, test_ids, metadata]
```

`data` is a long table consumed by both models:

| Column | Meaning |
|---|---|
| `ts_id` | Patient/admission id. |
| `minute` | Minutes since admission. Temporal inputs are restricted to the first 48 hours; static context is placed at minute `0.0`. |
| `variable` | Input concept or static feature name. Configured outcome concepts and terminal concepts are excluded from inputs. |
| `value` | Numeric input value. Categorical temporal values are expanded to indicator-style variables with value `1.0`. |

`labels` contains one row per patient:

| Column | Meaning |
|---|---|
| `ts_id` | Patient/admission id. |
| outcome columns | Binary labels for whether the outcome occurs during hours 48-336. |
| `{outcome}__first_hour` | First observed outcome hour in the prediction window, retained for traceability only; onset-time MAE is not reported for these baselines. |

The dataset loader converts this processed shape into model-specific tensors:

- STRATS: sparse padded triplets `values`, `times`, `varis`, plus `obs_mask`, `demo`, and multi-label `labels`.
- GRU-D: padded `x_t`, `m_t`, `delta_t`, `seq_len`, `demo`, and multi-label `labels`.

## Preprocess

Build the processed benchmark pickle:

```powershell
& .\.venv\Scripts\python.exe scripts\preprocess_mimic_iv.py
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

Full data process pipeline (from mimic iv to events dataset): https://github.com/shaharoded/MIMIC-IV-ETL

Low-support input concepts and configured outcomes are filtered using
`CONCEPT_SUPPORT_THRESHOLD` and `OUTCOME_SUPPORT_THRESHOLD` in `src/config.py`.

## Train STRATS

On a GPU machine:

```powershell
& .\.venv\Scripts\python.exe main.py --dataset user_mimic_iv --model_type strats --run 1o1 --train_frac 1.0 --device cuda --output_dir outputs/user_mimic_iv/strats
```

If CUDA is available, `--device cuda` can be omitted because the script auto-selects
GPU. Keep it explicit when launching jobs so failures are obvious.

## Train GRU-D

On a GPU machine:

```powershell
& .\.venv\Scripts\python.exe main.py --dataset user_mimic_iv --model_type grud --run 1o1 --train_frac 1.0 --device cuda --output_dir outputs/user_mimic_iv/grud
```

GRU-D uses the same labels, splits, static context, and evaluator as STRATS, but
converts sparse events into value/mask/delta tensors internally.

## Multi-Seed Runs

For confidence intervals, run both models across three paired seeds:

```bash
bash scripts/run_multiseed.sh
```

Defaults:

```text
SEEDS="2023 2024 2025"
MODELS="strats grud"
ROOT="outputs/user_mimic_iv/multiseed"
```

The script skips any seed folder that already has `test_per_outcome_metrics.csv`,
so interrupted runs can be restarted. When all runs finish, it writes:

```text
outputs/user_mimic_iv/multiseed/summary.csv
```

To aggregate manually:

```bash
.venv/bin/python scripts/aggregate_seed_results.py --root outputs/user_mimic_iv/multiseed --models strats grud --output outputs/user_mimic_iv/multiseed/summary.csv
```

## Results

Full benchmark results from the full-data RunPod run are documented in
[RESULTS.md](RESULTS.md).

Each model output directory contains:

- `checkpoint_best.bin`: best validation checkpoint.
- `log.txt`: training and evaluation log.
- `test_per_outcome_metrics.csv`: AUROC/AUPRC/F1/minRP/support by outcome.
- `test_predictions.csv`: patient-level labels and predicted probabilities.
- `test_risk_df.csv`: daily repeated risk rows for compatibility with the evaluation shape.

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
& .\.venv\Scripts\python.exe main.py --dataset user_mimic_iv --model_type strats --run 1o1 --train_frac 0.01 --device cpu --max_epochs 1 --train_batch_size 64 --eval_batch_size 128 --validate_after 999999 --max_obs 64 --hid_dim 16 --num_layers 1 --num_heads 2 --output_dir outputs/smoke/strats
```

GRU-D smoke test:

```powershell
& .\.venv\Scripts\python.exe main.py --dataset user_mimic_iv --model_type grud --run 1o1 --train_frac 0.01 --device cpu --max_epochs 1 --train_batch_size 64 --eval_batch_size 128 --validate_after 999999 --max_timesteps 64 --hid_dim 16 --output_dir outputs/smoke/grud
```
