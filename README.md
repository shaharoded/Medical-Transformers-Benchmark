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

RunPod full-data test results, reported as point estimate and 95 % CI from a
B=2000 patient-level bootstrap on the held-out test split
(`--bootstrap_resamples 2000` is the `main.py` default). Macro is an unweighted
mean across outcomes; weighted uses each outcome's positive support (`n_pos`)
as the weight.

| Model | Average | AUROC | AUPRC | F1@0.5 | Best F1 | minRP |
|---|---|---:|---:|---:|---:|---:|
| STRATS | Macro    | 0.8792 [0.8727, 0.8856] | 0.4937 [0.4810, 0.5086] | 0.4716 [0.4614, 0.4814] | 0.5085 [0.4997, 0.5230] | 0.4904 [0.4784, 0.5046] |
| STRATS | Weighted | 0.8896 [0.8848, 0.8946] | 0.6122 [0.5996, 0.6244] | 0.5677 [0.5585, 0.5769] | 0.5972 [0.5880, 0.6069] | 0.5822 [0.5718, 0.5924] |
| GRU-D  | Macro    | 0.8718 [0.8649, 0.8786] | 0.4837 [0.4717, 0.4983] | 0.4399 [0.4307, 0.4496] | 0.4998 [0.4916, 0.5144] | 0.4904 [0.4772, 0.5032] |
| GRU-D  | Weighted | 0.8864 [0.8814, 0.8915] | 0.6065 [0.5937, 0.6191] | 0.5428 [0.5335, 0.5521] | 0.5952 [0.5860, 0.6046] | 0.5853 [0.5749, 0.5952] |

Length-of-stay MAE on patients with a RELEASE event (n=9978):
STRATS 48.05 h [47.36, 48.79]; GRU-D 47.64 h [46.93, 48.39].

The two baselines are within bootstrap noise on the headline weighted metrics
(AUROC, AUPRC, Best F1, minRP). STRATS edges GRU-D on `F1@0.5` (~+0.025
absolute) and is fractionally ahead on weighted AUROC; weighted AUPRC is
essentially tied. `F1@0.5` uses a fixed probability threshold of 0.5;
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

The configured targets live in `src/config.py`. The six that pass the support
threshold and train are:

- `HYPERGLYCEMIA_EVENT`, `SEVERE_HYPERGLYCEMIA_EVENT`
- `HYPOGLYCEMIA_EVENT`, `SEVERE_HYPOGLYCEMIA_EVENT`
- `KIDNEY_COMPLICATION_EVENT`
- `DEATH_EVENT`

Disglycemia outcomes are synthesized from glucose measurements during
preprocessing (thresholds in `src/config.py`):

- `HYPOGLYCEMIA`: glucose `<= 70` (mg/dL); `SEVERE_HYPOGLYCEMIA`: `<= 54`.
- `HYPERGLYCEMIA`: glucose `>= 180`; `SEVERE_HYPERGLYCEMIA`: `>= 250`.

Full data process pipeline (from mimic iv to events dataset): https://github.com/shaharoded/MIMIC-IV-ETL

### What the model sees

The raw input file (`data/mimic-iv-input-data.csv`) has roughly **21.4 M rows**
across **57 k patients**. After preprocessing, **~11.6 M rows** remain in
the temporal input table. The drop is design, not noise:

- **Input window**: only events in the first 48 h after `ADMISSION` are kept as
  inputs. STraTS / GRU-D do not pretrain on the 48-336 h horizon — those rows
  are used exclusively to construct labels (`{outcome}` and
  `{outcome}__first_hour`). The 48 h truncation is the dominant cut.
- **Outcome events routed to labels**: rows whose `ConceptName` is one of the
  configured outcomes (`DEATH_EVENT`, `HYPERGLYCEMIA_EVENT`, etc.) are removed
  from the inputs so they don't leak the label.
- **Pre-admission rows** (`minute < 0`) are dropped.
- **Mediator bounds filter** drops out-of-range numeric values for measurements
  that have a clinical range (small fraction of rows).

**Non-numeric `Value` rows are NOT dropped.** Roughly 7.3 % of input rows
(1.56 M) have a text `Value` (`True` flags on BITZUA events, `MEAL` categories,
`ADMISSION`/`RELEASE` markers, …). The preprocess expands each one into an
indicator-style variable: a new variable name `{ConceptName}_{Value}` is
emitted with numeric `value = 1.0`. Every concept that exists in the input
file is therefore visible to STraTS and GRU-D, either as the original numeric
measurement or as a one-hot presence indicator.

Low-support input concepts and configured outcomes are filtered using
`CONCEPT_SUPPORT_THRESHOLD` and `OUTCOME_SUPPORT_THRESHOLD` in `src/config.py`.

## Train STRATS

The original STraTS paper (Tipirneni & Reddy 2022) defines two configurations:

- **STraTS** — full configuration. **Two stages**: (1) self-supervised
  forecasting pretraining on the unlabeled trajectory (`--pretrain 1`), then
  (2) supervised fine-tuning with the pretrained backbone loaded via
  `--load_ckpt_path`.
- **ss-STraTS** — supervised-only variant. The same architecture trained
  end-to-end with the multi-label BCE loss, **no pretraining**.

Both use the **identical architecture and hyperparameters** specified in the
official repository's `run_main.sh` for MIMIC-III (the closer comparator to
this cohort): `hid_dim=64`, `num_layers=2`, `num_heads=16`, `dropout=0.2`,
`attention_dropout=0.2`. The only differences between the three commands
below are the `--pretrain` / `--load_ckpt_path` flags and the learning rate
(pretrain and ss-STraTS use `5e-4`; fine-tune uses `5e-5`, 10× lower as in
the paper).

### Run STraTS (pretrain + fine-tune)

```bash
# Stage 1 — forecasting pretraining (no labels). Runs on the full 0-336 h
# trajectory in *_pretrain.pkl. Saves checkpoint_best.bin under
# outputs/user_mimic_iv/pretrain/strats,...
python main.py \
  --pretrain 1 \
  --dataset user_mimic_iv \
  --model_type strats \
  --hid_dim 64 --num_layers 2 --num_heads 16 \
  --dropout 0.2 --attention_dropout 0.2 \
  --lr 5e-4 --max_epochs 30 \
  --train_batch_size 16 --eval_batch_size 32 \
  --device cuda

# Stage 2 — supervised fine-tune (loads the stage-1 forecast backbone).
# Path is the file written by stage 1; substitute the auto-suffixed dir.
python main.py \
  --dataset user_mimic_iv \
  --model_type strats \
  --hid_dim 64 --num_layers 2 --num_heads 16 \
  --dropout 0.2 --attention_dropout 0.2 \
  --lr 5e-5 --max_epochs 50 \
  --run 1o1 --train_frac 1.0 \
  --bootstrap_resamples 2000 \
  --load_ckpt_path outputs/user_mimic_iv/pretrain/strats,num_layers:2,hid_dim:64,num_heads:16,dropout:0.2,attention_dropout:0.2,lr:0.0005/checkpoint_best.bin \
  --device cuda
```

### Run ss-STraTS (supervised-only, same architecture)

```bash
python main.py \
  --dataset user_mimic_iv \
  --model_type strats \
  --hid_dim 64 --num_layers 2 --num_heads 16 \
  --dropout 0.2 --attention_dropout 0.2 \
  --lr 5e-4 --max_epochs 50 \
  --run 1o1 --train_frac 1.0 \
  --bootstrap_resamples 2000 \
  --device cuda
```

If CUDA is available, `--device cuda` can be omitted because the script
auto-selects GPU. Keep it explicit when launching jobs so failures are
obvious.

## Train GRU-D

The official STraTS codebase does **not** pretrain GRU-D — only the STraTS
transformer is pretrained, because the forecasting objective is defined on the
sparse-triplet representation, not on GRU-D's value/mask/delta tensors. GRU-D
is trained supervised-only with the MIMIC-III defaults from `run_main.sh`
(`hid_dim=64`, `dropout=0.2`, `lr=5e-4`):

```bash
python main.py \
  --dataset user_mimic_iv \
  --model_type grud \
  --hid_dim 64 --dropout 0.2 \
  --lr 5e-4 --max_epochs 50 \
  --run 1o1 --train_frac 1.0 \
  --bootstrap_resamples 2000 \
  --device cuda
```

GRU-D uses the same labels, splits, static context, and evaluator as STRATS, but
converts sparse events into value/mask/delta tensors internally.


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
