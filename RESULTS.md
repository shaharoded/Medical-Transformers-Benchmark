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
/workspace/runs/results/baselines/strats
/workspace/runs/results/baselines/grud
```

## Dataset Snapshot

| Item | Value |
|---|---:|
| Temporal rows | 11,577,364 |

Configured prediction targets that passed the support threshold and trained:

- `HYPERGLYCEMIA_EVENT`
- `HYPOGLYCEMIA_EVENT`
- `SEVERE_HYPERGLYCEMIA_EVENT`
- `SEVERE_HYPOGLYCEMIA_EVENT`
- `KIDNEY_COMPLICATION_EVENT`
- `DEATH_EVENT`

A regression head also predicts `length_of_stay_hours` for patients with a
`RELEASE_EVENT` (n=9,978 in the test split).

Configured outcomes available in `data/processed/user_mimic_iv.pkl` but
dropped before training (low support / not part of the final benchmark
roster): `CARDIO-VASCULAR_DISORDER_EVENT`, `HYPEROSMOLALITY_EVENT`,
`KETOACIDOSIS_EVENT`, `ACIDOSIS_EVENT`, `INFECTION_EVENT`,
`DIABETIC_COMA_EVENT`, `ACUTE_RESPIRATORY_DISORDER_EVENT`,
`OTHER_COMPLICATION_EVENT`.

## Model Sizes

Parameter counts on the `user_mimic_iv` task (V=73 variables, D=7 static
features, K=6 supervised outcomes + 1 LoS regression scalar). Two
configurations are reported: the **Small** column matches `main.py`'s
argparse defaults in this repo (the architecture the current results below
were trained at); the **Full** column is the larger configuration adopted
by the updated training recipe in the README.

| Configuration | Small (hid=32, nh=4) | Full (hid=64, nh=16) |
|---|---:|---:|
| ss-STraTS (supervised only) | 23,387 | 86,055 |
| STraTS — pretrain stage (forecast head) | 27,677 | 94,569 |
| STraTS — fine-tune stage (forecast + binary) | 28,195 | 95,087 |
| GRU-D (supervised) | 22,745 | 55,577 |

The fine-tune stage of STraTS trains the same supervised task as ss-STraTS
— same `Dataset` (the `_finetune.pkl`), same multi-label BCE + LoS-MSE loss,
same `Evaluator`, same bootstrap. The only architectural difference is that
fine-tune stacks the pretrained `forecast_head` (Linear → V) under the
`binary_head` (Linear → K+1) so the supervised stage can leverage the
forecasting bottleneck learned in stage 1, while ss-STraTS puts `binary_head`
directly on the backbone embedding. Comparing the two isolates the
contribution of self-supervised pretraining on the same supervised endpoint.

## Overall Test Results

Values are point estimates and 95 % CIs from a B=2000 patient-level bootstrap
on the held-out test split (`--bootstrap_resamples 2000`, default in `main.py`).
Macro is an unweighted mean across the six trained outcomes; weighted uses
each outcome's positive support (`n_pos`) as the weight.

| Model | Average | AUROC | AUPRC | F1@0.5 | Best F1 | minRP |
|---|---|---:|---:|---:|---:|---:|
| STRATS | Macro    | 0.8792 [0.8727, 0.8856] | 0.4937 [0.4810, 0.5086] | 0.4716 [0.4614, 0.4814] | 0.5085 [0.4997, 0.5230] | 0.4904 [0.4784, 0.5046] |
| STRATS | Weighted | 0.8896 [0.8848, 0.8946] | 0.6122 [0.5996, 0.6244] | 0.5677 [0.5585, 0.5769] | 0.5972 [0.5880, 0.6069] | 0.5822 [0.5718, 0.5924] |
| GRU-D  | Macro    | 0.8718 [0.8649, 0.8786] | 0.4837 [0.4717, 0.4983] | 0.4399 [0.4307, 0.4496] | 0.4998 [0.4916, 0.5144] | 0.4904 [0.4772, 0.5032] |
| GRU-D  | Weighted | 0.8864 [0.8814, 0.8915] | 0.6065 [0.5937, 0.6191] | 0.5428 [0.5335, 0.5521] | 0.5952 [0.5860, 0.6046] | 0.5853 [0.5749, 0.5952] |

Length-of-stay MAE (hours, RELEASE-event patients only, n=9,978):

| Model | LoS MAE (h) | 95 % CI |
|---|---:|---:|
| STRATS | 48.05 | [47.36, 48.79] |
| GRU-D  | 47.64 | [46.93, 48.39] |

The primary comparison metrics are AUROC / AUPRC / F1 / minRP. These models
do not generate trajectories or true onset times, so onset-time MAE is not
reported. `F1@0.5` uses a fixed probability threshold of 0.5. `Best F1` is the
maximum F1 over the precision-recall threshold sweep on the test predictions.

The two baselines are within bootstrap noise on the weighted headline metrics
(AUROC, AUPRC, Best F1, minRP). STRATS edges GRU-D on `F1@0.5` (~+0.025
absolute) and is fractionally ahead on weighted AUROC; weighted AUPRC is
essentially tied. GRU-D is fractionally ahead on length-of-stay MAE.

## Per-Outcome Test Metrics

### STRATS

| Outcome | AUROC | AUPRC | F1@0.5 | Best F1 | minRP |
|---|---:|---:|---:|---:|---:|
| `HYPERGLYCEMIA_EVENT`        | 0.8988 [0.8927, 0.9049] | 0.7686 [0.7532, 0.7841] | 0.7012 [0.6887, 0.7147] | 0.7099 [0.6987, 0.7243] | 0.7055 [0.6912, 0.7183] |
| `HYPOGLYCEMIA_EVENT`         | 0.8459 [0.8310, 0.8591] | 0.2860 [0.2533, 0.3203] | 0.3109 [0.2891, 0.3324] | 0.3533 [0.3284, 0.3847] | 0.3323 [0.3013, 0.3661] |
| `SEVERE_HYPERGLYCEMIA_EVENT` | 0.8865 [0.8778, 0.8951] | 0.5583 [0.5293, 0.5850] | 0.5320 [0.5137, 0.5497] | 0.5653 [0.5470, 0.5868] | 0.5458 [0.5209, 0.5686] |
| `SEVERE_HYPOGLYCEMIA_EVENT`  | 0.8286 [0.8064, 0.8495] | 0.1375 [0.1144, 0.1699] | 0.1784 [0.1574, 0.2009] | 0.2218 [0.1961, 0.2606] | 0.2006 [0.1621, 0.2421] |
| `KIDNEY_COMPLICATION_EVENT`  | 0.9049 [0.8941, 0.9153] | 0.6431 [0.6132, 0.6720] | 0.5125 [0.4895, 0.5343] | 0.6099 [0.5878, 0.6370] | 0.6029 [0.5778, 0.6275] |
| `DEATH_EVENT`                | 0.8931 [0.8832, 0.9021] | 0.5178 [0.4855, 0.5496] | 0.4810 [0.4606, 0.5006] | 0.5330 [0.5112, 0.5580] | 0.5227 [0.4981, 0.5495] |

### GRU-D

| Outcome | AUROC | AUPRC | F1@0.5 | Best F1 | minRP |
|---|---:|---:|---:|---:|---:|
| `HYPERGLYCEMIA_EVENT`        | 0.8965 [0.8899, 0.9030] | 0.7694 [0.7538, 0.7849] | 0.7001 [0.6870, 0.7135] | 0.7146 [0.7017, 0.7281] | 0.7049 [0.6919, 0.7200] |
| `HYPOGLYCEMIA_EVENT`         | 0.8197 [0.8040, 0.8344] | 0.2384 [0.2106, 0.2702] | 0.2882 [0.2677, 0.3099] | 0.3184 [0.2962, 0.3491] | 0.3065 [0.2752, 0.3391] |
| `SEVERE_HYPERGLYCEMIA_EVENT` | 0.8879 [0.8786, 0.8966] | 0.5657 [0.5377, 0.5929] | 0.5248 [0.5068, 0.5433] | 0.5705 [0.5530, 0.5932] | 0.5617 [0.5390, 0.5855] |
| `SEVERE_HYPOGLYCEMIA_EVENT`  | 0.8154 [0.7928, 0.8377] | 0.1267 [0.1063, 0.1581] | 0.1912 [0.1702, 0.2147] | 0.2217 [0.1946, 0.2630] | 0.1898 [0.1590, 0.2332] |
| `KIDNEY_COMPLICATION_EVENT`  | 0.9072 [0.8961, 0.9180] | 0.6798 [0.6503, 0.7081] | 0.5230 [0.4994, 0.5447] | 0.6521 [0.6289, 0.6778] | 0.6439 [0.6208, 0.6699] |
| `DEATH_EVENT`                | 0.8972 [0.8882, 0.9056] | 0.5207 [0.4895, 0.5521] | 0.4563 [0.4357, 0.4747] | 0.5331 [0.5129, 0.5581] | 0.5277 [0.5014, 0.5514] |

## Result Files

Each model output directory contains:

- `checkpoint_best.bin`: best validation checkpoint.
- `log.txt` / `run.log`: training and final evaluation log.
- `test_per_outcome_metrics.csv`: AUROC / AUPRC / F1 / minRP with bootstrap
  point estimate and 95 % CI columns per outcome, plus positive / negative
  support.
- `test_predictions.csv`: patient-level labels and predicted probabilities.
- `test_risk_df.csv`: repeated daily risk rows for evaluator compatibility.

No standalone MAE file is produced for event onset. STRATS and GRU-D are
fixed-horizon discriminative models — onset-time MAE is reported as an
unsupported output rather than approximated. Length-of-stay MAE is emitted
by the same evaluation pass on patients with a RELEASE event.

## Commands Used

Preprocess (one-time, builds `data/processed/user_mimic_iv_finetune.pkl` and
`data/processed/user_mimic_iv_pretrain.pkl`):

```bash
python3 scripts/preprocess_mimic_iv.py --seed 42
```

STraTS — stage 1 (forecasting pretraining):

```bash
python3 main.py --pretrain 1 --dataset user_mimic_iv --model_type strats \
  --hid_dim 64 --num_layers 2 --num_heads 16 \
  --dropout 0.2 --attention_dropout 0.2 \
  --lr 5e-4 --max_epochs 30 \
  --train_batch_size 16 --eval_batch_size 32 \
  --device cuda
```

STraTS — stage 2 (supervised fine-tune):

```bash
python3 main.py --dataset user_mimic_iv --model_type strats \
  --hid_dim 64 --num_layers 2 --num_heads 16 \
  --dropout 0.2 --attention_dropout 0.2 \
  --lr 5e-5 --max_epochs 50 \
  --run 1o1 --train_frac 1.0 \
  --bootstrap_resamples 2000 \
  --load_ckpt_path outputs/user_mimic_iv/pretrain/strats,num_layers:2,hid_dim:64,num_heads:16,dropout:0.2,attention_dropout:0.2,lr:0.0005/checkpoint_best.bin \
  --device cuda
```

ss-STraTS (same architecture, supervised only):

```bash
python3 main.py --dataset user_mimic_iv --model_type strats \
  --hid_dim 64 --num_layers 2 --num_heads 16 \
  --dropout 0.2 --attention_dropout 0.2 \
  --lr 5e-4 --max_epochs 50 \
  --run 1o1 --train_frac 1.0 \
  --bootstrap_resamples 2000 \
  --device cuda
```

GRU-D:

```bash
python3 main.py --dataset user_mimic_iv --model_type grud \
  --hid_dim 64 --dropout 0.2 \
  --lr 5e-4 --max_epochs 50 \
  --run 1o1 --train_frac 1.0 \
  --bootstrap_resamples 2000 \
  --device cuda
```
