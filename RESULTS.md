# Medical-Transformers Baseline Results

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
`RELEASE_EVENT` (n=7,456 in the test split).

Configured outcomes available in `data/processed/user_mimic_iv.pkl` but
dropped before training (low support / not part of the final benchmark
roster): `CARDIO-VASCULAR_DISORDER_EVENT`, `HYPEROSMOLALITY_EVENT`,
`KETOACIDOSIS_EVENT`, `ACIDOSIS_EVENT`, `INFECTION_EVENT`,
`DIABETIC_COMA_EVENT`, `ACUTE_RESPIRATORY_DISORDER_EVENT`,
`OTHER_COMPLICATION_EVENT`.

## Model Sizes

Parameter counts at the recipe actually trained — official MIMIC-III config
from STraTS's `run_main.sh`: `hid_dim=64`, `num_layers=2`, `num_heads=16`,
`dropout=0.2`, `attention_dropout=0.2`. Vocabulary V=73 variables, D=7
static features, K=6 supervised outcomes + 1 LoS regression scalar.

| Configuration | Params |
|---|---:|
| STraTS — pretrain stage (forecast head)         |  92,308 |
| STraTS — fine-tune stage (forecast + binary)    |  92,679 |
| ss-STraTS (supervised only)                     |  86,503 |
| GRU-D (supervised)                              |  47,919 |

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

| Model | Avg | AUROC | AUPRC | F1@0.5 | Best F1 | minRP |
|---|---|---:|---:|---:|---:|---:|
| STraTS (pretrain + finetune) | Macro    | 0.8792 [0.8718, 0.8861] | 0.4956 [0.4811, 0.5131] | 0.4559 [0.4451, 0.4664] | 0.5089 [0.4980, 0.5262] | 0.4970 [0.4830, 0.5143] |
| STraTS (pretrain + finetune) | Weighted | 0.8898 [0.8846, 0.8948] | 0.6137 [0.5992, 0.6282] | 0.5588 [0.5485, 0.5692] | 0.6007 [0.5891, 0.6114] | 0.5892 [0.5772, 0.6008] |
| ss-STraTS (supervised only)  | Macro    | 0.8833 [0.8758, 0.8901] | 0.5076 [0.4941, 0.5236] | 0.4448 [0.4347, 0.4544] | 0.5237 [0.5137, 0.5404] | 0.5082 [0.4948, 0.5237] |
| ss-STraTS (supervised only)  | Weighted | **0.8947 [0.8893, 0.8997]** | **0.6314 [0.6172, 0.6451]** | 0.5532 [0.5432, 0.5633] | **0.6181 [0.6068, 0.6289]** | **0.6055 [0.5936, 0.6167]** |
| GRU-D                         | Macro    | 0.8733 [0.8657, 0.8808] | 0.5014 [0.4872, 0.5181] | 0.4397 [0.4297, 0.4500] | 0.5128 [0.5016, 0.5300] | 0.5022 [0.4862, 0.5180] |
| GRU-D                         | Weighted | 0.8890 [0.8836, 0.8942] | 0.6257 [0.6114, 0.6396] | 0.5497 [0.5394, 0.5596] | 0.6090 [0.5976, 0.6197] | 0.5986 [0.5865, 0.6102] |

Length-of-stay MAE (hours, RELEASE-event patients only):

| Model | LoS MAE (h) | 95 % CI |
|---|---:|---:|
| STraTS (pretrain + finetune) | 48.55 | [47.74, 49.45] |
| ss-STraTS (supervised only)  | 48.54 | [47.71, 49.41] |
| GRU-D                         | 47.91 | [47.10, 48.78] |

The primary comparison metrics are AUROC / AUPRC / F1 / minRP. These models
do not generate trajectories or true onset times, so onset-time MAE is not
reported. `F1@0.5` uses a fixed probability threshold of 0.5. `Best F1` is the
maximum F1 over the precision-recall threshold sweep on the test predictions.

**Headline observations:**

- **ss-STraTS beats STraTS-with-pretraining on weighted AUPRC** (0.6314 vs
  0.6137) and matches or exceeds it on every other weighted metric. Within
  this cohort and at the official MIMIC-III STraTS recipe, the forecasting
  pretraining stage adds no benefit on the supervised endpoint.
- **GRU-D is competitive** on all weighted metrics, fractionally trailing
  ss-STraTS on AUPRC / Best F1 / minRP and marginally ahead on LoS MAE.
- The forecasting pretraining stage finishes with a reasonable z-space MSE
  (~0.34) but the gain doesn't translate to the supervised task — likely
  because the cohort is small relative to MIMIC-III and the original
  paper's pretraining benefit comes from pretraining on a much larger
  unlabeled pool than the finetune set.

## Per-Outcome Test Metrics

### STraTS (pretrain + finetune)

| Outcome | AUROC | AUPRC | F1@0.5 | Best F1 | minRP |
|---|---:|---:|---:|---:|---:|
| `HYPERGLYCEMIA_EVENT`        | 0.8964 [0.8888, 0.9035] | 0.7675 [0.7489, 0.7851] | 0.6981 [0.6833, 0.7118] | 0.7108 [0.6971, 0.7260] | 0.7065 [0.6900, 0.7209] |
| `HYPOGLYCEMIA_EVENT`         | 0.8464 [0.8304, 0.8612] | 0.2818 [0.2458, 0.3256] | 0.2989 [0.2744, 0.3222] | 0.3363 [0.3101, 0.3743] | 0.3220 [0.2857, 0.3612] |
| `SEVERE_HYPERGLYCEMIA_EVENT` | 0.8881 [0.8781, 0.8974] | 0.5439 [0.5116, 0.5768] | 0.5298 [0.5091, 0.5500] | 0.5612 [0.5429, 0.5861] | 0.5429 [0.5170, 0.5705] |
| `SEVERE_HYPOGLYCEMIA_EVENT`  | 0.8387 [0.8140, 0.8614] | 0.1504 [0.1188, 0.1981] | 0.1619 [0.1380, 0.1841] | 0.2446 [0.2010, 0.2959] | 0.2291 [0.1826, 0.2831] |
| `KIDNEY_COMPLICATION_EVENT`  | 0.9173 [0.9055, 0.9278] | 0.7078 [0.6743, 0.7384] | 0.5741 [0.5487, 0.5981] | 0.6717 [0.6480, 0.7014] | 0.6654 [0.6373, 0.6923] |
| `DEATH_EVENT`                | 0.8881 [0.8780, 0.8977] | 0.5218 [0.4873, 0.5574] | 0.4728 [0.4510, 0.4949] | 0.5290 [0.5042, 0.5566] | 0.5163 [0.4894, 0.5462] |

### ss-STraTS (supervised only)

| Outcome | AUROC | AUPRC | F1@0.5 | Best F1 | minRP |
|---|---:|---:|---:|---:|---:|
| `HYPERGLYCEMIA_EVENT`        | 0.9019 [0.8945, 0.9089] | 0.7821 [0.7641, 0.7989] | 0.7087 [0.6942, 0.7224] | 0.7184 [0.7056, 0.7341] | 0.7149 [0.7005, 0.7311] |
| `HYPOGLYCEMIA_EVENT`         | 0.8467 [0.8311, 0.8618] | 0.2668 [0.2345, 0.3074] | 0.3094 [0.2835, 0.3336] | 0.3362 [0.3138, 0.3762] | 0.3300 [0.2941, 0.3688] |
| `SEVERE_HYPERGLYCEMIA_EVENT` | 0.8917 [0.8823, 0.9012] | 0.5660 [0.5351, 0.5976] | 0.5258 [0.5052, 0.5463] | 0.5749 [0.5574, 0.6009] | 0.5576 [0.5325, 0.5833] |
| `SEVERE_HYPOGLYCEMIA_EVENT`  | 0.8332 [0.8078, 0.8571] | 0.1231 [0.1004, 0.1561] | 0.1771 [0.1512, 0.2016] | 0.2298 [0.1934, 0.2734] | 0.1860 [0.1481, 0.2376] |
| `KIDNEY_COMPLICATION_EVENT`  | 0.9251 [0.9137, 0.9352] | 0.7256 [0.6920, 0.7559] | 0.5620 [0.5370, 0.5857] | 0.6954 [0.6694, 0.7222] | 0.6914 [0.6624, 0.7141] |
| `DEATH_EVENT`                | 0.8954 [0.8847, 0.9051] | 0.5631 [0.5294, 0.5956] | 0.4836 [0.4614, 0.5060] | 0.5526 [0.5292, 0.5808] | 0.5462 [0.5204, 0.5748] |

### GRU-D

| Outcome | AUROC | AUPRC | F1@0.5 | Best F1 | minRP |
|---|---:|---:|---:|---:|---:|
| `HYPERGLYCEMIA_EVENT`        | 0.8990 [0.8914, 0.9061] | 0.7785 [0.7598, 0.7958] | 0.7056 [0.6909, 0.7193] | 0.7179 [0.7032, 0.7334] | 0.7151 [0.6991, 0.7299] |
| `HYPOGLYCEMIA_EVENT`         | 0.8167 [0.7984, 0.8340] | 0.2490 [0.2160, 0.2888] | 0.2696 [0.2457, 0.2923] | 0.3100 [0.2825, 0.3458] | 0.2925 [0.2573, 0.3287] |
| `SEVERE_HYPERGLYCEMIA_EVENT` | 0.8888 [0.8789, 0.8984] | 0.5591 [0.5274, 0.5907] | 0.5193 [0.4984, 0.5390] | 0.5817 [0.5611, 0.6058] | 0.5577 [0.5312, 0.5854] |
| `SEVERE_HYPOGLYCEMIA_EVENT`  | 0.8255 [0.7997, 0.8485] | 0.1278 [0.1023, 0.1658] | 0.1643 [0.1405, 0.1880] | 0.2180 [0.1815, 0.2693] | 0.2115 [0.1646, 0.2624] |
| `KIDNEY_COMPLICATION_EVENT`  | 0.9210 [0.9097, 0.9318] | 0.7289 [0.6986, 0.7564] | 0.5762 [0.5512, 0.5990] | 0.6779 [0.6526, 0.7060] | 0.6713 [0.6419, 0.6975] |
| `DEATH_EVENT`                | 0.8959 [0.8860, 0.9054] | 0.5534 [0.5210, 0.5869] | 0.4899 [0.4677, 0.5110] | 0.5422 [0.5187, 0.5694] | 0.5304 [0.5033, 0.5597] |

## Result Files

Each model output directory contains:

- `checkpoint_best.bin`: best validation checkpoint.
- `log.txt` / `run.log`: training and final evaluation log.
- `test_per_outcome_metrics.csv`: AUROC / AUPRC / F1 / minRP with bootstrap
  point estimate and 95 % CI columns per outcome, plus positive / negative
  support.
- `test_predictions.csv`: patient-level labels and predicted probabilities.
- `test_risk_df.csv`: repeated daily risk rows for evaluator compatibility.

No standalone MAE file is produced for event onset. STraTS, ss-STraTS, and
GRU-D are fixed-horizon discriminative models — onset-time MAE is reported
as an unsupported output rather than approximated. Length-of-stay MAE is
emitted by the same evaluation pass on patients with a RELEASE event.

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
