#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-2023 2024 2025}"
MODELS="${MODELS:-strats grud}"
ROOT="${ROOT:-outputs/user_mimic_iv/multiseed}"
DATASET="${DATASET:-user_mimic_iv}"
RUN="${RUN:-1o1}"
TRAIN_FRAC="${TRAIN_FRAC:-1.0}"
DEVICE="${DEVICE:-cuda}"

for model in ${MODELS}; do
  for seed in ${SEEDS}; do
    out_dir="${ROOT}/${model}/seed_${seed}"
    if [[ -f "${out_dir}/test_per_outcome_metrics.csv" ]]; then
      echo "Skipping completed ${model} seed ${seed}: ${out_dir}"
      continue
    fi
    echo "Running ${model} seed ${seed}: ${out_dir}"
    .venv/bin/python main.py \
      --dataset "${DATASET}" \
      --model_type "${model}" \
      --run "${RUN}" \
      --train_frac "${TRAIN_FRAC}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      --output_dir "${out_dir}"
  done
done

.venv/bin/python scripts/aggregate_seed_results.py \
  --root "${ROOT}" \
  --models ${MODELS} \
  --output "${ROOT}/summary.csv"
