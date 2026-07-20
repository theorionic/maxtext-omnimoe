#!/bin/bash
# Training run for OmniMoE on TPU v5e-8.
#
# Uses src/maxtext/configs/models/omnimoe.yml as-is: per_device_batch_size=64,
# remat_policy=full, and GMM tile sizes tuned to the model's 512-wide expert
# FFN. That combination was benchmarked at ~33.5 TFLOP/s/device steady state
# (~17% MFU vs v5e's 197 TFLOP/s/chip bf16 peak) with 12.7GB/15.75GB HBM used
# per chip, so no batch/remat/tile overrides are passed here.
#
# IMPORTANT: DATASET_TYPE defaults to "synthetic" (random data) below, which is
# only useful for benchmarking/smoke-testing the training loop. Set DATASET_TYPE
# (and whatever dataset_path/hf_name args your real dataset needs) before using
# this for an actual training run.
#
# Optional env vars: RUN_NAME, OUTPUT_DIR, DATASET_TYPE, STEPS, ENABLE_CHECKPOINTING.
#
# Usage:
#   bash scripts/run_omnimoe_train.sh

set -euo pipefail

MAXTEXT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

RUN_NAME="${RUN_NAME:-omnimoe_run1}"
OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/working/omnimoe_output}"
DATASET_TYPE="${DATASET_TYPE:-synthetic}"
STEPS="${STEPS:-1000}"
ENABLE_CHECKPOINTING="${ENABLE_CHECKPOINTING:-true}"

cd "${MAXTEXT_ROOT}"

PYTHONPATH=src python3 -m maxtext.trainers.pre_train.train \
  src/maxtext/configs/base.yml \
  model_name=omnimoe \
  run_name="${RUN_NAME}" \
  base_output_directory="${OUTPUT_DIR}" \
  dataset_type="${DATASET_TYPE}" \
  steps="${STEPS}" \
  enable_checkpointing="${ENABLE_CHECKPOINTING}"
