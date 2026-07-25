#!/bin/bash
# Train OmniMoE-8B (8 experts x ~1B, one expert per v5e chip) on UltraFineWeb.
#
# Usage:
#   bash scripts/train_omnimoe_8b_ultrafineweb.sh
#
# Optional env vars:
#   HF_TOKEN         HuggingFace token (avoids rate limits; UltraFineWeb is public)
#   STEPS            training steps (default 50000)
#   HF_SUBSET        dataset subdir: "ultrafineweb_en" (default) or "ultrafineweb_zh"
#   OUTPUT_DIR       checkpoint/output dir (default ${HOME}/omnimoe_8b_output)
#   LEARNING_RATE    peak LR (default 3.0e-4)
#   CKPT_PERIOD      steps between checkpoints (default 2000)
#   KEEP_CKPTS       how many checkpoints to retain on local disk (default 3)
#
# Checkpoints live under ${HOME} rather than /kaggle/working: /kaggle/working is a
# 20G loop device, far too small for this model (~50G of params+AdamW state per
# checkpoint), while ${HOME} sits on the ~1T overlay. Use
# scripts/sync_omnimoe_checkpoints.sh to mirror them to Google Drive.

set -euo pipefail

MAXTEXT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${MAXTEXT_ROOT}"

# The HF data pipeline imports `transformers`, which lets `accelerate` import
# `torch_xla`. On a TPU box torch_xla grabs the device that JAX already owns and
# the process segfaults. USE_TORCH_XLA=0 makes accelerate skip that import.
export USE_TORCH_XLA=0

STEPS="${STEPS:-50000}"
HF_SUBSET="${HF_SUBSET:-ultrafineweb_en}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/omnimoe_8b_output}"
LEARNING_RATE="${LEARNING_RATE:-3.0e-4}"
CKPT_PERIOD="${CKPT_PERIOD:-2000}"
KEEP_CKPTS="${KEEP_CKPTS:-3}"

mkdir -p "${OUTPUT_DIR}"

PYTHONPATH=src python3 -m maxtext.trainers.pre_train.train \
  src/maxtext/configs/base.yml \
  model_name=omnimoe_8b \
  run_name=omnimoe_8b_ultrafineweb \
  base_output_directory="${OUTPUT_DIR}" \
  dataset_type=hf \
  hf_path=openbmb/Ultra-FineWeb \
  hf_train_files="data/${HF_SUBSET}/*.parquet" \
  train_split=train \
  train_data_columns="['content']" \
  tokenizer_type=huggingface \
  tokenizer_path=hf-internal-testing/llama-tokenizer \
  tokenize_train_data=true \
  hf_access_token="${HF_TOKEN:-}" \
  steps="${STEPS}" \
  learning_rate="${LEARNING_RATE}" \
  warmup_steps_fraction=0.01 \
  enable_checkpointing=true \
  checkpoint_period="${CKPT_PERIOD}" \
  max_num_checkpoints_to_keep="${KEEP_CKPTS}" \
  eval_interval=-1 \
  skip_jax_distributed_system=true
