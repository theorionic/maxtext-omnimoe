#!/bin/bash
# Train OmniMoE-8B (8 experts x ~1B, one expert per v5e chip) on UltraFineWeb.
#
# Usage:
#   bash scripts/train_omnimoe_8b_ultrafineweb.sh
#
# Optional env vars:
#   HF_TOKEN         HuggingFace token (avoids rate limits; UltraFineWeb is public)
#   STEPS            training steps (default 50000)
#   HF_SUBSET        train subdir: "ultrafineweb_en" (default) or "ultrafineweb_zh"
#   OUTPUT_DIR       checkpoint/output dir (default /kaggle/working/omnimoe_8b_output)
#   LEARNING_RATE    peak LR (default 3.0e-4)
#
#   --- Eval loss on a held-out / different dataset (built-in MaxText eval loop) ---
#   EVAL_INTERVAL    run eval every N steps (default 2000; set <=0 to disable eval)
#   EVAL_STEPS       number of eval batches per eval pass (default 20)
#   HF_EVAL_SUBSET   eval subdir under the SAME repo, should DIFFER from HF_SUBSET.
#                    Default "ultrafineweb_zh" so the eval-loss signal comes from data
#                    the model never trains on. (MaxText's HF eval iterator reuses
#                    hf_path=openbmb/Ultra-FineWeb, so eval data must be a different
#                    subset/file-glob of that repo — set HF_EVAL_SUBSET to a held-out
#                    English shard glob if you prefer a same-distribution eval.)
#
#   --- Sample text generation during training (qualitative health probe) ---
#   GEN_INTERVAL     greedy-generate every N steps (default 2000; set <=0 to disable)
#   GEN_MAX_NEW      new tokens generated per prompt (default 48)
#   GEN_PROMPTS      '|||'-separated prompts to sample each time
#
# Note on MFU: the one-expert-per-chip / top-1 / expert_parallelism=8 layout is
# all-to-all-communication bound, so ~15-16% MFU on v5e-8 is near the practical
# ceiling for this design (benchmarked: full remat + bs=2 + seq=1024 is optimal;
# larger batch/seq and the save_dot_except_mlpwi remat policy either regress MFU,
# OOM, or diverge to NaN at gradient_accumulation_steps=8). Leave remat_policy=full.

set -euo pipefail

MAXTEXT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${MAXTEXT_ROOT}"

# The HF data pipeline imports `transformers`, which lets `accelerate` import
# `torch_xla`. On a TPU box torch_xla grabs the device that JAX already owns and
# the process segfaults. USE_TORCH_XLA=0 makes accelerate skip that import.
export USE_TORCH_XLA=0

STEPS="${STEPS:-50000}"
HF_SUBSET="${HF_SUBSET:-ultrafineweb_en}"
OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/working/omnimoe_8b_output}"
LEARNING_RATE="${LEARNING_RATE:-3.0e-4}"

# --- Eval config ---
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
EVAL_STEPS="${EVAL_STEPS:-20}"
HF_EVAL_SUBSET="${HF_EVAL_SUBSET:-ultrafineweb_zh}"

# --- Generation config ---
GEN_INTERVAL="${GEN_INTERVAL:-2000}"
GEN_MAX_NEW="${GEN_MAX_NEW:-48}"
GEN_PROMPTS="${GEN_PROMPTS:-The history of artificial intelligence|||In the future, renewable energy will|||The most important scientific discovery of the century was}"

if [ "${HF_EVAL_SUBSET}" = "${HF_SUBSET}" ] && [ "${EVAL_INTERVAL}" -gt 0 ] 2>/dev/null; then
  echo "WARNING: HF_EVAL_SUBSET == HF_SUBSET (${HF_SUBSET}); eval data overlaps training data." >&2
fi

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
  checkpoint_period=2000 \
  skip_jax_distributed_system=true \
  eval_interval="${EVAL_INTERVAL}" \
  eval_steps="${EVAL_STEPS}" \
  eval_per_device_batch_size=2 \
  hf_eval_files="data/${HF_EVAL_SUBSET}/*.parquet" \
  hf_eval_split=train \
  eval_data_columns="['content']" \
  tokenize_eval_data=true \
  generate_interval="${GEN_INTERVAL}" \
  generate_max_new_tokens="${GEN_MAX_NEW}" \
  generate_prompts="${GEN_PROMPTS}"
