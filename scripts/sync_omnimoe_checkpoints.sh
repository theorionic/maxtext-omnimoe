#!/bin/bash
# Mirror OmniMoE-8B checkpoints from local disk to Google Drive via rclone.
#
# Checkpoints are written to ${HOME}/omnimoe_8b_output by
# scripts/train_omnimoe_8b_ultrafineweb.sh. That directory is on the local
# overlay and is not durable, so this script pushes it to Drive.
#
# Usage:
#   bash scripts/sync_omnimoe_checkpoints.sh            # one-shot sync
#   WATCH=1 bash scripts/sync_omnimoe_checkpoints.sh    # re-sync every ${INTERVAL}s
#
# Optional env vars:
#   OUTPUT_DIR   local dir to sync (default ${HOME}/omnimoe_8b_output)
#   REMOTE       rclone destination (default gdrive:omnimoe_8b_output)
#   INTERVAL     seconds between passes when WATCH=1 (default 600)
#   TRANSFERS    parallel file transfers (default 8)

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/omnimoe_8b_output}"
REMOTE="${REMOTE:-gdrive:omnimoe_8b_output}"
INTERVAL="${INTERVAL:-600}"
TRANSFERS="${TRANSFERS:-8}"

if [[ ! -d "${OUTPUT_DIR}" ]]; then
  echo "ERROR: ${OUTPUT_DIR} does not exist. Start training first." >&2
  exit 1
fi

# `sync` (not `copy`) so that checkpoints retired locally by
# max_num_checkpoints_to_keep are also retired on Drive, keeping the two sides
# identical instead of letting Drive grow without bound.
#
# --checkers/--transfers are modest on purpose: Drive rate-limits aggressively
# and orbax checkpoints are thousands of small files.
run_sync() {
  echo "[$(date '+%F %T')] syncing ${OUTPUT_DIR} -> ${REMOTE}"
  rclone sync "${OUTPUT_DIR}" "${REMOTE}" \
    --transfers "${TRANSFERS}" \
    --checkers 16 \
    --drive-chunk-size 128M \
    --drive-acknowledge-abuse \
    --fast-list \
    --retries 5 \
    --low-level-retries 20 \
    --stats 30s \
    --stats-one-line \
    --exclude "**/*.tmp" \
    --exclude "**/.todelete/**" \
    --exclude "**/*.orbax-checkpoint-tmp*/**" \
    --progress
  echo "[$(date '+%F %T')] sync complete"
}

if [[ "${WATCH:-0}" == "1" ]]; then
  while true; do
    # Never let a transient Drive error kill a long-running watch loop.
    run_sync || echo "[$(date '+%F %T')] sync failed, retrying in ${INTERVAL}s" >&2
    sleep "${INTERVAL}"
  done
else
  run_sync
fi
