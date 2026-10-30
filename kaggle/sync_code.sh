#!/usr/bin/env bash
# Stage src/ + configs/ into the private code dataset and create or version it.
# Usage: bash kaggle/sync_code.sh "message"
set -euo pipefail
cd "$(dirname "$0")/.."
D="$(cygpath -w "$PWD/kaggle/_code_dataset")"
rm -rf "$D/src" "$D/configs" "$D/data"; cp -r src configs "$D/"; mkdir -p "$D/data"
cp data_splits/splits.csv "$D/data/"
cp legacy/brats-supplied-grid-full-cohort-qc/output/brats_supplied_grid_qc/modality_stats.csv "$D/data/brats_modality_stats_owner.csv"
find "$D" -name __pycache__ -prune -exec rm -rf {} +
git rev-parse HEAD > "$D/GIT_COMMIT" 2>/dev/null || echo nogit > "$D/GIT_COMMIT"
K=.venv/Scripts/kaggle.exe
if $K datasets status leweletp/brats-dual-code >/dev/null 2>&1; then
  $K datasets version -p "$D" -m "${1:-update}" -r zip -q
else
  $K datasets create -p "$D" -r zip -q
fi
