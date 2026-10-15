#!/usr/bin/env bash
# One command: download the pinned data (if missing) and run the 1 h pipeline.
# Usage (from anywhere, inside an activated virtualenv with requirements installed):
#   bash scripts/run_all.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ ! -f data/raw/volunteer_data_en.csv ]; then
  bash scripts/fetch_data.sh
fi
python -m src.train --data-dir data/raw --horizon 1h --out reports
