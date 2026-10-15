#!/usr/bin/env bash
# Fetches the PINNED commit (full 40-char SHA, not "latest") of the
# air-raid-sirens dataset and copies the two CSVs we use into data/raw/.
#
# We do not vendor the CSVs in this repo (unclear redistribution terms for
# the underlying Telegram-scraped data), so this script plus
# src/config.py:PINNED_COMMIT are the single source of truth for "which
# data version was used".
#
# Shallow fetch by SHA (depth 1): the upstream repo re-commits ~30 MB CSVs
# daily, so a full clone downloads a very large history we do not need.
#
# Optional: DATA_DIR=/some/dir bash scripts/fetch_data.sh  (default: data/raw)
set -euo pipefail

REPO_URL="https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset.git"
PINNED_COMMIT="22d0443d1d1cdf6a0a413c6d15c3078b5ed26127"  # 2026-10-15 03:50:46 UTC

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DATA_DIR:-$ROOT/data/raw}"
WORK="$DEST/_upstream"   # git work tree; data/raw/ is gitignored

mkdir -p "$WORK"
cd "$WORK"
[ -d .git ] || git init -q
git remote get-url origin >/dev/null 2>&1 || git remote add origin "$REPO_URL"
git fetch -q --depth 1 origin "$PINNED_COMMIT"
git checkout -q --detach FETCH_HEAD

ACTUAL="$(git rev-parse HEAD)"
if [ "$ACTUAL" != "$PINNED_COMMIT" ]; then
  echo "ERROR: checked out $ACTUAL but expected $PINNED_COMMIT" >&2
  exit 1
fi

cp datasets/official_data_en.csv datasets/volunteer_data_en.csv "$DEST"/
echo "Fetched dataset at pinned commit $ACTUAL into $DEST"
sha256sum "$DEST"/official_data_en.csv "$DEST"/volunteer_data_en.csv
