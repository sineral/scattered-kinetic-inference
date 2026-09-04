#!/usr/bin/env bash
# Pack parquet files for a Zenodo upload. Run from anywhere:
#   bash data/pack_zenodo.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/scattered-kinetic-data.zip"
rm -f "$OUT"
zip -0 -r "$OUT" \
  data/README \
  data/default/README \
  data/default/train.parquet \
  data/default/val.parquet \
  data/default/test.parquet \
  data/combined/README \
  data/combined/train.parquet \
  data/combined/val.parquet \
  data/combined/test.parquet
ls -lh "$OUT"
echo "Upload $OUT to Zenodo. Current DOI: https://doi.org/10.5281/zenodo.21991341 (update README.md and data/README if a new version DOI is minted)."
