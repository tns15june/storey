#!/bin/sh
# Rebuild everything in data/plans and data/lab/gather_data. Run from anywhere: sh lab/gather_data/run_all.sh
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/.venv/bin/python -I"
big() { mkdir -p "$ROOT/data/plans/$1"; curl -sS -L -C - -A "curl/8.4" --retry 5 -o "$ROOT/data/plans/$1/$2" "$3" || echo "curl stopped on $2, usually 416 because it is complete: check its size"; }
big cubicasa5k cubicasa5k.zip https://zenodo.org/api/records/2613548/files/cubicasa5k.zip/content
big swiss_dwellings swiss-dwellings-v3.0.0.zip https://zenodo.org/api/records/7788422/files/swiss-dwellings-v3.0.0.zip/content
big floorplancad test-00.tar.xz "https://drive.usercontent.google.com/download?id=1jxpYgxnLUbXEzMOsjaMPQFSuvmvHimiZ&confirm=t"
$PY "$ROOT/lab/gather_data/fetch.py"
$PY "$ROOT/lab/gather_data/commons.py"
$PY "$ROOT/lab/gather_data/inventory.py"
$PY "$ROOT/lab/gather_data/swiss_stats.py"
$PY "$ROOT/lab/gather_data/build_catalog.py"
