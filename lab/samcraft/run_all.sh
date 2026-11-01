#!/bin/sh
# Rebuilds every number from the call cache in data/lab/samcraft/cache. Cached calls cost nothing; calls that failed the first time are retried.
cd "$(dirname "$0")/../.." || exit 1
export LAB_NAME=samcraft LAB_BUDGET=4 LAB_TOTAL=60
PY=.venv/bin/python
$PY lab/samcraft/check.py
$PY lab/samcraft/data.py
[ -f data/lab/samcraft/swiss_subset.json ] || $PY lab/samcraft/swiss_pick.py  # about 100 s, no API
$PY lab/samcraft/concept.py
$PY lab/samcraft/vocab.py
$PY lab/samcraft/rooms.py grid
$PY lab/samcraft/recipe.py
$PY lab/samcraft/snap.py
$PY lab/samcraft/pole.py
$PY lab/samcraft/viz.py
$PY lab/samcraft/stairs_doors.py
$PY lab/samcraft/track_swiss.py
$PY lab/samcraft/erase.py
$PY lab/samcraft/track2.py
$PY lab/samcraft/real.py sheets
$PY lab/samcraft/real.py bme 2000x1466 445,1050,830,1095
