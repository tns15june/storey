#!/bin/sh
# Reproduces every number. Spark and SAM results are cached under data/lab/walls, so a rerun spends nothing.
cd /Users/tsharma/Github/Meta
PY=".venv/bin/python -I -W ignore"
export LAB_NAME=walls LAB_BUDGET=6 LAB_TOTAL=60
(cd lab/walls && ../../.venv/bin/python -I plan.py && ../../.venv/bin/python -I dbl.py) || exit 1
$PY lab/walls/classical.py test 120 '{"walls":{"dbl_k":0.08}}' data/lab/walls/classical120_thick.json
$PY lab/walls/classical.py test 120 '{"block":{}}' data/lab/walls/classical120_block.json
$PY lab/walls/oracle.py test 120
.venv/bin/python -W ignore lab/walls/sam_rooms.py 22
$PY lab/walls/eval_models.py
.venv/bin/python -W ignore lab/walls/tile_doors.py 22
$PY lab/walls/eval_tiles.py
$PY lab/walls/clinic4.py 0
cp data/lab/walls/clinic.json data/lab/walls/clinic_pen_only.json
$PY lab/walls/clinic4.py 0 ring
$PY lab/walls/truepath.py
$PY lab/walls/clinic4.py 0 ring 0 no2as1
cp data/lab/walls/clinic.json data/lab/walls/clinic_no2as1.json
$PY lab/walls/scans.py
$PY lab/walls/handpaths.py
.venv/bin/python -W ignore lab/walls/pens.py
