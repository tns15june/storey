#!/bin/sh
# Re-renders everything from caches in data/lab/seed. Delete data/lab/seed/spark and data/lab/seed/sam to pay for a fresh run, about 2.75 USD.
set -e
cd "$(dirname "$0")/../.." || exit 1
export LAB_NAME=seed LAB_BUDGET=3 LAB_TOTAL=60
PY=".venv/bin/python -I"
$PY lab/seed/check.py
$PY lab/seed/prep.py
$PY lab/seed/prep_panel.py nibs_clinic mml_capitol mml_baxter
for v in plan plan_low doc; do $PY lab/seed/seed_spark.py $v; done
$PY lab/seed/seed_spark.py doc nibs_clinic mml_capitol mml_baxter
$PY lab/seed/seed_spark.py lift habs_stelizabeths_center habs_stelizabeths_atkins nibs_clinic bme_f
$PY lab/seed/sam_track.py matrix
$PY lab/seed/union.py doc
$PY lab/seed/union.py doc nibs_clinic mml_capitol mml_baxter
$PY lab/seed/union.py plan_low
$PY lab/seed/lift.py
$PY lab/seed/rebox.py
$PY lab/seed/score_sam.py > /dev/null
$PY lab/seed/extras.py
$PY lab/seed/final.py
