#!/bin/sh
# Reproduces every number. Tracks and runs are cached in data/lab/defects, so a rerun is free; delete tracks/ and runs*/ to pay about 5.80 USD again.
cd "$(dirname "$0")/../.." || exit 1
export LAB_NAME=defects LAB_BUDGET=6 LAB_TOTAL=60
P=.venv/bin/python
$P lab/defects/plant.py && $P lab/defects/rules.py
[ -f data/lab/defects/stack/bme_f/0.png ] || $P lab/defects/prep.py
[ -f data/lab/defects/seeds_raw.json ] || $P lab/defects/propose.py
$P lab/defects/clean.py
$P lab/defects/suite.py all
$P lab/defects/jitter.py
[ -f data/lab/defects/revision.json ] || $P lab/defects/revision.py
[ -f data/lab/defects/spark_checks.json ] || $P lab/defects/spark_checks.py
[ -f data/lab/defects/spark_only.json ] || $P lab/defects/spark_only.py
$P lab/defects/corridors.py
$P lab/defects/hybrid.py replay
$P lab/defects/hybrid.py fresh
$P -I lab/defects/rev_eval.py
$P lab/defects/final_stats.py
