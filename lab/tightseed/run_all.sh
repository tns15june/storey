#!/bin/sh
# Rescore everything from caches. No API call is reached while data/lab/tightseed/sam and spark are intact.
cd "$(dirname "$0")/../.." || exit 1
export LAB_NAME=tightseed LAB_BUDGET=6 LAB_TOTAL=75
.venv/bin/python -I lab/tightseed/held_score.py > data/lab/tightseed/held_score.log 2>&1
.venv/bin/python -I lab/tightseed/post3.py > data/lab/tightseed/post3.log 2>&1
.venv/bin/python -I lab/tightseed/fresh.py > data/lab/tightseed/fresh.log 2>&1
.venv/bin/python -I lab/tightseed/nosam_spark.py royce_east bme_k > data/lab/tightseed/nosam.log 2>&1
.venv/bin/python -I lab/tightseed/final.py > data/lab/tightseed/final.log 2>&1
.venv/bin/python -I lab/tightseed/final2.py > data/lab/tightseed/final2.log 2>&1
cat data/lab/tightseed/post3.log data/lab/tightseed/fresh.log data/lab/tightseed/nosam.log | cut -c1-300
