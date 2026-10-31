#!/bin/sh
# Rebuilds every number from cached Spark calls in data/lab/doors/spark. Uncached calls spend from the doors budget.
cd /Users/tsharma/Github/Meta
export LAB_NAME=doors LAB_BUDGET=6 LAB_TOTAL=60
P=.venv/bin/python
for s in ifc_truth vector_eval; do $P -I lab/doors/$s.py; done
$P lab/doors/spark_tiles.py F1
$P lab/doors/spark_tiles.py F2 t5low
for s in spark_admin raster_eval exits exits_more exits_hybrid exits_tiles fusion rooms cubicasa overlays review_fp capitol_width; do $P lab/doors/$s.py; done
