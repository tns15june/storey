#!/bin/sh
# Reproduce the modern CAD experiment. API spend about 0.70 USD.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
export LAB_NAME=modern LAB_BUDGET=4 LAB_TOTAL=60
$PY -I lab/modern/vec.py && $PY -I lab/modern/render.py && $PY -I lab/modern/stack.py && $PY -I lab/modern/stair_eval.py \
  && $PY -I lab/modern/detect.py && $PY -I lab/modern/measure.py && $PY -I lab/modern/halls.py && $PY -I lab/modern/doors.py && $PY -I lab/modern/corridor.py \
  && $PY lab/modern/clinic.py && $PY lab/modern/track.py && $PY lab/modern/track2.py && $PY lab/modern/track3.py
