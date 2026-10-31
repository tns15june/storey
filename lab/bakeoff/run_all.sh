#!/bin/sh
# Finished calls are cached in data/lab/bakeoff/runs.jsonl, so this spends only on what is missing.
# report.py alone recomputes every number with no API call.
cd /Users/tsharma/Github/Meta
export LAB_NAME=bakeoff LAB_BUDGET=${LAB_BUDGET:-8} LAB_TOTAL=${LAB_TOTAL:-60}
PY=".venv/bin/python -I"
$PY lab/bakeoff/truth_meta.py
$PY lab/bakeoff/e1_meta.py sweep
$PY lab/bakeoff/e1_meta.py wide s13-low-2k s13-min-2k
$PY lab/bakeoff/e2_tasks.py run
$PY lab/bakeoff/e3_pdf_floors.py cache
$PY lab/bakeoff/e3_pdf_floors.py pdf
$PY lab/bakeoff/e3_pdf_floors.py floors
$PY lab/bakeoff/report.py
