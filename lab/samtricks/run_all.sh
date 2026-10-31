#!/bin/zsh
# Rebuilds every samtricks score from the frozen truth and the saved results. It calls the API only for a result missing from disk.
cd /Users/tsharma/Github/Meta
export LAB_NAME=samtricks LAB_BUDGET=3 LAB_TOTAL=75
PY=.venv/bin/python
$PY lab/samtricks/item1.py dev
$PY lab/samtricks/item1.py score
$PY lab/samtricks/item2.py score
$PY lab/samtricks/item3.py
$PY -I lab/samtricks/item4.py score
$PY lab/samtricks/item5.py score
$PY lab/samtricks/item6.py
$PY lab/samtricks/item6b.py
$PY -I lab/samtricks/item7.py single
