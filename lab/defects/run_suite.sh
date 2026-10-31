#!/bin/sh
cd /Users/tsharma/Github/Meta
LAB_NAME=defects LAB_BUDGET=6 LAB_TOTAL=60 .venv/bin/python lab/defects/suite.py all 2>&1 | grep -v -E "fitz|warnings.warn" > data/lab/defects/suite.log
