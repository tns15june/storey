#!/bin/sh
# One command for the whole experiment. Cached Spark and SAM results are reused, so a rerun spends nothing.
# The landmark truth in data/lab/align/truth/ is not regenerated here: it holds eye checks. To redo it run truth_pages.py, reread the pages, then truth_fix.py.
set -e
cd /Users/tsharma/Github/Meta
export LAB_NAME=align LAB_BUDGET=3 LAB_TOTAL=60
P=".venv/bin/python -I"
$P lab/align/render.py
$P lab/align/reg.py
$P lab/align/sheet_read.py all low
$P lab/align/joint_redo.py
$P lab/align/pdf_order.py
$P lab/align/score_order.py
$P lab/align/eval_align.py
$P lab/align/overlay.py
$P lab/align/stack_sam.py
$P lab/align/sam_landmarks.py
$P lab/align/robustness.py
$P lab/align/synth.py
$P lab/align/summary.py
