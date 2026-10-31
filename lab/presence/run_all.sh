#!/bin/sh
# One command for the whole experiment. Cached Spark runs and evidence files are reused, so a rerun spends nothing.
# Truth is not regenerated here: lab/presence/truth.json is hashed in data/lab/presence/truth_meta.json. Its source is truth_src.py.
# data/lab/presence/cells/*.json are the record of evidence: crop.py reuses a nearby cached answer, so rebuilding them can change a cell.
set -e
cd /Users/tsharma/Github/Meta
export LAB_NAME=presence LAB_BUDGET=6 LAB_TOTAL=75
P=".venv/bin/python -I"
$P lab/presence/pres.py
$P lab/presence/cells.py
$P lab/presence/score.py
$P lab/presence/rule.py
$P lab/presence/sheet_read.py
[ -f data/lab/presence/align/bme_f.json ] || $P lab/presence/align.py
$P lab/presence/compare.py bme_f habs_stelizabeths_center nibs_clinic mml_capitol
$P lab/presence/compare.py --planted dev
$P lab/presence/compare.py bme_k habs_va_bldg1 wfu_huffman schenley habs_stelizabeths_atkins
$P lab/presence/compare.py --planted heldout
for k in bme_f habs_stelizabeths_center nibs_clinic mml_capitol bme_f__all habs_stelizabeths_center__all \
  bme_k habs_va_bldg1 wfu_huffman schenley habs_stelizabeths_atkins \
  bme_k__middle bme_k__discharge bme_k__top habs_va_bldg1__middle habs_va_bldg1__discharge habs_va_bldg1__top \
  wfu_huffman__middle wfu_huffman__discharge wfu_huffman__top schenley__middle schenley__discharge schenley__top \
  habs_stelizabeths_atkins__middle habs_stelizabeths_atkins__discharge habs_stelizabeths_atkins__top; do
  [ -f data/lab/presence/cells/$k.json ] || $P lab/presence/run_cells.py all $k
done
$P lab/presence/evaluate.py dev
$P lab/presence/evaluate.py heldout
$P lab/presence/report.py dev
$P lab/presence/report.py heldout
