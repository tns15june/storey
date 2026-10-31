#!/usr/bin/env python3
"""Cost and size figures for a 100-sheet drawing set, from list prices. Calls no API.

Run: /Users/tsharma/Github/Meta/.venv/bin/python /Users/tsharma/Github/Meta/lab/gather_event/cost_model.py
Writes: data/lab/gather_event/cost_model.json
"""
import base64
import json
from pathlib import Path

import cv2

PLANS = Path("/Users/tsharma/Github/Meta/data/plans")
OUT = Path("/Users/tsharma/Github/Meta/data/lab/gather_event/cost_model.json")

# USD per 1M tokens, from docs/pricing-rate-limits.md, read 2026-10-31.
SPARK = dict(input=1.25, cached=0.15, output=4.25)
SPARK_CONTRIB = dict(input=0.10, cached=0.002, output=0.20)
WONDER = dict(input=4.00, cached=0.20, output=20.00)
SAM_PER_FRAME = 0.20 / 1000
SAM_MIN_FRAMES = 50
SAM_MAX_OBJECTS = 16

# Measured by the earlier probe: about 5,000 reasoning tokens per sheet.
REASONING_MEASURED = 5000
# Assumptions, not measurements. Visible JSON per sheet, and input tokens for a 2000 px sheet.
VISIBLE_ASSUMED = 1000
INPUT_LOW, INPUT_HIGH = 2000, 4000  # docs give about 1,400 tokens at 1280 px; 2000 px is unmeasured
TRIAGE_OUTPUT_ASSUMED = 500
WONDER_TURNS, WONDER_CONTEXT, WONDER_OUT = 30, 30000, 1500  # assumed agent run shape
WONDER_CACHED_SHARE = 0.9  # assumed


def tok(p, i, o, cached=0):
    return (p["input"] * i + p["cached"] * cached + p["output"] * o) / 1e6


def sam_request(frames):
    return max(frames, SAM_MIN_FRAMES) * SAM_PER_FRAME


def main():
    sheet_out = REASONING_MEASURED + VISIBLE_ASSUMED
    r = {
        "per_sheet_spark_low": tok(SPARK, INPUT_LOW, sheet_out),
        "per_sheet_spark_high": tok(SPARK, INPUT_HIGH, sheet_out),
        "set100_all_sheets_low": 100 * tok(SPARK, INPUT_LOW, sheet_out),
        "set100_all_sheets_high": 100 * tok(SPARK, INPUT_HIGH, sheet_out),
        "set100_contributor_high": 100 * tok(SPARK_CONTRIB, INPUT_HIGH, sheet_out),
        "set100_triage_then_20_full_high": 100 * tok(SPARK, INPUT_HIGH, TRIAGE_OUTPUT_ASSUMED)
                                           + 20 * tok(SPARK, INPUT_HIGH, sheet_out),
        "sam_bme_4sheets_x4": sam_request(16),
        "sam_20floors_x4_one_request": sam_request(80),
        "sam_20floors_x4_three_requests": 3 * sam_request(80),
        "sam_image_prompt_per_image": 2.50 / 1000,
    }
    cached = int(WONDER_CONTEXT * WONDER_CACHED_SHARE)
    r["wonder_run_30_turns"] = WONDER_TURNS * tok(WONDER, WONDER_CONTEXT - cached, WONDER_OUT, cached)
    r["wonder_run_no_cache"] = WONDER_TURNS * tok(WONDER, WONDER_CONTEXT, WONDER_OUT)
    r["full_review_low"] = r["set100_triage_then_20_full_high"] + r["sam_20floors_x4_three_requests"] + r["wonder_run_30_turns"]
    r["full_review_high"] = r["set100_all_sheets_high"] + r["sam_20floors_x4_three_requests"] + r["wonder_run_no_cache"]
    r["sets_from_148_usd_at_high"] = 148 / r["full_review_high"]
    r["latency_100_sheets_serial_min"] = 100 * 20 / 60  # 20 s per sheet was measured
    r["latency_100_sheets_10_parallel_min"] = 100 * 20 / 10 / 60  # shortcut: assumes linear scaling, untested

    assert abs(tok(SPARK, 1_000_000, 0) - 1.25) < 1e-9 and sam_request(1) == 0.01
    # Does a real scanned sheet fit a background request body (1 MiB) as base64?
    sheets = {}
    for p in sorted((PLANS / "bme_f").glob("*.jpg")):
        h, w = cv2.imread(str(p)).shape[:2]
        sheets[p.name] = dict(w=w, h=h, bytes=p.stat().st_size,
                              base64_bytes=len(base64.b64encode(p.read_bytes())),
                              fits_1mib=len(base64.b64encode(p.read_bytes())) < 1048576)
    r["sheets"] = sheets
    v = cv2.VideoCapture(str(PLANS / "floors.mp4"))
    r["floors_mp4"] = dict(bytes=(PLANS / "floors.mp4").stat().st_size,
                           frames=int(v.get(cv2.CAP_PROP_FRAME_COUNT)), fps=v.get(cv2.CAP_PROP_FPS),
                           w=int(v.get(cv2.CAP_PROP_FRAME_WIDTH)), h=int(v.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    v.release()
    OUT.write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
