"""Runs a generated set through the real pipeline under a per-review cap, re-evaluates it free for two cities, and scores it against its truth.

    STOREY_DATA=<scratch>/reviews STOREY_CACHE=<scratch>/cache python -m eval.measure data/generated/3370_1_clean
"""
import functools
import json
import os
import shutil
import sys
from pathlib import Path

from storey import pipeline
from storey.assemble import assemble
from storey.ingest import ingest
from storey.meta import DATA, Meter

CAP_USD = 2.0
ROOT = Path(__file__).resolve().parent.parent
WORDS = {"resolved_looks": "Looks resolved", "open": "Still open", "unknown": "Cannot tell"}


def _run(folder, parent):
    path = DATA / folder.name
    upload = path / "upload/0000.pdf"
    if upload.exists() and upload.read_bytes() != (folder / "set.pdf").read_bytes():
        sys.exit(f"{path} holds another upload of {folder.name}, and a rerun would score that one. Move the folder or use a new STOREY_DATA.")
    if not (path / "sheets.json").exists():
        upload.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(folder / "set.pdf", upload)
        ingest([upload], path)
        pipeline.write_json(path / "stage_upload.json", {"name": folder.name, "parent": parent})
    pipeline.Meter = functools.partial(Meter, cap_usd=CAP_USD)
    pipeline.run(path)
    return path


def _raised(defect, finding, numbers):
    """A finding answers a defect when its check, sheet and title agree."""
    return finding["check"] == defect["check"] and numbers.get(finding["sheet"]) == defect["sheet"] and defect["phrase"] in finding["title"]


def score(truth, review):
    """(raised, missed, at another severity, findings with no planted cause), each a list of rows to print."""
    numbers = {s["id"]: s["number"] for s in review["sheets"]}
    found, missed, off, used = [], [], [], set()
    for d in truth["defects"]:
        hits = [f for f in review["findings"] if _raised(d, f, numbers)]
        used |= {f["key"] for f in hits}
        row = {"kind": d["kind"], "check": d["check"], "sheet": d["sheet"], "wanted": d["severity"], "got": [f["severity"] for f in hits], "title": [f["title"] for f in hits]}
        if not hits:
            missed.append(row)
        elif any(f["severity"] in d["severity"] for f in hits):
            found.append(row)
        else:
            off.append(row)
    other = [{"check": f["check"], "severity": f["severity"], "sheet": numbers.get(f["sheet"]), "title": f["title"], "measured": f["measured"]} for f in review["findings"] if f["key"] not in used]
    return found, missed, off, other


def comments(truth, review):
    """Each comment of a second review: the state truth expects against the one it got. A change no comment covers is listed apart, because it has no comment."""
    parent = json.loads((DATA / truth["parent"] / "review.json").read_text())
    planted = {d["kind"]: d for d in json.loads((ROOT / "data/generated" / truth["parent"] / "truth.json").read_text())["defects"]}
    numbers = {s["id"]: s["number"] for s in parent["sheets"]}
    got = {c["comment_id"]: WORDS[c["state"]] for c in review["recheck"]["comments"]}
    rows = []
    for c in truth["comments"]:
        if c["state"] != "New change":
            ids = [f["comment_id"] for f in parent["findings"] if _raised(planted[c["kind"]], f, numbers)]
            rows.append({"kind": c["kind"], "wanted": c["state"], "got": [got.get(i) for i in ids]})
    return rows, review["recheck"]["changes"], review["recheck"]["clouds"]


def main(folder):
    if "STOREY_DATA" not in os.environ or "STOREY_CACHE" not in os.environ:
        sys.exit("Set STOREY_DATA and STOREY_CACHE to a scratch folder, because a run writes reviews and a ledger there.")
    folder = Path(folder)
    truth = json.loads((folder / "truth.json").read_text())
    path = _run(folder, truth.get("parent"))
    out = {}
    for city in ("seattle", "houston"):   # houston last, so review.json stands at the pipeline's default
        out[city] = assemble(path, city)
    found, missed, off, other = score(truth, out["houston"])
    keep = lambda rows: {(f["check"], f["severity"], f["title"], f["measured"]) for f in rows}
    only = {city: sorted(keep(out[city]["findings"]) - keep(out[rival]["findings"])) for city, rival in (("houston", "seattle"), ("seattle", "houston"))}
    report = {"cost_usd": out["houston"]["cost"]["usd"], "minutes": round(out["houston"]["cost"]["seconds"] / 60, 2), "raised": found, "missed": missed, "other_severity": off, "no_planted_cause": other, "only_in": only, "open_items": [i["title"] for i in out["houston"]["open_items"]],
              "not_checked": out["houston"]["not_checked"], "matrix": [{"id": r["id"], "established": r["established"], "cells": {numbers: c["state"] for numbers, c in r["floors"].items()}} for r in out["houston"]["verticals"]]}
    if out["houston"].get("recheck"):
        report["comments"], report["changes"], report["clouds"] = comments(truth, out["houston"])
    (path / "score.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
