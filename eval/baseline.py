"""The comparison the video needs: Meta's model asked directly, with no Storey code between. One muse-spark-1.3 call per set holds every sheet, with our own plain prompt, which names none of Storey's checks.

    python -m eval.baseline 710_1_mixed <review folder> [--cap 0.5]

The sheets are the 2000 px copies in a review folder that eval.run made, so the Richmond page rule holds here too.
"""
import argparse
import json
import time
from pathlib import Path

from storey.assemble import _image
from storey.findings import _masked
from storey.meta import BudgetReached, Meter, _cache_key, spark
from storey.pipeline import read_json, write_json

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/eval/baseline"
MODEL = "muse-spark-1.3"
EFFORT = "medium"
MAX_IMAGES = 50   # what Meta takes in one request
CAP_USD = 0.5
PROMPT = ("These images are the sheets of one building permit drawing set, in page order. Review the set as a city plan examiner would before writing a correction letter, "
          "and list each problem you find. For each problem, give the sheet number printed on the sheet, or the page number if none is printed, and state the problem in one or two sentences.")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["problems"], "properties": {"problems": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["sheet", "statement"], "properties": {"sheet": {"type": "string"}, "statement": {"type": "string"}}}}}}


def request(review_dir):
    """(the sheet images sent, the number of sheets, the request's key). The key changes with any image, the prompt, the schema, the model or the effort, so a record whose key differs saw other drawings."""
    review_dir = Path(review_dir)
    sheets = read_json(review_dir / "sheets.json") or []
    paths = [review_dir / _image(review_dir, s)[0] for s in sheets[:MAX_IMAGES]]
    return paths, len(sheets), _cache_key(MODEL, EFFORT, None, PROMPT, SCHEMA, [p.read_bytes() for p in paths])


def ask(name, review_dir, meter):
    """The baseline's record for one set. A record made from the same request is returned as it stands, so its cost stays the cost of the call that made it."""
    paths, total, key = request(review_dir)
    path = OUT / f"{name}.json"
    old = read_json(path)
    if old.get("request") == key and not old.get("failed"):
        return old
    started, spent = time.monotonic(), meter.spent
    problems, failed = [], None
    try:
        reply = spark(PROMPT, paths, meter=meter, schema=SCHEMA, effort=EFFORT, model=MODEL, fresh=True)   # a cached reply costs nothing, and the record must carry what its call cost
        problems = [{"id": f"B{n:02d}", "sheet": str(p["sheet"]), "statement": str(p["statement"])} for n, p in enumerate(reply["problems"], 1)]
    except BudgetReached as e:
        failed = str(e)
    except (ValueError, KeyError, TypeError):
        failed = "The model's reply could not be read, so the baseline has no list for this set."
    record = {"set": name, "model": MODEL, "effort": EFFORT, "sheets_sent": len(paths), "sheets_total": total, "request": key, "failed": failed,
              "problems": _masked(problems), "usd": round(meter.spent - spent, 4), "minutes": round((time.monotonic() - started) / 60, 2)}   # model text can call the set approved
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(path, record)
    return record


def main():
    ap = argparse.ArgumentParser(description="Ask Meta's model directly for a set's problems.")
    ap.add_argument("set")
    ap.add_argument("review", type=Path, help="the set's review folder from eval.run")
    ap.add_argument("--cap", type=float, default=CAP_USD, help="USD for this call")
    a = ap.parse_args()
    print(json.dumps(ask(a.set, a.review, Meter(cap_usd=a.cap, ledger=OUT / "ledger.jsonl")), indent=1))


if __name__ == "__main__":
    main()
