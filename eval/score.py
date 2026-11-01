"""Scores one set in the spec's order: wrong passes, findings judged real, false findings, cost and minutes, with the baseline beside each.

A wrong pass is a stair cell shown Found where truth says the stair is absent or erased, or a second-review comment shown Looks resolved where truth holds it open. Storey has no pass state for any other check, so a missed defect there is a miss, listed by name.
Generated sets and plants are matched in code. Audit truth, the Richmond record and every baseline need judgement: this writes a match sheet of claims and truth items to data/eval/sheets/<set>.<method>.json and reads the judge's mapping back from eval/matches/<set>.<method>.json. A person or the parent agent judges, never this code.

Mapping file:
    {"set": str, "method": "Storey" or "baseline", "judge": who judged and what they saw, "sheet_sha256": the sha256 the match sheet carries,
     "matches": [{"id": a claim's id, "truth": [ids of the truth items it states, empty when it states none], "note": optional}]}
Each claim of the match sheet is ruled on exactly once, and every truth id is on the sheet.
Findings judged real count Fix and Check items. Cannot tell items are not claims, so they are counted apart.
A plant's truth covers stair presence alone and the Richmond record one examiner's letter, so there a claim that states no truth item is not judged, for either method, rather than false.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from storey.findings import LEGEND, SITE_TITLE
from storey.pipeline import read_json, write_json

from . import measure, run

ROOT = Path(__file__).resolve().parent.parent
MATCHES = ROOT / "eval/matches"
SHEETS = ROOT / "data/eval/sheets"
CITY = "houston"         # the pipeline's default, so a baseline is judged against the check 11 rows a default review is
NEAR = 50                # px on the 2000 px review sheet: a stair row's place moved up to 49 px between fresh runs
STAIR_CHECKS = (9, 10)   # a planted defect under one of these takes the stair off its floor
REAL = ("real", "likely")
TOO_FEW = "too few stairs"
STATES = {**LEGEND, "found": "Found", "not_followed": "Not followed", None: "No stair row at this place"}
POST_HOC = {"mml_capitol": re.compile(r"^(No site plan|Window \S+ has more than one size|Door \S+ has a material in the swing column|Dimension string)")}   # rules written after this set's audit
LATE = "Raised only by a rule written after this set's audit, so it counts in the post hoc row."
PARTIAL = ("plant", "record")   # truth that covers part of what a review may raise
HOW = ("For each claim, list the ids of the truth items it states: the same problem at the same place. An empty list means it states none of them. "
       "Write the mapping to eval/matches/<set>.<method>.json in the format eval/score.py describes.")


def read_truth(path, name):
    raw = json.loads(Path(path).read_text())
    kind = "generated" if "defects" in raw else "plant" if "cells" in raw else "record" if "decision_rule" in raw else "audit"
    source = {"generated": "swiss_dwellings", "plant": raw.get("source")}.get(kind, name)   # the catalog entry that decides whether eval.json may show this set
    return {"name": name, "kind": kind, "source": source, "path": str(path), "sha256": run._sha(path), "raw": raw}


def load(name):
    path = run.truth_path(name)
    if path is None:
        raise ValueError(f"No truth is written for {name}.")
    return read_truth(path, name)


def _expected(raw, city):
    """What a review of a generated set should raise: each planted defect, too few stairs on a floor above discharge, and a missing site plan.
    A floor where one stair depends on conditions is no truth item: its answer is a Cannot tell item, which is no claim."""
    rows = [d | {"id": d["kind"], "statement": d["wording"]} for d in raw["defects"]]
    rows += [{"id": f"check 11 on {f['floor']}", "check": 11, "floor": f["floor"], "outcome": f["outcome"], "severity": ["check"],
              "statement": f"{f['outcome'].capitalize()} on {f['floor']} under the {city} rule rows."} for x in raw.get("check_11", ()) if x["jurisdiction"] == city for f in x["floors"] if f["outcome"] == TOO_FEW]
    if not any(SITE_TITLE.search(s["title"]) for s in raw["sheets"]):
        rows.append({"id": "no site plan", "kind": "no site plan", "check": 1, "sheet": None, "phrase": "No site plan", "severity": ["check"], "statement": "The set holds no site plan."})
    return rows


def _changes(raw):
    """The changes a second review's truth plants with no cloud round them, which only a comparison with the first set can find."""
    return [c | {"id": c["kind"], "statement": c["wording"]} for c in raw.get("comments", ()) if c["state"] == "New change"]


def _answers(row, f, numbers):
    if "outcome" not in row:
        return measure._raised(row, f, numbers)
    return f["check"] == 11 and f["title"] == f"Too few stairs on {row['floor']}"


def items(truth, city=CITY):
    """The truth items a judge maps claims to: id, sheets, statement and status."""
    raw, kind = truth["raw"], truth["kind"]
    if kind == "generated":
        return [{"id": r["id"], "sheets": [r["sheet"]] if r.get("sheet") else [], "statement": r["statement"], "status": "real"} for r in _expected(raw, city)]
    if kind == "plant":
        return [{"id": _cell(c), "sheets": [raw["sheets"][c["index"]]], "statement": f"The stair {c['object']} is erased on floor {c['floor']}, the {c['kind']} plant.", "status": "real"} for c in raw["cells"]]
    if kind == "record":
        return [{"id": c["id"], "sheets": [raw["sheet"]["id"]], "statement": c["text"], "status": "real"} for c in raw["comments"] if c["sheet_only"]]
    return [{k: t[k] for k in ("id", "sheets", "statement", "status")} for t in raw["items"]]


def _cell(c):
    return f"{c['object']} on floor {c['floor']}"


def _near(review, i, box):
    """(row, cell) for each stair row whose place on sheet `i` lies within NEAR of `box`. Both are in the review's 2000 px sheet."""
    x1, y1, x2, y2 = box
    return [(row, row["floors"][i]) for row in review.get("verticals") or () if (p := (row["floors"].get(i) or {}).get("point"))
            and x1 - NEAR <= p[0] <= x2 + NEAR and y1 - NEAR <= p[1] <= y2 + NEAR]


def _nearest(stairs, i, point):
    """The stair truth draws on sheet `i` nearest `point`, if one lies within NEAR. The nearest wins, because on habs_stelizabeths_center exterior stairs touch the main ones."""
    gaps = {name: max(b[0] - point[0], point[0] - b[2], b[1] - point[1], point[1] - b[3], 0) for name, boxes in stairs.items() if (b := boxes.get(i))}
    name = min(gaps, key=gaps.get, default=None)
    return name if name is not None and gaps[name] <= NEAR else None


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _claims(review, numbers=None):
    numbers = numbers or {s["id"]: s["number"] for s in review["sheets"]}
    return [{"id": f["id"], "check": f["check"], "severity": f["severity"], "sheet": numbers.get(f["sheet"]), "statement": " ".join(filter(None, (f"{f['title']}.", f.get("measured"))))}
            for f in review["findings"]]


def _exact(raw, review):
    """Planted defects, check 11 and the site plan, matched by check, page and title at a severity truth allows, and on a second review each change no cloud marks.
    Found on a stair truth took off a floor is a wrong pass."""
    pages = [s["number"] for s in raw["sheets"]]
    if len(pages) != len(review["sheets"]):
        raise ValueError(f"The review holds {len(review['sheets'])} sheets and the truth {len(pages)}, so it reviewed another set.")
    numbers = {s["id"]: pages[int(s["id"])] for s in review["sheets"]}   # by page, so a sheet number the read got wrong does not move a finding off its page
    rows = _expected(raw, review["project"]["jurisdiction"])
    found, misses, real, used = 0, [], set(), set()
    for row in rows:
        hits = [f for f in review["findings"] if _answers(row, f, numbers)]
        good = {f["id"] for f in hits if f["severity"] in row["severity"] and f["severity"] != "unknown"}   # a Cannot tell item is no claim, whatever truth allows
        used |= {f["id"] for f in hits}
        real |= good
        found += bool(good)
        if not good:
            misses.append({"item": row["id"], "statement": row["statement"],
                           "why": f"Raised as {' and '.join(f['severity'] for f in hits)}, where truth allows {' or '.join(row['severity'])}." if hits else "Not raised."})
    k = raw["px_per_pt"]["review.json"]
    changes, listed = _changes(raw), (review.get("recheck") or {}).get("changes") or ()
    for c in changes:
        if any(x["sheet"] == str(pages.index(c["sheet"])) and _overlap(x["box"], [v * k for v in c["box"]]) for x in listed):
            found += 1
        else:
            misses.append({"item": c["id"], "statement": c["statement"], "why": "No change is listed there."})
    wrong = [{"item": d["kind"], "sheet": d["sheet"], "stair": row["id"], "state": "Found"} for d in raw["defects"] if d["check"] in STAIR_CHECKS
             for row, cell in _near(review, str(pages.index(d["sheet"])), [v * k for v in d["box"]]) if cell["state"] == "found"]
    claims = _claims(review, numbers)
    false = [c for c in claims if c["id"] not in used and c["severity"] != "unknown"]
    return {"wrong passes": len(wrong), "findings judged real": len(real), "false findings": len(false), "truth items found": found, "out of": len(rows) + len(changes),
            "cannot tell": sum(c["severity"] == "unknown" for c in claims), "wrong pass list": wrong, "misses": misses, "false list": false}


def comments(raw, review, parent, parent_raw):
    """(each comment truth gives a state, with the states Storey showed for it; the wrong passes among them: Looks resolved where truth holds the comment open)."""
    planted = {d["kind"]: d for d in parent_raw["defects"]}
    pages = [s["number"] for s in parent_raw["sheets"]]
    numbers = {s["id"]: pages[int(s["id"])] for s in parent["sheets"]}
    got = {c["comment_id"]: c["state"] for c in (review.get("recheck") or {}).get("comments", ())}
    rows = [{"item": c["kind"], "wanted": c["state"], "got": [measure.WORDS.get(got.get(f["id"]), "No comment") for f in parent["findings"] if measure._raised(planted[c["kind"]], f, numbers)]}
            for c in raw.get("comments", ()) if c["state"] != "New change"]
    return rows, [r for r in rows if r["wanted"] == "Still open" and "Looks resolved" in r["got"]]


def _planted(raw, review):
    """A stair row's cell on each sheet belongs to the stair truth draws nearest its place there. Found where that stair was erased, or on a floor where the row's stair is not drawn at all, is a wrong pass.
    A stair finding on an erased cell's row and sheet raises it. Truth holds stair presence alone, so no finding is judged false."""
    stairs, erased = raw["stairs"], {(c["object"], str(c["index"])): c for c in raw["cells"]}
    seen, wrong = {key: [] for key in erased}, []   # each erased cell's (row, cell) pairs
    for row in review.get("verticals") or ():
        near = {i: _nearest(stairs, i, cell["point"]) for i, cell in row["floors"].items() if cell.get("point")}
        own = Counter(name for name in near.values() if name).most_common(1)
        if not own:
            continue   # truth draws no stair at this row's place on any floor
        stair = own[0][0]
        for i, name in near.items():
            cell = row["floors"][i]
            if (name, i) in erased:
                seen[name, i].append((row, cell))
            if cell["state"] == "found" and ((name, i) in erased or (name is None and i not in stairs[stair])):
                wrong.append({"item": _cell(erased[name, i]) if name else f"{stair} on floor {raw['floors'][int(i)]}", "sheet": raw["sheets"][int(i)], "stair": row["id"], "state": "Found",
                              "truth": "erased" if name else "not drawn on this floor"})
    claims, states, misses, real = _claims(review), Counter(), [], set()
    for c in raw["cells"]:
        i = str(c["index"])
        pairs = seen[c["object"], i]
        shown = [cell["state"] for _, cell in pairs]
        state = "found" if "found" in shown else shown[0] if shown else None
        states[STATES.get(state, state)] += 1
        hits = {f["id"] for f in review["findings"] if f["check"] in STAIR_CHECKS and f["sheet"] == i and f["severity"] != "unknown" and f.get("vertical") in {row["id"] for row, _ in pairs}}
        real |= hits
        if not hits:
            why = "The stair's top floor, where an erasure reads like a stair that ends, the stated limit." if c["expected"] is None else f"It reads {STATES.get(state, state)}, and no finding names it."
            misses.append({"item": _cell(c), "statement": f"The stair {c['object']} is erased on floor {c['floor']}, the {c['kind']} plant.", "why": why})
    counted = [x for x in claims if x["severity"] != "unknown"]
    return {"wrong passes": len(wrong), "findings judged real": len(real), "false findings": None, "not judged": len(counted) - len(real), "truth items found": len(raw["cells"]) - len(misses),
            "out of": len(raw["cells"]), "cannot tell": len(claims) - len(counted), "states": dict(states), "wrong pass list": wrong, "misses": misses}


def match_sheet(name, method, claims, truth_items):
    sheet = {"set": name, "method": method, "how": HOW, "claims": [{k: c.get(k) for k in ("id", "sheet", "severity", "statement")} for c in claims], "truth": truth_items}
    return sheet | {"sha256": hashlib.sha256(json.dumps(sheet, sort_keys=True).encode()).hexdigest()}


def read_matches(path, sheet):
    """The judge's mapping as {claim id: [truth ids]}. Raises ValueError unless it answers this match sheet, rules on each claim once and names only truth items on the sheet."""
    m = json.loads(Path(path).read_text())
    rulings = m.get("matches") or []
    known = {t["id"] for t in sheet["truth"]}
    lists = all(isinstance(x.get("truth"), list) for x in rulings)
    named = {t for x in rulings if isinstance(x.get("truth"), list) for t in x["truth"]}
    problems = [text for bad, text in (
        (m.get("sheet_sha256") != sheet["sha256"], "It answers another match sheet, so the claims changed after it was written."),
        (Counter(x.get("id") for x in rulings) != Counter(c["id"] for c in sheet["claims"]), "Each claim needs exactly one ruling."),
        (not lists, "Each ruling needs a list of truth ids, empty when it states none."),
        (bool(named - known), f"Truth ids not on the sheet: {', '.join(sorted(map(str, named - known)))}.")) if bad]
    if problems:
        raise ValueError(f"{path}: {' '.join(problems)}")
    return {x["id"]: x["truth"] for x in rulings}


def _judged(truth, method, claims, truth_items):
    name = truth["name"]
    sheet = match_sheet(name, method, claims, truth_items)
    SHEETS.mkdir(parents=True, exist_ok=True)
    write_json(SHEETS / f"{name}.{method}.json", sheet)
    path = MATCHES / f"{name}.{method}.json"
    if claims and not path.exists():
        return {"waiting": f"A judge's mapping at {path}, for the match sheet at {SHEETS / f'{name}.{method}.json'}."}
    matches = read_matches(path, sheet) if claims else {}   # nothing listed leaves nothing to judge
    good = {t["id"]: t for t in truth_items if t["status"] in REAL}
    counted = [c for c in claims if c.get("severity") != "unknown"]
    late = [c for c in counted if method == "Storey" and name in POST_HOC and POST_HOC[name].search(c["statement"])]

    def tally(cs):
        real = [c for c in cs if set(matches[c["id"]]) & set(good)]
        return real, [c for c in cs if c not in real], {t for c in real for t in matches[c["id"]] if t in good}

    real, false, found = tally([c for c in counted if c not in late])
    late_real, late_false, late_found = tally(late)
    partial = truth["kind"] in PARTIAL
    out = {"findings judged real": len(real), "false findings": None if partial else len(false), "not judged": len(false) if partial else None, "truth items found": len(found), "out of": len(good),
           "cannot tell": len(claims) - len(counted), "found by status": dict(Counter(good[t]["status"] for t in found)), "false list": [] if partial else false,
           "misses": [{"item": t, "sheets": good[t]["sheets"], "statement": good[t]["statement"]} | ({"why": LATE} if t in late_found else {}) for t in good if t not in found]}
    if late:
        out["post hoc"] = {"findings judged real": len(late_real), "false findings": len(late_false), "truth items found": len(late_found - found)}
    return out


def storey(truth, review_dir):
    """Storey's score on one set from its review folder, whose review.json is what the screen showed."""
    review_dir = Path(review_dir)
    review = read_json(review_dir / "review.json")
    if not review:
        raise ValueError(f"{review_dir} holds no review.json, so the run did not finish.")
    raw, kind = truth["raw"], truth["kind"]
    if kind == "generated":
        out = _exact(raw, review)
        if raw.get("parent"):
            parent = read_json(review_dir.parent / raw["parent"] / "review.json")
            if not parent:
                raise ValueError(f"A second review is scored against its first, and {raw['parent']} has no review beside {review_dir}.")
            rows, wrong = comments(raw, review, parent, load(raw["parent"])["raw"])
            out |= {"comments": rows, "wrong passes": out["wrong passes"] + len(wrong), "wrong pass list": out["wrong pass list"] + wrong}
    elif kind == "plant":
        out = _planted(raw, review)
    else:
        out = _judged(truth, "Storey", _claims(review), items(truth)) | {"wrong passes": None}   # this truth holds no stair presence and no comment states
    record = read_json(review_dir / "eval_run.json")
    fresh = record.get("fresh") is True   # a run eval.run began in a new folder on a cold cache. Any other spent less than a first run would, or left a retried stage's earlier spend out
    return out | {"cost USD": record["usd"] if fresh else None, "minutes": record["minutes"] if fresh else None}


def baseline(truth, record):
    """The baseline's score from its record. It shows no stair matrix and no comment states, so it has no wrong pass to count, and what it leaves out is a miss."""
    cost = {"wrong passes": None, "cost USD": record.get("usd"), "minutes": record.get("minutes")}
    if record.get("failed"):
        return cost | {"failed": record["failed"]}
    return _judged(truth, "baseline", record["problems"], items(truth)) | cost
