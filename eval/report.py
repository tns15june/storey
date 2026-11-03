"""Writes data/eval/report.json, everything, and app/web/eval.json, the How well it works page's copy.

    python -m eval.report                         # every run eval.run recorded in data/eval/runs.json
    python -m eval.report mml_capitol=<review folder> 710_1_mixed=<review folder>

eval.json is public. It lists a set's misses, false findings and wrong passes only when the set's catalog entry says public_demo yes. Every other set keeps its counts and nothing else.
eval.json is not written while any set waits for the judge or has no baseline scored on the drawings its run saw: none recorded, a call that failed, or one asked on other drawings.
That set's misses would be missing from the page, and its blank baseline cells would shrink the page's baseline sums and what they are out of.
"""
import json
import sys
from pathlib import Path

from storey.pipeline import read_json, write_json

from . import baseline, run, score

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "data/eval/report.json"
PUBLIC = ROOT / "app/web/eval.json"
CATALOGS = (ROOT / "data/plans/catalog.json", ROOT / "data/plans/catalog_r2.json")
NEVER = ("va_trb", "bcoe")   # never public, whatever a catalog says
SCORES = ("wrong passes", "found where truth places no stair", "findings judged real", "false findings", "not judged", "cannot tell", "truth items found", "present cells", "cost USD", "minutes")
LISTS = {"wrong passes": "wrong pass list", "found where truth places no stair": "no stair list", "misses": "misses", "false findings": "false list"}
ABOUT = ("Truth was written by the assistant, one labeller, not a licensed examiner. Generated sets and planted erasures carry their truth from how they were made. "
         "Real sets were audited sheet by sheet by two auditors working blind to Storey, and one adjudicator ruled on each item. Richmond is the exception: its truth is the examiner's letter on its one sheet, not an audit, "
         "so it is scored on which of the letter's 10 sheet-only comments were raised, and no finding there is judged false. "
         "A wrong pass is a stair shown Found where truth says it is absent or was erased, or a comment shown Looks resolved where truth holds it open. Storey shows no pass for any other check, so a defect it does not raise is a miss, listed by name. "
         "On army_bt_bcof and queens_royce, stair truth gives each stair's place on every floor, drawn or absent. Their wrong passes are counted over the cells where truth says a stair is absent, and on their planted copies over the erased cells too, "
         "and the cells where truth says a stair is drawn are counted by the state Storey showed. A stair that ends at a floor is no defect, so an absent cell is never a miss. "
         "A stair row Storey shows where truth places no stair on any floor, drawn or absent, is counted apart: each Found cell there is listed as found where truth places no stair, not as a wrong pass, because truth leaves out some stair-like marks on purpose, such as the walk-off grilles on army_bt_bcof. queens_royce has stair truth alone, so no finding there is judged. "
         "Findings judged real count Fix and Check items. Cannot tell items are not claims and are counted apart. On generated sets, a finding with no planted cause counts as false. "
         "On planted erasures, truth covers stair presence alone, so for both methods a finding that names no erased stair is counted as not judged. "
         "A second review is also scored on a change planted with no cloud round it, which only a comparison with the first set can show, so the baseline, shown one set, is not scored on that change. "
         "The baseline is Meta's muse-spark-1.3 asked once per set, every sheet in one request, with a plain prompt that names none of Storey's checks. It shows no stair matrix and no comment states, so it has no wrong passes to count. "
         "Cost and minutes come only from a run eval.run began in a new folder on a cold model cache, and are blank otherwise. A second review shares its first review's cache, as it would on the server. "
         "Each number carries its status. Development numbers come from sets that earlier experiments touched. Held-out numbers come only from army_bt_bcof, queens_royce and their planted copies, "
         "from each one's one run on code, scoring and truth frozen by hash. "
         "A run with no truth hash recorded, or whose truth changed after it, is post hoc. On mml_capitol, findings from rules written after its audit are counted apart as post hoc. "
         "That split goes by rule, so a later change to how an older rule reads the drawing stays in the development row.")
QUOTED = [{"figure": "Declared values against the drawing on 13 real sets: 35 of 137 flags raised at frozen settings were real or likely, read by one unblinded reader.", "status": "frozen", "source": "docs/research/build-log.md, Round 2"},
          {"figure": "The stair rule on 15 planted erasures in the lab, in the states the product shows: 12 Check, 1 Missing, 2 Not on this floor, 0 Found. The 2 are stairs erased on their top floor, the stated limit.",
           "status": "frozen", "source": "docs/research/build-log.md, 31 Oct 23:55"}]


def catalog():
    return {e["id"]: e for path in CATALOGS if path.exists() for e in json.loads(path.read_text())}


def shown(source, entries):
    return bool(source) and not source.startswith(NEVER) and entries.get(source, {}).get("public_demo") == "yes"


def status(name, record, truth_sha, frozen):
    """A run's status from its eval_run.json and `frozen`, the content of eval/frozen.json now. A held-out set's run is held-out only if it is the set's one run and the code, scoring and truth it ran on are still the frozen ones."""
    if record.get("truth_sha256") != truth_sha:
        return "post hoc"   # no truth recorded with the run, or the truth changed after it
    if not run.held_out(name):
        return "development"
    pinned = run.pinned(frozen, name)
    held = record.get("frozen") == pinned == {"storey": record.get("storey_sha256"), "eval": run.eval_hash(), "truth": truth_sha} and frozen.get("runs", {}).get(name) == record.get("review")
    return "frozen, held-out" if held else "post hoc"


def collect(runs):
    """One entry per set: its status, its catalog source, and Storey's and the baseline's scores. A baseline asked on other drawings than the run's waits to be asked again."""
    frozen, entries = read_json(run.FROZEN), []
    for name, folder in runs.items():
        truth = score.load(name)
        answer = read_json(baseline.OUT / f"{name}.json")
        if answer and answer.get("request") != baseline.request(folder)[2]:
            beside = {"waiting": f"A baseline asked on the drawings in {folder}: the record at {baseline.OUT / name}.json saw others. Run python -m eval.baseline {name} {folder}."}
        else:
            beside = score.baseline(truth, answer) if answer else None
        entries.append({"set": name, "status": status(name, read_json(Path(folder) / "eval_run.json"), truth["sha256"], frozen), "source": truth["source"], "truth": truth["paths"], "truth_sha256": truth["sha256"],
                        "Storey": score.storey(truth, folder), "baseline": beside})
    return entries


def build(entries, catalog):
    """(the full report, the public page's copy, or None while a set waits). Score rows hold counts only, so every set keeps them."""
    rows, lists = [], {key: [] for key in LISTS}
    for e in entries:
        s, b = e["Storey"], e.get("baseline") or {}
        beside = {"wrong passes": {"absent cells": s.get("absent cells"), "erased cells": s.get("erased cells")}, "truth items found": {"out of": s.get("out of"), "baseline out of": b.get("out of")},
                  "present cells": {"shown": s.get("present shown")}}   # what each count is out of, all counts, so every set keeps them
        for k in SCORES:
            rows.append({"set": e["set"], "status": e["status"], "score": k, "Storey": s.get(k), "baseline": b.get(k)} | beside.get(k, {}))
        rows += [{"set": e["set"], "status": "post hoc", "score": k, "Storey": v, "baseline": None} for k, v in (s.get("post hoc") or {}).items()]
        for method, part in (("Storey", s), ("baseline", b)):
            for key, field in LISTS.items():
                lists[key] += [{"set": e["set"], "method": method, **row} for row in part.get(field, ())]
    public = {e["set"] for e in entries if shown(e.get("source"), catalog)}
    waiting = [part["waiting"] for e in entries for part in (e["Storey"], e.get("baseline") or {}) if part.get("waiting")]
    for e in entries:
        b = e.get("baseline")
        if not b or b.get("failed"):
            why = f"its last call failed. {b['failed']}" if b else f"none is recorded at {baseline.OUT / e['set']}.json."
            waiting.append(f"A baseline scored for {e['set']}: {why} Run python -m eval.baseline {e['set']} <its review folder>.")
    full = {"about": ABOUT, "scores": rows, **lists, "quoted": QUOTED, "waiting for the judge": waiting, "sets": entries}
    return full, None if waiting else {"about": ABOUT, "scores": rows, **{key: [r for r in v if r["set"] in public] for key, v in lists.items()}, "quoted": QUOTED}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    runs = dict(arg.split("=", 1) for arg in argv) if argv else read_json(run.RUNS)
    if not runs:
        sys.exit("No runs to score. Run python -m eval.run first, or name set=folder pairs.")
    full, public = build(collect(runs), catalog())
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    write_json(REPORT, full)
    print(json.dumps(full["scores"], indent=1))
    for line in full["waiting for the judge"]:
        print("Waiting for", line)
    if public is None:
        sys.exit(f"{PUBLIC} was not written, because a set above is still waiting.")
    write_json(PUBLIC, public)


if __name__ == "__main__":
    main()
