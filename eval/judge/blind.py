"""Copies each match sheet for the judges with nothing that names its method, and writes the key that undoes it.

    python eval/judge/blind.py <repo>/data/eval/sheets <out dir>
    python eval/judge/blind.py --unblind <out dir> <rulings.json>

<out dir>/sheet_NN.json holds set, how, claims and truth. Each claim keeps only its sheet and statement, under a neutral id, C01 and on, in an order shuffled by a seed from the
sheet's sha256, so neither the ids, a severity nor the order tells Storey's claims from the baseline's. Sheets are numbered in sha256 order, not by set and method.
<out dir>.key.json maps each NN to the sheet's set, method, path and sha256, and each neutral id to the claim's own id. It sits beside the folder, not in it, because the judges are given the folder.
--unblind reads what eval/judge/workflow.js returns, saved as JSON, and writes eval/matches/<set>.<method>.json in the format eval/score.py reads.
"""
import json
import random
import sys
from pathlib import Path

MATCHES = Path(__file__).resolve().parent.parent / "matches"
JUDGE = ("Two judges, each a Claude agent shown only a blinded copy of the match sheet: its truth items, and its claims under neutral ids in a shuffled order with each claim's sheet and statement "
         "and nothing that names the method, ruled on every claim without seeing each other's rulings. An adjudicating agent ruled on each claim where they differed. "
         "Rules: the same problem at the same place; a general claim states nothing; when unsure, no match.")


def key_path(folder):
    folder = Path(folder).resolve()
    return folder.parent / f"{folder.name}.key.json"


def blind(src, out):
    """Writes to `out` a blinded copy of each match sheet in `src` that needs a judge, and the key beside `out`. Returns the key."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sheets = sorted(((path, json.loads(path.read_text())) for path in Path(src).glob("*.json")), key=lambda x: x[1]["sha256"])
    key = {}
    for n, (path, sheet) in enumerate([(p, s) for p, s in sheets if s["claims"] and s["truth"]], 1):   # with no claim or no truth item the scorer has nothing to judge
        claims = list(sheet["claims"])
        random.Random(sheet["sha256"]).shuffle(claims)
        ids = {f"C{i:02d}": c["id"] for i, c in enumerate(claims, 1)}
        name = f"sheet_{n:02d}"
        (out / f"{name}.json").write_text(json.dumps({"set": sheet["set"], "how": sheet["how"], "claims": [{"id": i, "sheet": c["sheet"], "statement": c["statement"]} for i, c in zip(ids, claims)],
                                                      "truth": sheet["truth"]}, indent=1))
        key[name] = {"set": sheet["set"], "method": sheet["method"], "path": str(path), "sha256": sheet["sha256"], "ids": ids}
    key_path(out).write_text(json.dumps(key, indent=1))
    return key


def unblind(folder, results, matches=MATCHES):
    """Writes the judges' rulings on each blinded sheet as eval/matches/<set>.<method>.json, each neutral id back to the claim's own id. Returns the paths written.
    `results` is the list eval/judge/workflow.js returns. A batch it returned no sheets for is skipped, so the report keeps waiting for those sheets.
    Nothing is written unless every sheet ruled on passes and is ruled on in one place only, and no mapping already there is replaced."""
    key = json.loads(key_path(folder).read_text())
    ruled = [(key[s["sheet"]], s) for part in results for s in part.get("sheets") or ()]
    for k, s in ruled:
        known = {t["id"] for t in json.loads((Path(folder) / f"{s['sheet']}.json").read_text())["truth"]}
        if sorted(r["id"] for r in s["rows"]) != sorted(k["ids"]) or not all(isinstance(r.get("truth"), list) and set(r["truth"]) <= known for r in s["rows"]):
            raise ValueError(f"{s['sheet']} needs exactly one ruling per claim, each with a list of truth ids from that sheet, so no mapping is written.")
    paths = [Path(matches) / f"{k['set']}.{k['method']}.json" for k, _ in ruled]
    if twice := sorted({p.name for p in paths if paths.count(p) > 1}):
        raise ValueError(f"No mapping is written, because the rulings answer {', '.join(twice)} more than once, and one judging would replace another.")
    if kept := [str(p) for p in paths if p.exists()]:
        raise ValueError(f"No mapping is written, because {', '.join(kept)} already exist and may be the published record. Move them aside to judge those sheets again.")
    for (k, s), path in zip(ruled, paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"set": k["set"], "method": k["method"], "judge": JUDGE, "sheet_sha256": k["sha256"],
                                    "matches": [{"id": k["ids"][r["id"]], "truth": r["truth"], "note": r.get("note", "")} for r in s["rows"]]}, indent=1))
    return paths


if __name__ == "__main__":
    if sys.argv[1] == "--unblind":
        print("\n".join(map(str, unblind(sys.argv[2], json.loads(Path(sys.argv[3]).read_text())))))
    else:
        print(json.dumps(blind(sys.argv[1], sys.argv[2]), indent=1))
