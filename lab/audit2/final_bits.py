"""No-SAM comparator rescored, and the real examiner comments in full. No API. Run with -I."""
import json, sys
sys.path.insert(0, "lab/audit2")
import rescore_tightseed as S
for st in ("bme_k", "royce_east"):
    print("no SAM", st, S.tally(S.rows_of(st, S.load(st, "NS"))), "| P", S.tally(S.rows_of(st, S.load(st, "P_960")))["pairs"], "Ppin", S.tally(S.rows_of(st, S.load(st, "Ppin_960")))["pairs"])
r = json.load(open("data/lab/realsets/va_trb/richmond_truth.json"))
for c in r["comments"]:
    print(c["id"], "sheet_only" if c["sheet_only"] else "not", "|", c["text"][:230], "|", c.get("note", "")[:120])
print(r["declared"] if "declared" in r else "")
d = json.load(open("data/lab/realsets/navy/defect_truth.json"))
print([k for k in d])
rows = d.get("defects") or d.get("rows") or []
import collections
print(len(rows), collections.Counter(x.get("kind") or x.get("type") or x.get("class") for x in rows))
print(json.dumps(rows[0])[:400] if rows else "")
