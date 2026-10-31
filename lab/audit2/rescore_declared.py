"""Retally declared from flags and verdicts, and diff the pre-freeze-named v2a flags against v2. No API. Run with -I."""
import json, sys, collections
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/declared")
import verdicts as V
D = "/Users/tsharma/Github/Meta/data/lab/declared/"
a = json.load(open(D + "flags_held_v2a.json")); b = json.load(open(D + "flags_held_v2.json"))
ia = {f["id"]: f for f in a["flags"]}; ib = {f["id"]: f for f in b["flags"]}
print("v2a", len(ia), "v2", len(ib), "only v2a", sorted(set(ia) - set(ib))[:20], "only v2", sorted(set(ib) - set(ia))[:20])
ch = [k for k in ia if k in ib and ia[k].get("text") != ib[k].get("text")]
print("changed text", len(ch), ch[:8])
for k in ch[:4]:
    print("  A:", ia[k]["text"][:200]); print("  B:", ib[k]["text"][:200])
sub = {"not_a_submission": 0}
rows = []
for f in b["flags"]:
    if f["set"] in ("af_ces_admin", "af_ces_shops", "wfu_efird", "wfu_huffman", "brandeis_hall") and f["check"] == 2:
        sub["not_a_submission"] += 1; continue
    v = V.held_verdict(f)
    rows.append((f["set"], f["check"], f["sub"], v[0], v[1], f["id"]))
c = collections.Counter(r[3] for r in rows); print("held flags", len(rows), c, "real+likely", c["real"] + c["likely"], sub)
print("by set", collections.Counter((r[0], r[3]) for r in rows))
print("by check", sorted(collections.Counter((r[1], r[3]) for r in rows).items()))
reader = collections.Counter()
for f in b["flags"]:
    if f["id"] in [r[5] for r in rows]:
        reader[(f.get("source") or f.get("read") or "?", V.held_verdict(f)[0])] += 1
print(reader.most_common(10))
print(list(b["flags"][0].keys()))
