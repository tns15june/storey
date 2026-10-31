"""Prints the tables used in the report and writes data/lab/declared/final.json. No model."""
import collections, json, sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/declared")
import rules as R
OUT = R.__file__.replace("lab/declared/rules.py", "data/lab/declared/")
v2 = json.load(open(OUT + "yield_v2.json"))
v3 = json.load(open(OUT + "yield_v3.json"))
v1 = json.load(open(OUT + "flags_dev_v1.json"))
keep = {f["text"] for f in json.load(open(OUT + "flags_dev_v2.json"))["flags"]}
scored1 = [f for f in v1["flags"] if not (f["set"] in R.NOT_A_SUBMISSION and f["check"] == 2)]
print("dev v1", len(scored1), "kept in v2", sum(f["text"] in keep for f in scored1))
rows = [r for r in v2["rows"] if r["verdict"] != "not_a_submission"]
line = lambda q: f"{len(q)}f {sum(r['verdict'] == 'real' for r in q)}r {sum(r['verdict'] == 'likely' for r in q)}l {sum(r['verdict'] == 'false' for r in q)}x"
table = {}
for s in R.DEV + R.HELD:
    q = [r for r in rows if r["set"] == s]
    q3 = [r for r in v3["rows"] if r["set"] == s and r["verdict"] != "not_a_submission"]
    table[s] = dict(frozen=line(q), posthoc=line(q3), per_check={k: line([r for r in q if r["check"] == k]) for k in range(1, 9) if [r for r in q if r["check"] == k]},
                    items={k: v["items"] for k, v in v2["by_set_check"][s].items() if v["items"]})
    print(s, table[s])
stats = collections.Counter()
for r in rows:
    stats[(r["source"], r["scope"], r["verdict"])] += 1
print(sorted(stats.items()))
real_roots = sorted({r["root"] for r in rows if r["verdict"] == "real"})
likely_only = sorted({r["root"] for r in rows if r["verdict"] == "likely"} - set(real_roots))
print(len(real_roots), real_roots)
print(len(likely_only), likely_only)
rr3 = [r for r in v3["rows"] if r["verdict"] != "not_a_submission"]
print("v3 real roots", len({r["root"] for r in rr3 if r["verdict"] == "real"}), "likely only", len({r["root"] for r in rr3 if r["verdict"] == "likely"} - {r["root"] for r in rr3 if r["verdict"] == "real"}))
print("source of flags frozen held", collections.Counter((r["source"], r["verdict"]) for r in rows if r["scope"] == "held"))
text_only = [r for r in rows if r["source"] == "text"]
print("text only", line(text_only), "spark", line([r for r in rows if r["source"] == "spark"]))
held_text = [r for r in text_only if r["scope"] == "held"]
print("held text only", line(held_text))
print(v2["not_a_submission"])
json.dump(dict(table=table, real_roots=real_roots, likely_only=likely_only), open(OUT + "final.json", "w"), indent=1)
MISSED = [
    dict(set="mml_baxter", sheet="A2.1", what="tag 10 sits on a pocket door, row 10 says 36 in swing. The arc of the untagged rear door paired with tag 10, so nothing fired"),
    dict(set="mml_baxter", sheet="A2.1", what="tag 12 sits on a bifold, row 12 says 36 in swing. No arc means no comparison"),
    dict(set="mml_baxter", sheet="A2.1", what="tag 16 and row 9 entry door handing. LEFT and RIGHT are never compared with the drawn swing"),
    dict(set="carpinteria_adu1", sheet="G-001", what="cover note says strike A2-202, Plan 1 has A1-202. No check reads sheet references in notes"),
    dict(set="carpinteria_adu2", sheet="G-001", what="same note names A1-121, A1-122 and A1-201, Plan 2 numbers its sheets A2-"),
    dict(set="nibs_clinic", sheet="A1-4", what="SHEET __ OF 316, sequence blank on the sheet. No check compares the printed sequence"),
    dict(set="mml_capitol", sheet="A2.2", what="finish row 102 prints KTICHEN, same in Baxter. Fuzzy room match passes it, trivial"),
]
d = json.load(open(OUT + "final.json"))
json.dump(dict(d, missed_by_hand=MISSED), open(OUT + "final.json", "w"), indent=1)
