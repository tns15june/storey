"""Yield tables from flags and verdicts. No model. Usage: yields.py. Writes data/lab/declared/yield.json."""
import collections
import json
import sys

sys.path.insert(0, "lab/declared")
import rules as R
import verdicts as V

OUT = R.__file__.replace("lab/declared/rules.py", "data/lab/declared/")


def load(name):
    return json.load(open(OUT + name))


def judge(f, dev):
    if f["set"] in R.NOT_A_SUBMISSION and f["check"] == 2:
        return "not_a_submission", "", f["text"]
    return V.DEV[f["id"]] if dev else V.held_verdict(f)


def tally(rows):
    c = collections.Counter(r["verdict"] for r in rows)
    return dict(flags=sum(c[k] for k in ("real", "likely", "false", "cannot_tell")), real=c["real"], likely=c["likely"], false=c["false"], cannot_tell=c["cannot_tell"], roots=len({r["root"] for r in rows if r["verdict"] in ("real", "likely") and r["root"]}))


def outcome(pooled, held):
    """The frozen ship rule, in rules.py. Real means real, likely counts only inside p."""
    n, p = pooled["flags"], (pooled["real"] + pooled["likely"]) / pooled["flags"] if pooled["flags"] else 0
    if n >= 3 and p >= 0.8 and held["false"] == 0:
        return "Fix"
    if (n >= 3 and p >= 0.5) or (n < 3 and pooled["real"] >= 1):
        return "Check"
    return "Drop"


def main():
    dev, held = load("flags_dev_v2.json"), load("flags_held_v2.json")
    rows = [dict(f, verdict=(j := judge(f, True))[0], root=j[1], note=j[2], scope="dev") for f in dev["flags"]]
    rows += [dict(f, verdict=(j := judge(f, False))[0], root=j[1], note=j[2], scope="held") for f in held["flags"]]
    stats = {**dev["stats"], **held["stats"]}
    res = dict(version=R.VERSION)
    res["by_set_check"] = {}
    for s in R.DEV + R.HELD:
        res["by_set_check"][s] = {str(k): dict(tally([r for r in rows if r["set"] == s and r["check"] == k]), items=stats.get(s, {}).get(str(k), {}).get("items", 0), note=stats.get(s, {}).get(str(k), {}).get("note", "")) for k in range(1, 9)}
    res["by_scope"] = {k: tally([r for r in rows if r["scope"] == k]) for k in ("dev", "held")}
    res["by_check"] = {}
    subs = sorted({(r["check"], r["sub"]) for r in rows if r["verdict"] != "not_a_submission"})
    res["by_sub"] = {}
    for k in range(1, 9):
        q = [r for r in rows if r["check"] == k and r["verdict"] != "not_a_submission"]
        t, h = tally(q), tally([r for r in q if r["scope"] == "held"])
        res["by_check"][k] = dict(t, held=h, items=sum(v.get(str(k), {}).get("items", 0) for v in stats.values()), rule=outcome(t, h))
    for k, sub in subs:
        q = [r for r in rows if r["sub"] == sub]
        t, h = tally(q), tally([r for r in q if r["scope"] == "held"])
        res["by_sub"][f"{k}.{sub}"] = dict(t, held=h, rule=outcome(t, h))
    res["false_by_step"] = collections.Counter(r["note"].split(":")[0] for r in rows if r["verdict"] == "false").most_common()
    res["not_a_submission"] = collections.Counter(r["set"] for r in rows if r["verdict"] == "not_a_submission")
    res["roots"] = sorted({(r["root"], r["set"]) for r in rows if r["verdict"] in ("real", "likely") and r["root"]})
    json.dump(dict(res, rows=[{k: r[k] for k in ("id", "set", "check", "sub", "sheet", "text", "source", "scope", "verdict", "root", "note")} for r in rows]), open(OUT + "yield_v2.json", "w"), indent=1, default=str)
    print(json.dumps(res["by_scope"]))
    for k, v in res["by_check"].items():
        print(k, {a: b for a, b in v.items() if a != "held"}, "heldfalse", v["held"]["false"])
    for k, v in res["by_sub"].items():
        print(k, v["flags"], v["real"], v["likely"], v["false"], v["rule"])
    print(res["false_by_step"])
    print(len(res["roots"]), res["roots"])


if __name__ == "__main__":
    main()
