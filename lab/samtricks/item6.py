"""Item 6: image before the question for prefix caching. Same prompt and schema as the title read, two runs per arm, plus a second question on the same image. Usage: item6.py"""
import json
import sys

sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/samtricks")
import api as a  # noqa: E402
import reg  # noqa: E402
import rules as R  # noqa: E402

sys.argv = sys.argv[:1] + ["score"] if len(sys.argv) < 2 else sys.argv
import e1_meta as e1  # noqa: E402  (constants only; its runner is not called)
import item1  # noqa: E402  (the plan box prompt and schema)
import truth_meta as tm  # noqa: E402

OUT = reg.OUT / "item6.json"
res = json.load(open(OUT)) if OUT.exists() else {}


def ask(k, order, q, schema, effort):
    im = reg.sheet(k)
    parts = [a.text(q), a.image(im)] if order == "text" else [a.image(im), a.text(q)]
    return a.chat(parts, schema=schema, effort=effort, cache_key=f"samtricks-{k}", kind="item6")


def sheet_run(k):
    """Serial inside one sheet, so each call sees the cache state the design needs."""
    if k in res:
        return k, res[k]
    o = {}
    for run in (0, 1):
        for arm in ("text", "image"):
            r, info = ask(k, arm, e1.PROMPT, e1.SCHEMA, "low")
            o[f"{arm}{run}"] = dict(out=r, info=info)
        if run == 0:
            for arm in ("image", "text"):
                r, info = ask(k, arm, item1.P_ONE, item1.ONE, "low")
                o[f"{arm}_q2"] = dict(out=r, info=info)
    return k, o


if __name__ == "__main__":
    for k, v in a.pmap(sheet_run, R.CACHE_SHEETS, 3):
        res[k] = v
    OUT.write_text(json.dumps(res, indent=0))
    tot = {}
    print("field scores of 3 per call: num level scale")
    blank = {"sheet_class": "other", "sheet_number": "", "sheet_title": "", "level": "", "scale": "", "confidence": 0}
    base = 0
    for k in R.CACHE_SHEETS:
        row = {}
        for arm in ("text0", "text1", "image0", "image1"):
            s = tm.score(k, res[k][arm]["out"])
            row[arm] = (int(s["num"]), int(s["level"]), int(s["scale"]))
            tot[arm[:-1]] = tot.get(arm[:-1], 0) + sum(row[arm])
        b = tm.score(k, blank)
        base += 2 * (int(b["num"]) + int(b["level"]) + int(b["scale"]))
        print(" ", k, row, {x: (res[k][x]["info"]["input"], res[k][x]["info"]["cached"]) for x in ("text0", "image0", "text1", "image1", "image_q2", "text_q2")})
    print("of 36:", tot, "blank answers baseline", base)
    q2 = [res[k]["image_q2"]["info"] for k in R.CACHE_SHEETS]
    q2t = [res[k]["text_q2"]["info"] for k in R.CACHE_SHEETS]
    print("second question, image first: cached/input", sum(i["cached"] for i in q2), "/", sum(i["input"] for i in q2), "usd", round(sum(i["usd"] for i in q2), 4), "output tokens", sum(i["output"] for i in q2))
    print("second question, text first: cached/input", sum(i["cached"] for i in q2t), "/", sum(i["input"] for i in q2t), "usd", round(sum(i["usd"] for i in q2t), 4), "output tokens", sum(i["output"] for i in q2t))
    pi, pc, po = a.m.PRICES["muse-spark-1.3"]
    saved = sum(i["cached"] for i in q2) * (pi - pc) / 1e6
    print("input usd saved on 6 second questions", round(saved, 5), "per question", round(saved / 6, 5))
    print("spent", round(a.m.spent("samtricks"), 3))
