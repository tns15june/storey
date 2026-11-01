"""Item 6 diagnostic, post hoc: the frozen second-question test cached nothing. Which change breaks the prefix, the schema or the question? Usage: item6b.py"""
import json
import sys

sys.path.insert(0, "lab/samtricks")
import api as a  # noqa: E402
import item6 as i  # noqa: E402

OUT = i.reg.OUT / "item6b.json"
Q2 = "Read this sheet once more and fill every field, paying most attention to the scale."
rows = {}


def pair(k, schema, q1, q2):
    r1 = i.ask(k, "image", q1, schema, "low")[1]
    r2 = i.ask(k, "image", q2, schema, "low")[1]
    return r1, r2


if not OUT.exists():
    for k, label, schema, q1, q2 in (("n161", "same schema", i.e1.SCHEMA, i.e1.PROMPT, Q2), ("c3", "same schema", i.e1.SCHEMA, i.e1.PROMPT, Q2),
                                     ("u2", "no schema", None, "What is the sheet number? Answer with the number only.", "What is the scale of the main drawing? Answer with the scale only."),
                                     ("af1", "no schema", None, "What is the sheet number? Answer with the number only.", "What is the scale of the main drawing? Answer with the scale only.")):
        r1, r2 = pair(k, schema, q1, q2)
        rows[f"{k} {label}"] = dict(first=r1, second=r2)
    OUT.write_text(json.dumps(rows, indent=1))
rows = json.load(open(OUT))
for k, v in rows.items():
    print(k, "first in/cached", v["first"]["input"], v["first"]["cached"], "second in/cached", v["second"]["input"], v["second"]["cached"], "second usd", round(v["second"]["usd"], 5))
print("spent", round(a.m.spent("samtricks"), 3))

# Third arm, post hoc: image first, JSON asked for in the text with no response_format, so the prefix can hit. A first short question warms the image.
OUT2 = i.reg.OUT / "item6c.json"
JSON_TAIL = "\nReturn only one JSON object with exactly these keys: sheet_class, sheet_number, sheet_title, level, scale, confidence."


def plain(k):
    r1, info1 = i.ask(k, "image", "What is the sheet number? Answer with the number only.", None, "low")
    r2, info2 = i.ask(k, "image", i.e1.PROMPT + JSON_TAIL, None, "low")
    return k, dict(first=info1, second=info2, text=r2)


if not OUT2.exists():
    OUT2.write_text(json.dumps(dict(a.pmap(plain, i.R.CACHE_SHEETS, 3)), indent=1))
rows = json.load(open(OUT2))
tot = in_ = hit = usd = 0
for k, v in rows.items():
    t = v["text"]
    o = json.loads(t[t.index("{"):t.rindex("}") + 1])
    s = i.tm.score(k, o)
    tot += int(s["num"]) + int(s["level"]) + int(s["scale"])
    in_ += v["second"]["input"]
    hit += v["second"]["cached"]
    usd += v["second"]["usd"]
    print(k, (s["num"], s["level"], s["scale"]), v["second"]["input"], v["second"]["cached"], round(v["second"]["usd"], 5))
print("plain JSON arm fields right of 18:", tot, "cached/input", hit, "/", in_, "usd per call", round(usd / 6, 5))
print("spent", round(a.m.spent("samtricks"), 3))
