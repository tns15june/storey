"""Step 3. Spark floor comparison, 3 runs per stack, medium effort, raw 2000 px sheets in the Spark joint order. Cached per run, so a rerun spends nothing.
The stair prompt is Round 1's FLOOR_P word for word. One paragraph is appended for elevators and shafts. Floors are coded F1..Fn so answers are matched by code, not array index.
Usage: compare.py <building ...>"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "lab/presence")
import pres
from pres import OUT, SH
import metalab as m

FLOOR_P = """These images are floor plans of one building, one image per floor, in this order: {order}. Staircases in the same place on consecutive floors are the same staircase.
List every distinct staircase in the building. For each give a short name, and for every floor in the list say whether it is drawn on that floor (present) and its centre x, y in the 0-1000 range of that floor's image (0, 0 if absent). A stair drawn struck through or filled in counts as absent. Compare the floors carefully, stair by stair."""
EXTRA = """
Also list every elevator and every vertical shaft in a separate list called shafts, in the same form, with kind set to elevator or shaft. A shaft is a box drawn with a diagonal cross, a duct, a chimney or a ventilation shaft. Set floor to the floor's code, F1, F2 and so on."""
STR = {"type": "string"}
FL = {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["floor", "present", "x", "y"],
                                 "properties": {"floor": STR, "present": {"type": "boolean"}, "x": {"type": "integer"}, "y": {"type": "integer"}}}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["stairs", "shafts"], "properties": {
    "stairs": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["name", "floors"], "properties": {"name": STR, "floors": FL}}},
    "shafts": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["name", "kind", "floors"],
                                          "properties": {"name": STR, "kind": {"type": "string", "enum": ["elevator", "shaft"]}, "floors": FL}}}}}
EFFORT = "medium"


def order_of(b):
    """Sheet indices bottom to top from the joint call, and the printed labels, both straight from Step 1."""
    s = pres.load_spark()
    j = s[f"joint/{b}"]
    got = [j["shown"][q - 1] for q in j["read"]["bottom_to_top"]]
    return got, [s[f"{b}/{i}"]["read"]["printed_floor_label"] for i in got]


def run(key, paths, order, labels, r):
    p = OUT / f"spark/{key}_r{r}.json"
    if p.exists():
        return json.load(open(p))
    txt = ", ".join(f"F{k + 1} ({lab})" for k, lab in enumerate(labels))
    out, info = m.spark(FLOOR_P.format(order=txt) + EXTRA, [str(x) for x in paths], schema=SCHEMA, effort=EFFORT)
    o = dict(key=key, run=r, order=order, out=out, info=info)
    json.dump(o, open(p, "w"), indent=1)
    print(key, "run", r, "stairs", len(out["stairs"]), "shafts", len(out["shafts"]), "usd", round(info["usd"], 4), "s", round(info["seconds"], 1), "in", info["input"], "out", info["output"], flush=True)
    return o


def clean_paths(b, order):
    return [SH / f"{b}/{i}.jpg" for i in order]


if __name__ == "__main__":
    (OUT / "spark").mkdir(exist_ok=True)
    runs = (0, 1, 2)
    args = sys.argv[1:]
    if args and args[0] == "--planted":
        import planted
        jobs = []
        for key, (b, cases) in (planted.DEV_KEYS if args[1] == "dev" else planted.HELD_KEYS).items():
            if len(args) > 2 and b not in args[2:]:
                continue
            order, labels = order_of(b)
            jobs += [lambda key=key, b=b, order=order, labels=labels, r=r: run(key, planted.paths(b, key, order), order, labels, r) for r in runs]
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(lambda f: f(), jobs))
        print("spent", round(m.spent("presence"), 4))
        sys.exit(0)
    if args and args[0] == "--one":
        runs, args = (0,), args[1:]
    jobs = []
    for b in args:
        order, labels = order_of(b)
        jobs += [lambda b=b, order=order, labels=labels, r=r: run(b, clean_paths(b, order), order, labels, r) for r in runs]
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda f: f(), jobs))
    print("spent", round(m.spent("presence"), 4))
