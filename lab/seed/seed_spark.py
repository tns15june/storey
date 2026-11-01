"""Spark finds staircases on every floor of every building. Cached per call. Usage: seed_spark.py variant [buildings]"""
import sys; sys.path.insert(0, "lab/seed"); sys.path.insert(0, "lab")
from common import *
from concurrent.futures import ThreadPoolExecutor
import metalab as m

BOX = {"type": "object", "additionalProperties": False, "required": ["object_name", "x_min", "y_min", "x_max", "y_max"],
       "properties": {"object_name": {"type": "string"}, **{k: {"type": "integer"} for k in ("x_min", "y_min", "x_max", "y_max")}}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["objects"], "properties": {"objects": {"type": "array", "items": BOX}}}
TAIL = ' For each object, answer in the format {"object_name": "<name>", "x_min": <int>, "y_min": <int>, "x_max": <int>, "y_max": <int>}. The coordinates should be in the 0-1000 range, x from the left edge and y from the top edge of the image. If you cannot find an object, omit it.'
PLAN = ('You are an object grounding expert. This image is an architectural floor plan. Provide the bounding box coordinates of every staircase drawn on it. '
        'A staircase is drawn as a run of closely spaced parallel lines (the treads), usually one or two flights side by side with a break line or an UP or DN arrow. '
        'Include curved stairs, exterior stairs and short flights. One box per staircase, tight around all its flights. Ensure the objects accurately match the request and do not miss any objects.' + TAIL)
DOC = 'You are an object grounding expert. Provide the bounding box coordinates of the objects in the image: "staircase". Ensure the objects accurately match the request and do not miss any objects.' + TAIL
LIFT = ('You are an object grounding expert. This image is an architectural floor plan. Provide the bounding box coordinates of every elevator shaft (a small box with an X through it, or labelled ELEVATOR) '
        'and every toilet room (labelled TOILET, WC, LAV or showing toilet fixtures). Name each object "elevator" or "toilet". Ensure the objects accurately match the request and do not miss any objects.' + TAIL)
VARIANTS = {"plan": (PLAN, None), "plan_low": (PLAN, "low"), "doc": (DOC, None), "lift": (LIFT, None)}
BUILDINGS = ["bme_f", "schenley", "habs_stelizabeths_atkins", "habs_stelizabeths_center"]


def one(args):
    bid, i, var = args
    path = OUT / "spark" / f"{bid}_f{i}_{var}.json"
    if path.exists():
        return json.load(open(path))
    prompt, effort = VARIANTS[var]
    r, info = m.spark(prompt, images=[str(OUT / bid / f"f{i}.png")], schema=SCHEMA, effort=effort)
    out = {"objects": r["objects"], "info": info}
    json.dump(out, open(path, "w"), indent=1)
    print(bid, i, var, len(r["objects"]), round(info["usd"], 4), round(info["seconds"]), flush=True)
    return out


if __name__ == "__main__":
    (OUT / "spark").mkdir(exist_ok=True)
    var = sys.argv[1]
    bids = sys.argv[2:] or BUILDINGS
    jobs = [(b, i, var) for b in bids for i in range(len(CAT[b]["floors"]))]
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, jobs))
    print(var, "calls", len(res), "usd", round(sum(r["info"]["usd"] for r in res), 4), "mean s", round(np.mean([r["info"]["seconds"] for r in res]), 1))
