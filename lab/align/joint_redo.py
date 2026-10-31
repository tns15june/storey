"""Rerun the three joint-order calls whose seeded shuffle happened to be the true order, with the sheets reversed."""
import json
import sys

import cv2

sys.path.insert(0, "/Users/tsharma/Github/Meta/lab")
import metalab as m
from pathlib import Path

OUT = Path("/Users/tsharma/Github/Meta/data/lab/align")
S = json.load(open(OUT / "spark_sheets.json"))
schema = {"type": "object", "additionalProperties": False, "required": ["bottom_to_top"], "properties": {"bottom_to_top": {"type": "array", "items": {"type": "integer"}}}}
for bid in ("schenley", "mml_baxter", "nibs_clinic"):
    n = len(S[f"joint/{bid}"]["shown"])
    order = list(range(n))[::-1]
    if S[f"joint/{bid}"]["shown"] == order:  # already redone, do not spend again
        continue
    ims = [cv2.imread(str(OUT / f"sheets/{bid}/{i}.jpg")) for i in order]
    p = (f"These are {n} sheets from one building's drawing set, numbered 1 to {n} in the order attached. Each shows one storey. "
         "List the sheet numbers from the lowest storey to the highest storey, using the floor names printed on the sheets.")
    r, info = m.spark(p, ims, schema=schema, effort="low")
    S[f"joint/{bid}"] = {"shown": order, "read": r, "info": info}
    print(bid, r, [order[q - 1] for q in r["bottom_to_top"]])
json.dump(S, open(OUT / "spark_sheets.json", "w"), indent=1, ensure_ascii=False)
