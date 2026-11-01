"""Part 1 scores. Truth is the catalog floor order, bottom to top, which the renderer keeps as sheet index 0..n-1."""
import json
import re
import unicodedata
from pathlib import Path

OUT = Path("data/lab/align")
S = json.load(open(OUT / "spark_sheets.json"))
I = json.load(open(OUT / "sheets/index.json"))
pdf = json.load(open(OUT / "pdf_order.json"))
# Printed truth, read off the sheets by eye. bme_k 3 is printed II. EMELET.
PRINTED = {"bme_f": ["ALAGSOR", "FOLDSZINT", "I. EMELET", "II. EMELET"], "bme_k": ["ALSO FOLDSZINT", "FOLDSZINT", "I. EMELET", "II. EMELET"]}


def norm(t):
    return unicodedata.normalize("NFD", t).encode("ascii", "ignore").decode().upper().replace("FLOOR PLAN", "").replace("PLAN", "").replace("FLOOR", "").strip(" .")


def label_rank(t):
    """Fixed table: the product should rank from the printed label, not the model's integer."""
    t = norm(t)
    if "SUB" in t and "BASE" in t: return -2
    if "ALAGSOR" in t or "BASEMENT" in t or "CELLAR" in t: return -1
    if "ALSO FOLDSZINT" in t: return -0.5
    if "FOLDSZINT" in t or "GROUND" in t: return 0
    if t.startswith("MAIN"): return 1
    if "UPPER" in t: return 2
    m = re.match(r"^(I{1,3}|IV|V)\.? ?EMELET", t)
    if m: return {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}[m.group(1)]
    for k, w in enumerate(["FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH"], 1):
        if w in t: return k
    if "ATTIC" in t: return 90
    if "ROOF" in t: return 91
    return None


rep = {}
for bid, rows in I.items():
    n = len(rows)
    reads = [S[f"{bid}/{i}"]["read"] for i in range(n)]
    ranks = [r["storey_rank"] for r in reads]
    labs = [label_rank(r["printed_floor_label"]) for r in reads]
    J = S[f"joint/{bid}"]
    joint = [J["shown"][p - 1] for p in J["read"]["bottom_to_top"]]
    rep[bid] = {"n": n, "rank_strict": all(a < b for a, b in zip(ranks, ranks[1:])), "ranks": ranks, "label_rank_strict": all(a is not None and b is not None and a < b for a, b in zip(labs, labs[1:])),
                "label_ranks": labs, "labels": [r["printed_floor_label"] for r in reads], "joint_ok": joint == list(range(n)), "joint": joint,
                "pdf_text": None if bid not in pdf else all(a["rank"] is not None and b["rank"] is not None and a["rank"] < b["rank"] for a, b in zip(pdf[bid]["sheets"], pdf[bid]["sheets"][1:]))}
    print(f"{bid:26s} n={n} rank_sort={rep[bid]['rank_strict']!s:5} label_table={rep[bid]['label_rank_strict']!s:5} joint={rep[bid]['joint_ok']!s:5} pdf={rep[bid]['pdf_text']} {rep[bid]['labels']} {joint}")
print("rank_sort", sum(r["rank_strict"] for r in rep.values()), "label_table", sum(r["label_rank_strict"] for r in rep.values()), "joint", sum(r["joint_ok"] for r in rep.values()), "of", len(rep))
print("pdf", {k: v["pdf_text"] for k, v in rep.items() if v["pdf_text"] is not None})
print("spark cost per sheet", round(sum(v["info"]["usd"] for k, v in S.items() if not k.startswith("joint")) / sum(not k.startswith("joint") for k in S), 4), "joint per building", round(sum(v["info"]["usd"] for k, v in S.items() if k.startswith("joint")) / 10, 4))
json.dump(rep, open(OUT / "order_scores.json", "w"), indent=1, ensure_ascii=False)
