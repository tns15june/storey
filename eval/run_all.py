"""Runs every evaluation set once on the frozen code, four at a time, then the baseline for each. A second cycle starts after its first.
This is the driver that made the published numbers on 1 November 2026, after python -m eval.run --freeze. A held-out set runs once: a second call refuses it.

    .venv/bin/python eval/run_all.py <data folder> [set ...]

Writes <data folder>/progress.jsonl: one line per finished step with its exit code and the tail of its output.
"""
import concurrent.futures as cf
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = str(REPO / ".venv/bin/python")
BUILDINGS = ["710", "3370", "5615", "16281", "3128", "2828", "6267"]
PLANTS = ["habs_stelizabeths_center_plant", "habs_stelizabeths_atkins_plant", "habs_va_bldg1_plant", "schenley_plant", "wfu_huffman_plant", "bme_f_plant"]   # bme_k_plant left out: its truth boxes are tighter than two of its three stairs, so those erasures left most treads
FIRST = [f"{b}_1_clean" for b in BUILDINGS] + [f"{b}_1_mixed" for b in BUILDINGS] + ["2828_1_corridor"] + PLANTS + ["mml_capitol", "va_trb_richmond", "army_bt_bcof", "army_bt_bcof_plant", "queens_royce"]
SECOND = {f"{b}_1_second": f"{b}_1_mixed" for b in BUILDINGS}
lock = threading.Lock()


def step(log, name, args):
    t = time.time()
    p = subprocess.run([PY, *args], cwd=REPO, capture_output=True, text=True)
    row = {"set": name, "step": args[1], "code": p.returncode, "seconds": round(time.time() - t), "out": (p.stdout + p.stderr)[-1500:]}
    with lock, open(log, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"{name} {args[1]} exit {p.returncode} in {row['seconds']} s", flush=True)
    return p.returncode == 0


def review(data, log, name):
    ok = step(log, name, ["-m", "eval.run", name, "--data", str(data / "reviews")])
    if ok:
        step(log, name, ["-m", "eval.baseline", name, str(data / "reviews" / name)])
    return ok


def main():
    data = Path(sys.argv[1]).resolve()
    only = set(sys.argv[2:])
    data.mkdir(parents=True, exist_ok=True)
    log = data / "progress.jsonl"
    first = [s for s in FIRST if not only or s in only]
    second = {s: p for s, p in SECOND.items() if not only or s in only}
    with cf.ThreadPoolExecutor(4) as pool:
        done = {s: pool.submit(review, data, log, s) for s in first}
        later = []
        for s, parent in second.items():
            if parent in done:
                later.append(pool.submit(lambda s=s, parent=parent: done[parent].result() and review(data, log, s)))
            else:
                later.append(pool.submit(review, data, log, s))
        for f in [*done.values(), *later]:
            f.result()


if __name__ == "__main__":
    main()
