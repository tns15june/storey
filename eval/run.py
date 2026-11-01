"""Runs the real pipeline, the code the server runs, on one evaluation set in a scratch data folder under a spend cap, and records USD and minutes per stage.

    python -m eval.run 710_1_mixed --data <scratch>/reviews [--cap 2]
    python -m eval.run --freeze

The data folder holds the reviews, their ledger and a model cache for each set, which a second review shares with its first, as on the server. eval_run.json says whether the run began in a new folder on a cold cache, and only such a run's cost and minutes are published.
A held-out set runs only when the code under storey/, the scoring files and the set's truth hash to the values in eval/frozen.json, and only once: a later call may resume its folder and nothing else.
A set's truth is every file written for it: a blind audit <set>.json and a stair truth <set>_stairs.json in eval/truth, either or both, and frozen.json pins one hash per set over them.
A plant of a held-out set is held out too, whatever HELD_OUT lists. When a plant's truth records the sha256 of each of its images, as eval.plant writes it, a run refuses an image changed after the plant was made.
Freeze once the product code is final, never before. A hash frozen.json holds never changes, so code tuned on held-out output cannot be frozen again.
"""
import argparse
import functools
import hashlib
import json
import shutil
import time
from pathlib import Path

import pymupdf

from storey import meta, pipeline
from storey.ingest import ingest
from storey.meta import Meter
from storey.pipeline import read_json, write_json

ROOT = Path(__file__).resolve().parent.parent
STOREY = ROOT / "storey"
FROZEN = ROOT / "eval/frozen.json"
TRUTH = ROOT / "eval/truth"
GENERATED = ROOT / "data/generated"
PLANTS = ROOT / "data/eval/plants"
RUNS = ROOT / "data/eval/runs.json"
HELD_OUT = ("army_bt_bcof", "queens_royce", "army_bt_bcof_plant", "queens_royce_plant")
EVAL = ("run.py", "score.py", "baseline.py")   # which pages are sent, how a run is scored and what the baseline is asked: none may change once held-out output is seen
CAP_USD = 2.0
RICHMOND = "va_trb_richmond"
RICHMOND_TRUTH = ROOT / "data/lab/realsets/va_trb/richmond_truth.json"
PACKET = ROOT / "data/plans/va_trb/agendas-trb-agenda-06-12-2026.pdf"
PACKET_PAGE = 107                  # counted from 1, as printed on the packet: the sheet as submitted, before anyone marked it
REFUSED = (84, 86, 87, 101, 176)   # the examiner's and the appellant's marks, which hold the answers
# Pages counted from 1. queens_royce is its Phase 2, the four pages its stair truth lists.
SOURCES = {"mml_capitol": (ROOT / "data/plans/mml_capitol/THE-CAPITOL.pdf", None), "army_bt_bcof": (ROOT / "data/plans/army_cos/bt_bcof_sample.pdf", None),
           "queens_royce": (ROOT / "data/plans/queens_royce/jean_royce_hall.pdf", [6, 7, 8, 9])}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hash(root=None):
    """(one sha256 over every file under storey/, {path: sha256}). Compiled caches and Finder files change with no code change, so they are left out."""
    root = Path(root or STOREY)
    files = {p.relative_to(root).as_posix(): _sha(p) for p in sorted(root.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and p.name != ".DS_Store"}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), files


def eval_hash():
    return {name: _sha(ROOT / "eval" / name) for name in EVAL}


def pinned(frozen, name):
    """The hashes a held-out run of `name` must match, as `frozen`, the content of eval/frozen.json, holds them."""
    return {"storey": frozen.get("storey"), "eval": frozen.get("eval"), "truth": frozen.get("truth", {}).get(name)}


def held_out(name):
    """Whether a set is held out: HELD_OUT lists it, or it is a plant of a set HELD_OUT lists, whatever the list says of the plant."""
    return name in HELD_OUT or read_json(PLANTS / name / "truth.json").get("source") in HELD_OUT


def truth_paths(name):
    """Every file holding a set's truth, empty when none is written: a generated set's or a plant's, the Richmond record, or a blind audit and a stair truth, either or both."""
    for path in (GENERATED / name / "truth.json", PLANTS / name / "truth.json"):
        if path.exists():
            return [path]
    if name == RICHMOND:
        return [RICHMOND_TRUTH] if RICHMOND_TRUTH.exists() else []
    return [path for path in (TRUTH / f"{name}.json", TRUTH / f"{name}_stairs.json") if path.exists()]


def truth_hash(paths):
    """One sha256 over all of a set's truth files, or None when it has none."""
    return hashlib.sha256(json.dumps({Path(p).name: _sha(p) for p in paths}, sort_keys=True).encode()).hexdigest() if paths else None


def inputs(name):
    """[(file, pages counted from 1, or None for the whole file)] to upload for a set."""
    if (GENERATED / name / "set.pdf").exists():
        return [(GENERATED / name / "set.pdf", None)]
    if (PLANTS / name).is_dir():
        return [(p, None) for p in sorted((PLANTS / name).glob("*.png"))]
    if name == RICHMOND:
        return [(PACKET, [PACKET_PAGE])]
    if name in SOURCES:
        return [SOURCES[name]]
    raise ValueError(f"No drawings are listed for {name}.")


def pages(pdf, numbers, out):
    """Writes the listed pages of a PDF, counted from 1, to `out`. The Richmond packet gives up page 107 and nothing else, whatever is asked."""
    numbers = list(numbers or ())
    if Path(pdf).name == PACKET.name and (set(numbers) & set(REFUSED) or numbers != [PACKET_PAGE]):
        raise ValueError(f"The Richmond packet may pass page {PACKET_PAGE} only. Pages {', '.join(map(str, REFUSED))} hold the examiner's marks and the answers.")
    with pymupdf.open(pdf) as source, pymupdf.open() as picked:
        if not numbers or not all(1 <= n <= source.page_count for n in numbers):
            raise ValueError(f"{Path(pdf).name} has no pages {numbers}.")
        for n in numbers:
            picked.insert_pdf(source, from_page=n - 1, to_page=n - 1)
        picked.save(out)
    return out


def _upload(source, numbers, target):
    if numbers is None and source.name != PACKET.name:
        shutil.copyfile(source, target)
    else:
        pages(source, numbers, target)   # the packet always passes the page rule, even when a caller asks for all of it


def run(name, data, cap=CAP_USD):
    """Runs one set in `data`/<name> and returns that folder. A held-out set on code, scoring or truth that is not frozen raises ValueError before anything is read or spent, and so does a second held-out run of a set."""
    digest, files = code_hash()
    truth = truth_paths(name)
    truth_sha = truth_hash(truth)
    frozen = read_json(FROZEN)
    held = held_out(name)
    if held and not (truth_sha and pinned(frozen, name) == {"storey": digest, "eval": eval_hash(), "truth": truth_sha}):
        raise ValueError(f"{name} is held out, so it runs only on the code, scoring and truth recorded in {FROZEN.name}. Freeze them with python -m eval.run --freeze once the product code is final.")
    path = Path(data).resolve() / name
    if (path / "eval_run.json").exists():
        raise ValueError(f"{path} holds a finished run. Use a new data folder.")
    first = frozen.get("runs", {}).get(name)
    if held and first not in (None, str(path)):
        raise ValueError(f"{name} had its held-out run in {first}, and a held-out set runs once, so its number cannot be the best of several. Resume that folder if it did not finish.")
    sources = inputs(name)
    own = read_json(truth[0]) if truth else {}   # a second review's generated truth and a plant's are each one file
    if "png_sha256" in own and own["png_sha256"] != {p.name: _sha(p) for p, _ in sources}:
        raise ValueError(f"The images of {name} are not the ones its truth records, so they changed after the plant was made. Make the plant again.")
    parent = own.get("parent")
    cache = path.parent / "cache" / (parent or name)
    # shortcut: a second review run again after its folder was deleted finds its own replies in the shared cache and reads fresh, use a new data folder for a published rerun.
    fresh = not (path / "sheets.json").exists() and (parent is not None or not cache.exists())
    sent = hashlib.sha256(json.dumps([[_sha(p), n] for p, n in sources]).encode()).hexdigest()
    if (path / "sheets.json").exists() and read_json(path / "stage_upload.json").get("inputs") != sent:
        raise ValueError(f"{path} holds another upload of {name}, and a rerun would score that one. Use a new data folder.")
    if held and first is None:
        write_json(FROZEN, frozen | {"runs": frozen.get("runs", {}) | {name: str(path)}})
    if not (path / "sheets.json").exists():
        (path / "upload").mkdir(parents=True, exist_ok=True)
        uploads = [path / "upload" / f"{n:04d}{p.suffix.lower()}" for n, (p, _) in enumerate(sources)]
        for (source, numbers), target in zip(sources, uploads):
            _upload(source, numbers, target)
        started = time.monotonic()
        ingest(uploads, path)
        write_json(path / "cost.json", {"ingest": {"usd": 0, "seconds": round(time.monotonic() - started, 1)}})
        write_json(path / "stage_upload.json", {"name": name, "parent": parent, "inputs": sent})
    live, old = pipeline.Meter, meta.CACHE
    pipeline.Meter, meta.CACHE = functools.partial(Meter, cap_usd=cap), cache
    try:
        pipeline.run(path)
    finally:
        pipeline.Meter, meta.CACHE = live, old
    costs = read_json(path / "cost.json")
    write_json(path / "eval_run.json", {
        "set": name, "review": str(path), "fresh": fresh, "storey_sha256": digest, "files": files, "eval": eval_hash(), "frozen": pinned(frozen, name), "truth": [str(p) for p in truth], "truth_sha256": truth_sha,
        "cap_usd": cap, "usd": round(sum(c["usd"] for c in costs.values()), 4), "minutes": round(sum(c["seconds"] for c in costs.values()) / 60, 2),
        "stages": {stage: {"usd": c["usd"], "minutes": round(c["seconds"] / 60, 2)} for stage, c in costs.items()}})
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    write_json(RUNS, read_json(RUNS) | {name: str(path)})
    return path


def freeze():
    """Records the code, scoring and truth hashes in eval/frozen.json. Run it once the product code is final and every held-out truth is written, before any held-out run.
    A recorded code, scoring or held-out truth hash never changes. A held-out truth written later may join, and any other truth's hash is brought up to date."""
    old = read_json(FROZEN)
    names = {p.stem.removesuffix("_stairs") for p in TRUTH.glob("*.json")} | set(HELD_OUT) | {p.parent.name for p in PLANTS.glob("*/truth.json") if held_out(p.parent.name)}   # sets, so a stair truth joins its set's hash
    truth = {name: sha for name in sorted(names) if (sha := truth_hash(truth_paths(name)))}
    new = {"storey": code_hash()[0], "eval": eval_hash()}
    moved = [key for key in new if old.get(key) not in (None, new[key])] + [name for name in sorted(names) if held_out(name) and old.get("truth", {}).get(name) not in (None, truth.get(name))]
    if moved:
        raise ValueError(f"{FROZEN.name} already holds other hashes for {', '.join(moved)}. A held-out number must come from the code, scoring and truth frozen before its output was seen.")
    write_json(FROZEN, old | new | {"truth": old.get("truth", {}) | truth, "frozen": old.get("frozen") or time.strftime("%Y-%m-%d %H:%M %z")})


def main():
    ap = argparse.ArgumentParser(description="Run one evaluation set through the pipeline.")
    ap.add_argument("set", nargs="?")
    ap.add_argument("--data", type=Path, help="a scratch folder for the reviews, their ledger and a model cache per set")
    ap.add_argument("--cap", type=float, default=CAP_USD, help="USD for this run")
    ap.add_argument("--freeze", action="store_true", help="record the code, scoring and truth hashes that held-out runs need")
    a = ap.parse_args()
    if a.freeze:
        freeze()
        print(json.dumps(read_json(FROZEN), indent=1))
    elif a.set and a.data:
        print(run(a.set, a.data, a.cap))
    else:
        ap.error("name a set and --data, or pass --freeze")


if __name__ == "__main__":
    main()
