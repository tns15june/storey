"""Runs a review's stages in order. Each stage leaves one JSON file in the review's folder, so a rerun picks up where the last run stopped."""
import json
import logging
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .meta import BudgetReached, Meter

log = logging.getLogger(__name__)


def read_json(path: Path):
    """The parsed file, or {} if there is no such file."""
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")   # unique, because a stage's worker threads report progress at the same time
    tmp.write_text(json.dumps(obj, allow_nan=False))   # a browser's JSON parser throws on NaN and Infinity, so they must fail here, visibly
    tmp.replace(path)   # a poller must never read a half-written file


@dataclass
class Context:
    dir: Path
    meter: Meter
    sheets: list[dict]
    fresh: bool
    status: dict = field(default_factory=dict)

    def stage(self, name: str) -> dict:
        """The stored result, or {}. Stored results are JSON, so dict keys come back as strings and an index map is keyed by str(index)."""
        return read_json(self.dir / f"stage_{name}.json")

    def progress(self, step: str, done: int = 0, total: int = 0):
        for row in self.status.get("stages", ()):
            if row["state"] == "running":
                row.update(done=done, total=total)
        self.status.update(step=step, usd=self.meter.spent)
        write_json(self.dir / "status.json", self.status)


# Empty until the stage modules land. Order: read, floors, verticals, declared, vector, letter. evaluate and assemble run after them.
STAGES: list[tuple[str, str, Callable[[Context], dict]]] = []


def stage_rows(stages: list) -> list[dict]:
    return [{"name": name, "label": label, "state": "queued", "done": 0, "total": 0} for name, label, _ in stages]


def run(dir: Path, *, fresh: bool = False, stages: list | None = None) -> None:
    """Runs each stage that has not finished. A failed stage writes {"failed": message}, the next stage still runs, and a rerun retries it."""
    stages = STAGES if stages is None else stages
    status = read_json(dir / "status.json") | {"state": "running", "stages": stage_rows(stages)}
    # shortcut: a missing sheets.json means no sheets because the server tests' stubbed ingest writes none, raise here once those stubs write it.
    sheets = read_json(dir / "sheets.json") or []
    ctx = Context(dir, Meter(ledger=dir.parent / "ledger.jsonl"), sheets, fresh, status)   # one ledger for all review folders, so the daily cap sees them all
    # shortcut: spend after a dead process's last status write is not carried over, tag ledger rows by review if a resumed run must count it.
    ctx.meter._spent = status.get("usd", 0.0)   # the cap is per review, so a rerun goes on counting
    stale = fresh
    for row, (name, label, fn) in zip(status["stages"], stages):
        path = dir / f"stage_{name}.json"
        failed = "failed" in ctx.stage(name)
        if path.exists() and not failed and not stale:
            row["state"] = "done"
            continue
        stale = stale or failed   # later stages ran without this stage's result, so they must run again with it
        row["state"] = "running"
        ctx.progress(label)
        try:
            write_json(path, fn(ctx))
            row["state"] = "done"
        except Exception as e:
            # Only the cap's text is written for an examiner. Other exception text can quote a request or a reply, so neither the file nor the log keeps it.
            write_json(path, {"failed": str(e) if isinstance(e, BudgetReached) else f"{label} did not finish."})
            row["state"] = "failed"
            log.error("Stage %s failed with %s\n%s", name, type(e).__name__, "".join(traceback.format_tb(e.__traceback__)))
    status["state"] = "done"
    ctx.progress("Done")
