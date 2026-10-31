# Storey Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn an uploaded drawing set into a draft correction list, stair and shaft matrix, comment letter and second-review recheck in a working web app, with a published evaluation.

**Architecture:** A stage pipeline writes one JSON file per stage into the review's folder. Models read and locate, deterministic code compares and sets severity, and `evaluate` turns stored facts plus a jurisdiction into findings with no model call. The existing FastAPI server and vanilla JS review screen stay.

**Tech Stack:** Python 3.12, FastAPI, PyMuPDF, OpenCV, NumPy, SciPy, shapely, openai client against `https://api.meta.ai/v1`, meta-sam-parser, pytest, vanilla JS.

**Spec:** `docs/superpowers/specs/2026-10-31-storey-design.md`

## Global Constraints

- Status line on every screen: "Draft for examiner. Not an approval."
- No pass, approve or compliant value anywhere in schemas, JSON, UI or letter. Findings states are exactly `fix`, `check`, `unknown`. Recheck states are exactly `resolved_looks`, `open`, `unknown`, `new_change`.
- Severity is set in code, never by a model.
- Every model call sets a token cap, default 24000.
- Per-review spend cap default 8 USD, server daily cap default 25 USD. Hitting one ends the stage with a visible `unknown` item.
- Runtime models are Meta only: `muse-spark-1.3`, `sam-3.1`, `muse-wonder-1.0` if the agent experiment supports it. Never a Contributor variant.
- Never read or print `.env`, `meta_api_key.rtf` or the key. Load the key inside `storey/meta.py` only, from `MODEL_API_KEY` in the environment or `.env`.
- Do not weaken `storey/server.py` guards: API key, cross-site check, upload limits, child process ingest.
- Rule values shipped: only rows used by shipped checks, in `storey/rules/rows.json`. Never code sentences.
- Drawings shown publicly must have `public_demo: yes` in `data/plans/catalog.json`.
- Commit each task on branch `build`, end commit messages with the attribution lines in the session reminder, and scan staged files for keys.
- Tests never call the network. Model calls are stubbed by injecting a fake into `storey.meta`.
- Plain, direct prose in docs and UI text. No metaphors.

## Review Focus

1. A set with no floor plans, or one floor plan: stair stage leaves a `Not checked` entry and the review still finishes.
2. A floor that fails the alignment gate: its cells show `unknown`, and no gap finding is raised on it.
3. A budget stop in the middle of a stage: a visible `unknown` item, a finished review, nothing lost.
4. A malformed or empty model response on one sheet: that sheet is marked unread and the other sheets proceed.
5. A second upload whose sheet numbers do not match the first: unmatched sheets are listed and no comment is called resolved.

Each has a named test in its owning task below.

---

## File Structure

```
storey/
  meta.py        client, cache, Meter, spark(), sam_video(), sam_image()
  pipeline.py    stage runner, status, resume
  read.py        per-sheet read
  floors.py      floor selection, order, alignment, stack video
  verticals.py   presence decision, crop checks, ink check, SAM masks
  declared.py    code sheet, door schedule, life safety markers, table arithmetic
  vector.py      walls, arcs, rooms, routes, dimensions (vector sheets)
  rules.py       rule lookups by jurisdiction
  rules/rows.json
  findings.py    all checks, severity table, keys, comment ids
  assemble.py    review.json from stages plus findings
  letter.py      letter text from findings
  recheck.py     second review
  testtool.py    remove a stair from a sheet
  server.py, ingest.py, __main__.py   existing
app/web/         existing, plus start.html, start.js, progress in app.js, tabs, letter, recheck
eval/
  run.py         runs the pipeline over the eval sets
  plant.py       planted defects, wraps lab/defects/plant.py logic
  truth/         one JSON per set
  report.py      writes data/eval/report.json and app/web/eval.json
tests/           one file per module
```

## Execution order

Task 1 and Task 2 land first, alone, because everything else imports them. Tasks 3 to 10 touch separate files and run in parallel. Tasks 11 to 13 integrate. Tasks 5 and 6 wait for the `presence`, `tightseed` and `declared` experiment results, whose decision rules and check lists they implement.

---

### Task 1: Meta client, cache and spend control

**Files:**
- Create: `storey/meta.py`
- Test: `tests/test_meta.py`

**Interfaces:**
- Produces:
  - `class BudgetReached(RuntimeError)`
  - `class Meter: __init__(self, cap_usd: float = 8.0, daily_cap_usd: float = 25.0, ledger: Path = DATA/"ledger.jsonl"); add(self, usd: float, **fields); spent: float (property); check(self) -> None` raises `BudgetReached`
  - `spark(prompt: str, images: Sequence = (), *, meter: Meter, schema: dict | None = None, effort: str = "low", model: str = "muse-spark-1.3", system: str | None = None, max_tokens: int = 24000, fresh: bool = False) -> dict | str`
  - `sam_video(prompt: str, path: Path, frames: int, *, meter: Meter, direction: str = "both") -> list[dict]`, each dict `{"object": int, "frame": int, "box": [x1,y1,x2,y2], "conf": float}`
  - `sam_image(prompt: str, image, *, meter: Meter) -> list[dict]`, each dict `{"object": int, "box": [...], "conf": float, "mask": np.ndarray}`
  - `CACHE: Path`, default `data/cache`, env `STOREY_CACHE`
  - `set_transport(fake)` test seam that replaces the HTTP call

- [ ] **Step 1: Write the failing tests** in `tests/test_meta.py`:
  - `test_cache_hit_costs_nothing`: two identical `spark` calls with a fake transport, transport called once, `meter.spent` unchanged after the second.
  - `test_token_cap_always_sent`: the fake sees `max_completion_tokens == 24000`.
  - `test_budget_stops_before_call`: meter at cap raises `BudgetReached`, transport not called.
  - `test_daily_cap_counts_other_reviews`: ledger with 25 USD of today's rows raises `BudgetReached`.
  - `test_malformed_json_is_retried_once_then_raises_valueerror`.
- [ ] **Step 2: Run:** `.venv/bin/pytest tests/test_meta.py -v`. Expected: import error.
- [ ] **Step 3: Implement** by porting `_retry`, `data_url`, `_sam`, `_masks`, `raster` from `lab/metalab.py`. Cache key is sha256 of model, effort, system, prompt, schema JSON and each image's bytes. Encode images as PNG or pass file bytes through. Request `mask_encoding: one_bit` on SAM. SAM `object_id` is a string, so convert with `int()`. Upload video from bytes, not an open handle. Spark `usd` uses prices `(1.25, 0.15, 4.25)` and Wonder `(4.0, 0.20, 20.0)` per million input, cached, output. SAM image 0.0025 USD. SAM video `max(frames, 50) * 0.0002` USD. Key read order: `MODEL_API_KEY` env, then `.env` in the repo root.
- [ ] **Step 4: Run:** `.venv/bin/pytest tests/test_meta.py -v`. Expected: 5 passed.
- [ ] **Step 5: Commit.**

### Task 2: Stage pipeline

**Files:**
- Create: `storey/pipeline.py`
- Modify: `storey/server.py:132-143` so `_run` calls `pipeline.run` after ingest
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `Meter`, `BudgetReached`.
- Produces:
  - `@dataclass class Context: dir: Path; meter: Meter; sheets: list[dict]; fresh: bool; def stage(self, name: str) -> dict` returns a stored stage result or `{}`; `def progress(self, step: str, done: int = 0, total: int = 0)`
  - `STAGES: list[tuple[str, str, Callable[[Context], dict]]]`, entries `(name, label, fn)` in the order read, floors, verticals, declared, vector, letter. `evaluate` and `assemble` run after them.
  - `run(dir: Path, *, fresh: bool = False, stages: list | None = None) -> None`
  - A stage that throws or hits `BudgetReached` writes `stage_<name>.json` as `{"failed": "<plain message>"}` and the run continues.
  - `status.json` fields: `state` (queued, running, done, failed), `step`, `stages` (list of `{name, label, state, done, total}`), `usd`.

- [ ] **Step 1: Write the failing tests:** `test_stages_run_in_order`, `test_finished_stage_is_skipped_on_rerun`, `test_failed_stage_does_not_stop_later_stages`, `test_budget_stop_is_recorded_and_review_finishes`, `test_status_has_each_stage`.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_pipeline.py -v`. Expected: import error.
- [ ] **Step 3: Implement.** Use the atomic write already in `server._status`, moved to `pipeline.write_json(path, obj)`.
- [ ] **Step 4: Run** the pipeline tests and `tests/test_server.py`. Expected: all pass, and the existing server tests are updated only where `status.json` gains fields.
- [ ] **Step 5: Commit.**

### Task 3: Sheet read

**Files:**
- Create: `storey/read.py`
- Test: `tests/test_read.py`

**Interfaces:**
- Consumes: `spark`, `Context`.
- Produces: `read_stage(ctx) -> dict` returning `{"sheets": [{"index": int, "number": str, "title": str, "cls": str, "level": str, "scale": str, "plan_box": [x1,y1,x2,y2], "read_ok": bool}]}`. `cls` is one of the `sheet_class` values in `lab/bakeoff/e1_meta.py`. `plan_box` is in sheet pixels.

- [ ] **Step 1: Write the failing tests:** `test_prompt_uses_2000px_image`, `test_grid_coordinates_map_to_pixels` (500 on a 0 to 1000 grid of a 4000 px wide sheet maps to x 2000), `test_empty_or_bad_reply_marks_sheet_unread`, `test_parallel_calls_keep_sheet_order`.
- [ ] **Step 2: Run.** Expected: import error.
- [ ] **Step 3: Implement.** Merge the schemas and prompts of `lab/bakeoff/e1_meta.py` and `lab/align/sheet_read.py` into one call at effort low, on a copy with long edge 2000 px. Eight concurrent calls.
- [ ] **Step 4: Run.** Expected: 4 passed.
- [ ] **Step 5: Live check** on `data/plans/mml_capitol/THE-CAPITOL.pdf` under `LAB`-free meter with cap 1 USD. Expected: all 22 sheet numbers from `CAP_TRUTH` in `lab/bakeoff/e3_pdf_floors.py`.
- [ ] **Step 6: Commit.**

### Task 4: Floors

**Files:**
- Create: `storey/floors.py`
- Test: `tests/test_floors.py`

**Interfaces:**
- Consumes: `read_stage` output, `spark`.
- Produces:
  - `floors_stage(ctx) -> dict` returning `{"order": [sheet index bottom to top], "reference": int, "transforms": {index: 3x3 list}, "inliers": {index: int}, "unaligned": [index], "stack": "stack.mp4", "frame_size": [w, h], "discharge": index | None}`
  - `register(ref_img, mov_img, ref_box, mov_box) -> tuple[np.ndarray, int]`
  - `warp_point(H, x, y) -> tuple[float, float]`
  - `ink(img) -> np.ndarray`, ported from `lab/align/reg.py`
  - `write_stack(frames: list[np.ndarray], path: Path) -> None`, H.264, one frame per floor, long side 960 px

- [ ] **Step 1: Write the failing tests:**
  - `test_register_recovers_known_transform`: a synthetic plan moved by `sim(0.93, 1.5, 40, -25)`, error under 2 px at the box corners.
  - `test_wrong_building_fails_gate`: two unrelated synthetic plans give fewer than 25 inliers and the floor lands in `unaligned`.
  - `test_single_floor_plan_gives_no_stack`: the stage returns `{"order": [i], "stack": None}` with a `not_checked` note.
  - `test_no_floor_plans_stage_note`: no sheets of class floor plan, stage returns a `not_checked` note.
  - `test_discharge_prefers_ground_label`: labels `Basement, Ground, Level 1` give the Ground index.
- [ ] **Step 2: Run.** Expected: import error.
- [ ] **Step 3: Implement.** Port `coarse`, `matches`, `blocks` from `lab/align/reg.py`. Fall back to the wide search of `lab/align/synth.py` below 25 inliers, then mark unaligned. Joint order uses the prompt in `lab/align/sheet_read.py` and sends sheets shuffled. Never use a model's rank integer. Reference floor is the one with the most inliers to its neighbours, or the first if there is a tie.
- [ ] **Step 4: Run.** Expected: 5 passed.
- [ ] **Step 5: Live check** on `habs_stelizabeths_center`. Expected: order matches the catalog, 0 unaligned, overlay file saved to the review folder and viewed.
- [ ] **Step 6: Commit.**

### Task 5: Verticals

Waits for the `presence` and `tightseed` results. The decision rule comes from those reports. The default rule in the spec applies if they do not change it.

**Files:**
- Create: `storey/verticals.py`
- Test: `tests/test_verticals.py`

**Interfaces:**
- Consumes: `floors_stage` output, `spark`, `sam_video`, `register`, `ink`.
- Produces:
  - `verticals_stage(ctx) -> dict` returning `{"verticals": [{"id": "A", "kind": "stair"|"shaft"|"elevator", "floors": {index: {"state": str, "signals": {"spark": "3/3", "crop": bool|None, "ink": float|None, "sam": bool|None}, "point": [x, y]|None, "box": [..]|None}}}]}`. `state` is one of `found`, `missing`, `check`, `unknown`, `end`.
  - `decide(runs: list[bool], crop: bool | None, ink_ratio: float | None, aligned: bool) -> str`, the pure rule.

- [ ] **Step 1: Write the failing tests** for `decide` as a table: all runs present gives `found`, all absent with crop false and low ink gives `missing`, mixed gives `check`, unaligned gives `unknown`. Then `test_stairs_matched_across_runs_by_position`, `test_gap_needs_found_on_both_sides`, `test_top_and_bottom_become_end`, `test_sixteen_object_limit_splits_requests`, `test_unaligned_floor_raises_no_gap`.
- [ ] **Step 2: Run.** Expected: import error.
- [ ] **Step 3: Implement.** Comparison prompt and schema from `lab/bakeoff/e3_pdf_floors.py` (`FLOOR_P`, `FLOOR_S`), extended to shafts and elevators per the `presence` result. Ink check from `lab/defects/hybrid.py`.
- [ ] **Step 4: Run.** Expected: all pass.
- [ ] **Step 5: Live check** on `habs_stelizabeths_center` and `bme_k`. Compare to `lab/seed/truth.json` and `lab/audit/truth.json`. Expected: no wrong pass on absent floors.
- [ ] **Step 6: Commit.**

### Task 6: Declared reads

Waits for the `declared` result, which fixes the check list and prompts.

**Files:**
- Create: `storey/declared.py`
- Test: `tests/test_declared.py`

**Interfaces:**
- Consumes: `read_stage` output, `spark`.
- Produces: `declared_stage(ctx) -> dict` returning:
  - `code`: `{field: {"value": str, "sheet": int, "box": [..]|None}}` for `edition, occupancy, construction, sprinklers, stories, area, occupant_load`
  - `index`: `[{"number": str, "title": str}]`
  - `doors`: `[{"mark": str, "width_in": float|None, "type": str, "sheet": int}]`
  - `exits`: `[{"sheet": int, "point": [x, y], "width_mm": float|None, "capacity": int|None}]`, tile duplicates removed
  - `tables`: `[{"sheet": int, "kind": str, "rows": [...]}]`
  - `labels`: `[{"sheet": int, "title_block": str, "under_drawing": str}]`

- [ ] **Step 1: Write the failing tests:** `test_text_layer_used_before_model`, `test_schedule_strips_repeat_header`, `test_exit_tiles_deduplicated_before_summing` (two points 5 px apart with the same label count once), `test_missing_code_sheet_returns_empty_code`.
- [ ] **Step 2: Run.** Expected: import error.
- [ ] **Step 3: Implement** from the `declared` experiment code and `lab/bakeoff/e2_tasks.py`.
- [ ] **Step 4: Run.** Expected: all pass.
- [ ] **Step 5: Commit.**

### Task 7: Vector geometry

**Files:**
- Create: `storey/vector.py`
- Test: `tests/test_vector.py`, fixture `tests/fixtures/vector_room.pdf` written by the test with PyMuPDF

**Interfaces:**
- Consumes: sheet records with `kind == "vector"`, verticals, declared exits.
- Produces: `vector_stage(ctx) -> dict` returning `{"sheets": {index: {"scale_in_per_pt": float|None, "scale_checked": int, "doors": [{"point": [x,y], "width_in": float, "swing": str, "hinge": [x,y]}], "routes": [{"path": [[x,y],...], "length_ft": float, "from_room": str, "to_exit": str}], "corridors": [{"point": [x,y], "clear_in": float, "printed_in": float|None}], "skipped": str|None}}}`

- [ ] **Step 1: Write the failing tests:**
  - `test_arc_found_with_width`: a PDF with a 36 in door arc at known scale gives a door at 36 in within 0.5.
  - `test_scale_chain_rejects_bad_scale`: a dimension string disagreeing by over 1 in returns `scale_in_per_pt` None and a reason.
  - `test_route_stays_inside_walkable`: a two-room plan with one door, the route passes the door.
  - `test_scan_sheet_is_skipped_with_reason`.
- [ ] **Step 2: Run.** Expected: import error.
- [ ] **Step 3: Implement** by porting `lab/walls/vec.py`, `lab/doors/vec_arcs.py`, `lab/modern/vec.py`, `lab/modern/measure.py`, `lab/modern/halls.py`, `lab/walls/truepath_lib.py`. Wall pen chosen by the dominant stroke-width rule, then checked by reach. Stair shafts closed except at detected doors. Vector route is reported as a minimum.
- [ ] **Step 4: Run.** Expected: all pass.
- [ ] **Step 5: Live check** on `nibs_clinic` sheets 052 and 059 against `lab/walls/clinic4.py` figures, with exits from the life safety read rather than the IFC. Report the change in numbers.
- [ ] **Step 6: Commit.**

### Task 8: Rules

**Files:**
- Create: `storey/rules.py`, `storey/rules/rows.json`
- Test: `tests/test_rules.py`

**Interfaces:**
- Produces:
  - `rule(id: str, jurisdiction: str, **inputs) -> {"value": ..., "unit": str, "section": str, "adopted": str, "title": str}`
  - `JURISDICTIONS: dict[str, str]`, keys `houston`, `philadelphia`, `seattle`, label is the adopted code's name
  - `stricter_when_unknown(id, jurisdiction, **inputs)`, the stricter of the sprinklered and unsprinklered values

- [ ] **Step 1: Write the failing tests:** `test_dead_end_seattle_25_houston_20`, `test_travel_distance_b_sprinklered`, `test_unknown_sprinkler_uses_stricter`, `test_every_row_has_section_value_unit`, `test_no_row_contains_a_sentence_over_12_words`.
- [ ] **Step 2: Run.** Expected: import error.
- [ ] **Step 3: Write** `rows.json` from `data/rules/ibc2021_egress.json`, keeping only rows used by Task 9's checks: travel distance limits, exits by occupant load, corridor width, door width, swing threshold, exit width factors, stair discharge reference.
- [ ] **Step 4: Run.** Expected: 5 passed.
- [ ] **Step 5: Commit.**

### Task 9: Findings

**Files:**
- Create: `storey/findings.py`
- Test: `tests/test_findings.py`

**Interfaces:**
- Consumes: stage dicts from Tasks 3 to 7, `rule`.
- Produces:
  - `evaluate(stages: dict, jurisdiction: str, sprinklers: str | None) -> dict` returning `{"findings": [...], "open_items": [...], "not_checked": [...]}`. No model call, no file access.
  - Each finding: `{"key": str, "check": int, "severity": "fix"|"check"|"unknown", "discipline": "A", "sheet": int|None, "box": [..]|None, "title": str, "measured": str, "required": str, "rule": str, "evidence": str, "vertical": str|None}`
  - `SEVERITY: dict[int, str]`, the table from the spec
  - `assign_ids(findings, previous: dict[str, str]) -> None`, keeps a comment id for a known `key`, otherwise issues the next `A-01` style id

- [ ] **Step 1: Write the failing tests,** one per check in the spec's table, 1 to 15, each from a small hand-built stages dict. Also: `test_no_pass_value_anywhere` (no field in a full evaluate output contains pass, approved or compliant), `test_failed_stage_appears_in_not_checked`, `test_route_never_fix`, `test_scan_distance_is_unknown`, `test_jurisdiction_changes_the_list`, `test_key_is_stable_across_runs`.
- [ ] **Step 2: Run.** Expected: import error.
- [ ] **Step 3: Implement.** One small function per check, registered in a list. Margin on route check: raise when route length is at least 75% of the limit, since routes run 16 to 19% under the architect's figures.
- [ ] **Step 4: Run.** Expected: all pass.
- [ ] **Step 5: Commit.**

### Task 10: Letter

**Files:**
- Create: `storey/letter.py`
- Test: `tests/test_letter.py`

**Interfaces:**
- Consumes: findings, `spark` or Wonder per the `agent` result, project facts.
- Produces: `letter_stage(ctx) -> dict`, `draft(findings, project) -> {"header": {...}, "items": [{"comment_id": str, "sheet": str, "text": str}]}`, `no_new_facts(text: str, finding: dict) -> bool`.

- [ ] **Step 1: Write the failing tests:** `test_deterministic_draft_without_model` (the letter builds from findings if the model fails), `test_new_number_is_rejected` (`no_new_facts` is False when the text holds a number absent from the finding), `test_no_approval_box`, `test_grouped_by_sheet`.
- [ ] **Step 2: Run.** Expected: import error.
- [ ] **Step 3: Implement.** The template wording comes first. The model only rewrites wording, in the style of `data/lab/gather_practice/examples/1_sonoma_comment_letter_1_commercial_BLD19-3421.txt`. Reject a rewrite that fails `no_new_facts` and keep the template text.
- [ ] **Step 4: Run.** Expected: all pass.
- [ ] **Step 5: Commit.**

### Task 11: Assemble and serve

**Files:**
- Create: `storey/assemble.py`
- Modify: `storey/server.py`, `storey/pipeline.py`
- Test: `tests/test_assemble.py`, `tests/test_server.py`

**Interfaces:**
- Consumes: all stages, `evaluate`, `assign_ids`.
- Produces:
  - `assemble(dir: Path, jurisdiction: str = "houston", sprinklers: str | None = None) -> dict`, writes `review.json` per the spec's contract. Sheet images resized to long edge 2000 px and saved as JPEG in the review folder. Image URLs are `/api/reviews/<id>/files/<name>`.
  - `POST /api/reviews/{id}/evaluate {jurisdiction, sprinklers}` returns the new review without any model call.
  - `POST /api/reviews/{id}/decisions {key, decision}`, stored in `decisions.json`.
  - `POST /api/reviews/{id}/recheck` multipart resubmittal, returns a new review id with `parent` set.
  - `POST /api/reviews/{id}/remove {vertical, sheet}`, calls Task 13.
  - `GET /api/reviews`, list of past reviews.

- [ ] **Step 1: Write the failing tests:** `test_review_json_matches_contract`, `test_evaluate_makes_no_model_call` (fake transport counts zero), `test_decision_is_saved`, `test_key_required_on_new_routes`, `test_cross_site_post_refused_on_new_routes`.
- [ ] **Step 2: Run.** Expected: failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full suite. Expected: all pass.
- [ ] **Step 5: End-to-end live check:** upload `mml_capitol` through the API, wait for `done`, open `review.json`, and report cost and minutes. This is the Sunday checkpoint.
- [ ] **Step 6: Commit.**

### Task 12: Screens

**Files:**
- Create: `app/web/start.html`, `app/web/start.js`
- Modify: `app/web/app.js`, `app/web/style.css`, `app/web/body.html`, `app/web/index.html`
- Test: `tests/test_web.py` (static contract checks), and a browser check by the `run` skill

**Interfaces:**
- Consumes: Task 11 routes and the contract.

- [ ] **Step 1:** Add `#key=` handling on load: post to `/api/session`, then clear the hash with `history.replaceState`.
- [ ] **Step 2:** Build Start: upload, sample reviews, past reviews.
- [ ] **Step 3:** Build Progress: polls `status.json` every 2 seconds, shows stages, counts, thumbnails, spend, and failure messages.
- [ ] **Step 4:** Change the matrix to read `state`. Legend: Found, Missing, Check, Not on this floor. Tooltip shows signals.
- [ ] **Step 5:** Add tabs Findings, Set, Open items, Not checked; jurisdiction and sprinkler selectors; routes drawn with the label "Minimum. Examiner to confirm."
- [ ] **Step 6:** Save Keep and Drop to the server. Letter view reads from `letter` output, editable, and uses the browser print view for PDF.
- [ ] **Step 7:** Second review screen: comment states, old and new crops side by side.
- [ ] **Step 8:** Test tool button on a Found cell, labelled "Test: remove this stair from the sheet".
- [ ] **Step 9:** How well it works page reading `eval.json`.
- [ ] **Step 10:** `tests/test_web.py`: the status line is on every page, no page contains "approved", "compliant" or "pass" as a state. Run: expected pass.
- [ ] **Step 11:** Drive it with the `run` skill on desktop and at phone width, in light and dark, and fix what is broken.
- [ ] **Step 12: Commit.**

### Task 13: Second review and test tool

**Files:**
- Create: `storey/recheck.py`, `storey/testtool.py`
- Test: `tests/test_recheck.py`, `tests/test_testtool.py`

**Interfaces:**
- Consumes: `register`, `sam_video`, `assign_ids`, Task 5, `lab/defects/plant.py` logic.
- Produces:
  - `recheck_stage(ctx) -> dict` returning `{"comments": [{"comment_id": str, "state": str, "old_crop": str, "new_crop": str, "why": str}], "changes": [{"sheet": int, "box": [..], "note": str}], "unmatched_sheets": [str]}`
  - `remove_stair(dir: Path, vertical: str, sheet: int) -> Path`, returns a new review folder with the edited sheet, reusing finished stages except verticals, evaluate and letter

- [ ] **Step 1: Write the failing tests:** `test_changed_location_looks_resolved`, `test_unchanged_location_stays_open`, `test_unmatched_sheet_is_listed_and_not_resolved`, `test_added_partition_stays_unknown`, `test_comment_ids_persist`, `test_remove_stair_leaves_original_untouched`, `test_removal_produces_a_gap_finding` (stubbed Spark).
- [ ] **Step 2: Run.** Expected: import error.
- [ ] **Step 3: Implement.** Registered ink difference per comment location, plus SAM tracks across the two versions where available. Use the thresholds from `lab/defects` revision results and re-test them in Task 14 on held-out cases, because they were set after seeing data.
- [ ] **Step 4: Run.** Expected: all pass.
- [ ] **Step 5: Live check** of the test tool on `habs_stelizabeths_center`: remove one stair on a middle floor, confirm the finding appears, report seconds and USD.
- [ ] **Step 6: Commit.**

### Task 14: Evaluation

**Files:**
- Create: `eval/run.py`, `eval/plant.py`, `eval/report.py`, `eval/truth/*.json`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `pipeline.run`, findings.
- Produces: `data/eval/report.json`, `app/web/eval.json`.

- [ ] **Step 1: Write the failing tests:** `test_wrong_pass_counted_first`, `test_baseline_beside_each_score`, `test_miss_listed_by_name`.
- [ ] **Step 2: Run.** Expected: import error.
- [ ] **Step 3: Implement.** Freeze truth files and record their sha256 before any run. Sets, defects and metrics per the spec. Budget 25 USD.
- [ ] **Step 4: Run** the tests. Expected: pass.
- [ ] **Step 5: Run the evaluation** after Tasks 5, 6, 9 and 13. Commit the report and the list of misses.

### Task 15: Ship

**Files:**
- Create: `README.md`, `LICENSE`
- Modify: `app/web/sample/*`, `.gitignore`

- [ ] **Step 1:** Replace the sample with real pipeline output on `habs_stelizabeths_center`, and delete `app/web/sample/{L2,L1,L0,B1}.jpg`.
- [ ] **Step 2:** Replace absolute `/Users/tsharma/...` paths in `lab/` with paths relative to the repo root, and confirm one lab script still runs.
- [ ] **Step 3:** Write the README: what it is, how to run, evaluation with misses, Meta tooling, "Built during the hackathon", "Pre-existing assets", licence.
- [ ] **Step 4:** Confirm no file in the repo holds a key, IBC tables beyond `rows.json`, or a sheet without `public_demo: yes`.
- [ ] **Step 5:** Record the video, 3 minutes at most, with the link checked signed out.
- [ ] **Step 6:** Make the repo public, merge `build` to `main`, and submit.
