# Storey design

Written 31 Oct 2026, 15:30 IST. Author: the assistant, working solo with the user. The user opted out of design questions, so every choice here is the author's call unless marked as the user's. Measured numbers are in `docs/research/build-log.md`.

## What it is

Storey does the first pass of plan review for a city building department. An examiner uploads a permit set as a PDF or as images. A few minutes later Storey shows an inventory of the set, a stair and shaft matrix across floors, a draft correction list, a list of what it did not check, and a comment letter in the format cities already issue. On a resubmittal it shows which earlier comments look resolved, which look open, and what changed that nobody mentioned.

It comes from Y Combinator's Winter 2027 request for "AI for City Hall". It is an entry in the Meta Global AI Developer Hackathon, theme Push the Frontier, Indie track. Submission target is Wed 4 Nov 22:30 IST. The hard deadline is 07:30 IST on 5 Nov.

## Why

More than 80% of standard permit plans in Portland needed corrections. Seattle middle housing took a median 3.72 correction cycles and 161 days. A third of US code departments have one to four staff, and most examiners are near retirement. The real letters read in Round 1 mostly ask the applicant to show something or to correct a mismatch, so a tool that catches those before the examiner starts saves a cycle. Seattle's own pilot recommended completeness screening over compliance checking, and its published wrong-pass rate of 6.4% is the number to beat.

## Goals, in priority order

1. A real upload produces a real draft correction list end to end, with no human steps, on a set the product has not seen.
2. Every item carries the sheet, the location on the drawing, the measured or read value, the rule, and the evidence, so an examiner can verify it in seconds.
3. The product is wrong in the safe direction. It raises extra items rather than missing a real gap, and when it cannot tell it says so on screen.
4. The demo holds up live: a judge can remove a stair from a sheet and watch the finding appear, or change the jurisdiction and watch the list change.
5. The evaluation is published with its misses, including wrong passes.

## Non-goals

- Raster travel distance, dead-end length, door counts per room, exit door detection from plans alone, and tracker-based detection of added partitions. All were measured and failed. See the build log.
- Electrical, plumbing, mechanical, structural, zoning and energy review. Those sheets are inventoried and listed as not reviewed.
- Fine-tuning, and any integration with Accela, ProjectDox or Bluebeam.
- Applicant accounts, billing, multi-tenant login, and multi-building sets. One upload is one building.

## Anti-goals

These would make the product a failure even if everything else worked.

- Showing a pass, approval, or green tick for a building, a floor or a rule. The word "compliant" does not appear. The findings schema has no such value, so a model cannot return one.
- A number on a scan that looks measured but is not. Scans show "Cannot tell" for distance and width.
- A code section or value not traceable to a row in the shipped rule file. Models receive rule rows, never code prose, and no code sentence is displayed.
- A model deciding severity. Models read and locate. Code measures, compares and assigns severity.
- A silent failure. A stage that fails, runs out of budget or hits a low-confidence alignment leaves a visible item.
- Showing a drawing publicly when `public_demo` is not `yes` in the catalog.
- Sending customer drawings to a Contributor model variant.

## Who it serves

The user is a plan examiner at a city building department. The status line on every screen reads "Draft for examiner. Not an approval." An applicant pre-check uses the same engine and needs no extra work, but the product is written for the examiner.

## Checks that ship

Severity is fixed per check in code. Fix means the set contradicts itself or a rule on direct evidence. Check means the evidence points to a problem and a person must confirm. Cannot tell means the product could not measure it.

| # | Check | Input | Failure direction | Severity |
|---|---|---|---|---|
| 1 | Required sheets present: code summary, life safety plan, door schedule, floor plan per storey | Sheet read | Missed sheet raises a flag | Fix |
| 2 | Sheet index against sheets in the set, duplicates, unreadable sheets | Sheet read | Flag | Check |
| 3 | Floor label in title block against label under the drawing | Sheet read | Flag | Check |
| 4 | Code summary fields present: edition, occupancy, type, sprinklers, stories, area, occupant load | Spark on code sheet | Flag | Fix when absent |
| 5 | Stories declared against floor plans present; edition against the jurisdiction's adopted edition | Facts | Flag | Fix |
| 6 | Table arithmetic: room loads, totals, exit width against occupants, worked examples against tables | Text layer or Spark | Flag | Check |
| 7 | Door tags on plans against schedule rows, both ways; width and type against the drawn arc on vector sheets | Text, Spark strips, vector arcs | Flag | Check |
| 8 | Exit markers on life safety sheets: capacity sum after removing tile duplicates, against declared load | Spark tiles | Flag | Check |
| 9 | Stair and shaft continuity: gap between floors | Spark comparison, crops, ink | False absence raises a flag | Fix when all signals agree, otherwise Check |
| 10 | Stair does not reach the level of exit discharge | Same, plus the discharge level | Flag | Fix when all signals agree, otherwise Check |
| 11 | Stair count per floor against exits required for the declared load | Matrix, rule rows | Flag | Check |
| 12 | Every stair's top and bottom floor listed for the examiner | Matrix | Always shown | Open item |
| 13 | Travel distance on vector sheets, with the route drawn | Vector walls and arcs | Under-measures by 16 to 19% | Never Fix. Check when within 25% of the limit or over it. Shown as a minimum |
| 14 | Corridor clear width on vector sheets with wall layers or a dominant wall pen | Vector | Measured against printed value | Check |
| 15 | Door swing against occupant load on vector sheets | Vector arcs, load | Flag | Check |
| 16 | Cycle 2: each earlier comment rechecked at its location, plus unmentioned changes | Registered diff, SAM tracks | See below | Looks resolved, Still open, Cannot tell, New change |

Round 2 results may change checks 6 to 8 or their severities. The `declared` experiment sets the final list.

Checks are geometric or arithmetic. No check requires code text.

## How stair presence is decided

Round 1 refuted SAM as the presence signal and OR fusion of the two models. Spark floor comparison decides. Default rule until the `presence` experiment reports:

1. Spark compares all floor sheets in one call, three runs. Stairs are matched across runs by position after alignment.
2. If all three runs agree, the cell takes that value.
3. If they disagree, or any run says absent, a Spark crop question and an ink and tread check run at the registered location.
4. A gap or an end-of-stair finding is Fix only when Spark, the crop and the ink check agree. Any disagreement makes it Check.
5. SAM masks are drawn as evidence and give each floor's registration offset. SAM never decides presence, unless the `tightseed` experiment shows it can, in which case it joins step 4 as one more signal.

A floor that fails the alignment gate shows as unaligned, and its cells become Cannot tell.

## Architecture

Python 3.12, FastAPI, vanilla JS, no database. Everything for a review sits in one folder. Models: `muse-spark-1.3` for reading, `sam-3.1` for tracking, and `muse-wonder-1.0` only where the `agent` experiment shows it earns its price. Every runtime model is Meta's.

```
upload
  ingest        PDF or images to sheet PNGs, text layer words, vector or scan   (built)
  read          Spark low effort per sheet: number, title, class, level, scale, plan box
  floors        Joint Spark order, alignment, inlier gate, 960 px stack video
  verticals     Spark comparison, crops, ink, SAM masks, matrix
  declared      Code sheet, door schedule strips, life safety tiles, text layer tables
  vector        Walls, arcs, rooms, routes, scale chain. Vector sheets only
  evaluate      Facts plus rule rows to findings. Deterministic and free to rerun
  audit         Optional model check of overlays, per the agent experiment
  letter        Draft from findings. Adds no fact
  review.json   What the screen reads
```

Modules in `storey/`: `meta.py` (client, retries, cache, spend cap), `read.py`, `floors.py`, `verticals.py`, `declared.py`, `vector.py`, `rules.py`, `findings.py`, `letter.py`, `recheck.py`, `pipeline.py`. Code is ported from `lab/` and trimmed. The lab folder stays as the record of the experiments.

Stage rules:

- Each stage writes its result to `data/reviews/<id>/stage_<name>.json`. A rerun skips finished stages.
- Each model call is cached by a hash of model, prompt, schema, effort and image bytes in `data/cache/`. Reruns cost nothing. A live demo can bypass the cache with a flag.
- Every model call sets a token cap.
- Each review has a spend cap, default 8 USD, and the server has a daily cap, default 25 USD. Hitting a cap ends the stage with a visible Cannot tell item.
- A stage that throws does not stop the review. Later stages run on what exists, and the failed stage appears in Not checked.
- Status goes to `status.json` per stage with counts, polled by the browser every 2 seconds.

Facts and rules are separate. Stages write facts with their source (sheet, box, method). `evaluate` turns facts plus a jurisdiction into findings, and it makes no model call, so switching jurisdiction or the sprinkler assumption in the screen is instant.

## Rule data

`storey/rules/rows.json` holds only the rows the shipped checks use, each with section number, value, unit and jurisdiction overrides. The full research file stays in `data/rules/` and out of git. Titles are in our own words. Unknown sprinkler status uses the stricter value and raises an item asking the examiner. Jurisdictions shipped: Houston and Philadelphia, which matched the model code on all 144 checks, plus Seattle, whose 25 ft dead-end and other amendments show the list changing. Each item names the adopted code of that city. Occupant load calculations appear in the item so the examiner can see the rounding.

## Review data contract

The current `review.json` stays and gains these fields.

- `project`: `jurisdiction`, `adopted_code`, `discharge_level`, `facts` with sources.
- `sheets[]`: `number`, `class`, `kind`, `scale`, `reviewed` (bool), `reason`.
- `verticals[].floors[sheet]`: `state` (found, missing, check, unknown, end), `signals` (`spark` as n of 3, `crop`, `ink`, `sam`), `point`, `box`. `conf` stays as SAM's number but no longer drives the state.
- `findings[]`: `key` (a stable hash of check, sheet number and object), `comment_id` (discipline letter and number, kept across cycles), `check`, `discipline`, `state` for cycle 2.
- `routes[]`: `sheet`, `path`, `length`, `label`.
- `open_items[]`, `not_checked[]`, `cost` (USD and seconds per stage).

## Screens

1. **Start.** Upload, sample reviews, past reviews. Handles `#key=` by posting to `/api/session` and clearing the hash.
2. **Progress.** Stages with counts, thumbnails as sheets are read, spend so far.
3. **Review.** The existing title block and three panes, changed as follows.
   - Matrix cells read the `state` field. Dashed means Check. The legend reads Found, Missing, Check, Not on this floor.
   - Tabs in the right pane: Findings, Set, Open items, Not checked.
   - Jurisdiction selector and sprinkler selector in the title block, wired to the free re-evaluate call.
   - Routes drawn on the sheet with the label "Minimum. Examiner to confirm."
   - Keep and Drop decisions saved on the server.
   - A test tool, "Remove this stair from the sheet", which edits a copy of the sheet and reruns the stair stage. It is labelled as a test. It proves the check is live.
4. **Letter.** Draft in the real letter format, sheet first, imperative wording, section at the end, status per item with no approval box. Editable. Downloads as PDF.
5. **Second review.** Upload a resubmittal against a review. Each comment gets its state, with old and new crops side by side. Unmentioned changes are listed.
6. **How well it works.** The published evaluation, misses included, reading from the eval output.

Design language stays as built: a drawing-office title block, IBM Plex, red for the reviewer's pen.

## Second review

Sheets are matched by sheet number, then registered. For each earlier comment the location is compared in the new sheet. Erased, moved and exit door changes are detected. Added partitions are not, so a resolved state for those reads Cannot tell. Clouded areas are listed if `samtricks` finds a method for them. Comment IDs persist, and items carried forward keep their history, matching how Sonoma's letters mark them.

## Evaluation

`eval/` runs the real pipeline, not lab code.

- Held-out real sets, truth written first, one labeller, stated as such.
- Planted defects on real sheets with `lab/defects/plant.py`: stair gap, stair cut off above discharge, duplicate sheet, door tag changed, table value changed.
- Any real letter paired with its drawing set from the `realsets` experiment. This is the most valuable test available, and it is reported whether it flatters the product or not.
- Metrics, in this order: wrong passes, findings per set an examiner would keep, false findings per set, cost and minutes per set, and a baseline beside each.

## Demo sets

Public on screen: `habs_stelizabeths_center` (public domain, 5 floors, stairs and shafts), `mml_capitol` and `mml_baxter` (vector, schedules, real door tag and label findings). `nibs_clinic` supplies numbers only until WBDG gives permission. `bme_f` is not shown, because its date tag cannot be trusted. Replace the sample review, which still uses `bme_f`, with real pipeline output on `habs_stelizabeths_center`. The `realsets` experiment may add a better set.

## Hackathon use of Meta tooling

- SAM 3.1 video tracker reads a stack of floor plans as frames: registration across floors, masks as evidence, tracked changes between cycles.
- Spark multi-image comparison across a building, grounding, structured output, tiled reading of fine text.
- A findings schema with no pass value, enforced by constrained decoding.
- If it earns its place: Wonder reviewing tool results that return overlay images.
- Stretch: Glimmer running locally so a city's drawings stay on its own machine. Test on the 24 GB laptop before any claim.

## Constraints

- Budget: 148 USD total, 29 spent after Round 1, Round 2 capped so the ledger stops at 75. Reserve 25 USD for the final evaluation and demo rehearsals.
- Repo `tns15june/storey`, private until submission, branch `build`. Commit often, because the platform flags single-commit repos. Scan staged files for secrets.
- Never print the API key. Security rules from the first three reviews stay in force: the API always needs a key, uploads are bounded, PDFs are read in a child process.
- Video at most 3 minutes, in English, link opens without login. README has "Built during the hackathon" and "Pre-existing assets" sections and a licence.
- Public demo stays on free, cached sample reviews. Uploads need the access key.

## Left to the builder

File layout inside modules, test structure, exact prompts (start from the lab prompts that scored), concurrency numbers, CSS details, the PDF letter library if one is already installed, otherwise the browser's print view.

## Schedule

| When | Done means |
|---|---|
| Sat 18:00 | Scope fixed. This document. |
| Sun night | One building end to end from upload to letter in the real app |
| Mon night | All demo sets, second review, test tool, evaluation harness |
| Tue night | Numbers frozen, video recorded, README |
| Wed 22:30 | Repo public, `build` merged to `main`, submitted |
