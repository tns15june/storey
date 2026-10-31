# Storey design

Written 31 Oct 2026, 15:30 IST, updated 19:00 IST with the Round 2 results. Author: the assistant, working solo with the user. The user opted out of design questions, so every choice here is the author's call unless marked as the user's. Measured numbers are in `docs/research/build-log.md`.

## What it is

Storey does the first pass of plan review for a city building department. An examiner uploads a permit set as a PDF or as images. A few minutes later Storey shows an inventory of the set, a stair matrix across floors, a draft correction list, a list of what it did not check, and a comment letter in the format cities already issue. On a resubmittal it shows which earlier comments look resolved, which look open, and what changed that nobody mentioned.

It comes from Y Combinator's Winter 2027 request for "AI for City Hall". It is an entry in the Meta Global AI Developer Hackathon, theme Push the Frontier, Indie track. Submission target is Wed 4 Nov 22:30 IST. The hard deadline is 07:30 IST on 5 Nov.

## Why

More than 80% of standard permit plans in Portland needed corrections. Seattle middle housing took a median 3.72 correction cycles and 161 days. A third of US code departments have one to four staff, and most examiners are near retirement. The real letters read in Round 1 mostly ask the applicant to show something or to correct a mismatch, so a tool that catches those before the examiner starts saves a cycle. Seattle's own pilot recommended completeness screening over compliance checking, and its published wrong-pass rate of 6.4% is the number to beat.

## Goals, in priority order

1. A real upload produces a real draft correction list end to end, with no human steps, on a set the product has not seen.
2. Every item carries the sheet, the location on the drawing, the measured or read value, the rule, and the evidence, so an examiner can verify it in seconds.
3. The product is wrong in the safe direction. It raises extra items rather than missing a real gap, and when it cannot tell it says so on screen.
4. The demo holds up live. In the video and at the finals, a stair is removed from a sheet and the finding appears, and a change of jurisdiction changes the rule values and any item that depends on them.
5. The evaluation is published with its misses, including wrong passes.

## Non-goals

- Raster travel distance, dead-end length, door counts per room, exit door detection from plans alone, and tracker-based detection of added partitions. All were measured and failed. See the build log.
- SAM as evidence that a stair is present, absent or unchanged. With accurate seeds and the treads erased, 14 of 20 masks stayed on the empty room at the same confidence. A replaced door symbol stayed tracked in 3 of 3 runs.
- Elevators and shafts as a claim. No run listed 14 of 21 present cells, and none were planted. They are listed in Not checked.
- An agent driving the review. It passed one bar and failed the other in every configuration, and two runs dropped a real discrepancy. Spark video input scored 12 of 59 against 50 of 59 for always-present.
- Electrical, plumbing, mechanical, structural, zoning and energy review. Those sheets are inventoried and listed as not reviewed.
- Fine-tuning, and any integration with Accela, ProjectDox or Bluebeam.
- Applicant accounts, billing, multi-tenant login, and multi-building sets. One upload is one building.

## Anti-goals

These would make the product a failure even if everything else worked.

- Showing a pass, approval, or green tick for a building, a floor or a rule. The word "compliant" does not appear. The findings schema has no such value, so a model cannot return one.
- A number on a scan that looks measured but is not. Scans show "Cannot tell" for distance and width.
- A code section or value not traceable to a row in the shipped rule file. Models receive rule rows, never code prose, and no code sentence is displayed.
- A model deciding severity. Models read and locate. Code measures, compares and assigns severity.
- A silent failure. A stage that fails, runs out of budget or hits a low-confidence alignment leaves a visible item. An empty model reply is Cannot tell. A stair seen by only one run stays on the matrix as Not followed.
- A model removing a finding. An optional audit can only add a disputed tag.
- Showing a drawing publicly when `public_demo` is not `yes` in the catalog.
- Sending customer drawings to a Contributor model variant.

## Who it serves

The user is a plan examiner at a city building department. The status line on every screen reads "Draft for examiner. Not an approval." An applicant pre-check uses the same engine and needs no extra work, but the product is written for the examiner.

## Checks that ship

Severity is fixed per check in code. Fix means the set contradicts itself or a rule on direct evidence. Check means the evidence points to a problem and a person must confirm. Cannot tell means the product could not measure it.

| # | Check | Input | Failure direction | Severity |
|---|---|---|---|---|
| 1 | Required sheets present: code summary, life safety plan, door schedule, floor plan per storey | Sheet read | Missed sheet raises a flag | Fix |
| 2 | Sheet index against the set: sheet not on the index, index row with no sheet (rows marked not used are skipped), one number listed twice | Text layer, else Spark title blocks | Flag | Fix. A number listed twice is Check |
| 3 | Floor named in the title block against the label under the drawing, by a floor-word test in code. Other title differences only when Spark says different in both reads | Text layer or Spark | Flag | Check |
| 4 | Code summary fields present and consistent: edition, occupancy, type, sprinklers, stories, area, occupant load | Text layer or Spark on the code sheet | Flag | Check |
| 5 | Stories declared against floor plans present; edition against the jurisdiction's adopted edition | Facts | Flag | Fix |
| 6 | Table arithmetic: room loads, totals, exit width against occupants, worked examples against tables. The item states the direction: a load rounded down is unsafe, a load over-declared is safe | Text layer or two equal Spark reads | Flag | Check |
| 7 | Door and window tags on plans against schedule rows, both ways; schedule type against the drawn arc on vector sheets | Text layer, Spark strips cut on row lines, vector arcs | Flag. A wrong row with no detected arc is not seen | Fix for a tag or row with no partner. Check for type against arc |
| 8 | Exit capacity on life safety sheets. Sum shown only under the half-sheet tile rule with two equal reads, and compared only with a printed load | Spark tiles | Never reported as a pass | Cannot tell by default. Check when the sum is below a printed load |
| 9 | Stair continuity: gap between Found floors | Presence rule below | False absence raises a flag | Fix when the cell is Missing, Check when the cell is Check |
| 10 | Stair does not reach the level of exit discharge | Same, plus the discharge level | Flag | Fix when the cell is Missing, Check when the cell is Check |
| 11 | Stair count per floor against exits required for the declared load | Matrix, rule rows | Flag | Check |
| 12 | Every stair's top and bottom floor listed for the examiner, plus Check cells at the end of a run and every Not followed row | Matrix | Always shown | Open item |
| 13 | Travel distance on vector sheets, with the route drawn | Vector walls and arcs. Route ends: stairs from the matrix on upper floors, declared exit markers where a life safety sheet exists | Under-measures by 16 to 19% | Never Fix. Check when within 25% of the limit or over it. Shown as a minimum. Cannot tell where neither source of route ends exists |
| 14 | Corridor clear width on vector sheets with wall layers or a dominant wall pen | Vector | Measured against printed value | Check |
| 15 | Cycle 2: each earlier comment rechecked at its location, plus unmentioned changes | Registered pixel difference, Spark cloud list | See below | Looks resolved, Still open, Cannot tell, New change |
| 16 | Room names: finish or door schedule against plan labels. A room with no label read on a Spark-read plan is Cannot tell, because Spark misses overprinted labels | Text layer, Spark tiles of the plan area | Flag | Check |

Severities above come from frozen scores with at least 3 flags. Checks with fewer stay at Check until a new set confirms them. The declared-versus-drawn list rests on 13 real sets: 25 real and 19 likely discrepancies, but 35 of 137 held-out flags at frozen settings, and post hoc fixes that have not met a new set. Dropped because they raised false alarms: marker with no capacity, plan room not in schedule, door room not in finish, plan key not in legend, width against arc, closed swing vocabulary, swing with no arc, title compare by normaliser alone, occupancy word match, and any use of an IFC model in place of the drawing.

Any value a finding rests on comes from the text layer or from two agreeing Spark reads. Two reads that disagree give Cannot tell. Every finding carries a crop tight enough to read.

Checks are geometric or arithmetic. No check requires code text.

Limits stated on screen and in the README:

- A stair that ends on its top floor looks the same as one erased from that floor. Check 12 lists it as an open item. That is the accepted answer for now, and it is labelled as a limit, not a pass.
- A stair no run lists cannot be followed. On held-out buildings 3 of 79 main stair cells were not listed.
- Found means a stair is drawn at this place on this floor. It says nothing about whether the stair complies.
- The miss rate on real defects is not known. Every wrong-pass number comes from defects planted or labelled by one labeller.
- Door swing against occupant load is left out, because no code joins a door to a room's load.

One- and two-family dwellings use the International Residential Code, which this product does not check. When the code sheet or room labels show a dwelling, the rule-based checks (1 for the life safety sheet, 11, 13, 14) move to Not checked with that reason, and the coordination checks (2, 3, 6, 7, 9, 10) still run.

The discharge level comes from floor labels: the sheet labelled ground, first or street level. The title block has a selector so an examiner can change it. If no label settles it, check 10 is Cannot tell.

## How stair presence is decided

The rule is arm C of the `presence` experiment, frozen before held-out, in `lab/presence/rule.py`. On 15 planted erasures it gave 0 Found (12 Check, 3 Missing), against 10 Found for three-run voting alone and 13 for one run. On 38 tested natural absent cells it gave 0 Found. It costs about 0.32 USD per building and sends about 7 stair cells to Check. Reproduced by the audit, including live repeats of the crop. On harder plants that keep the walls, ink fired on only 5 of 22 and the crop caught them, so the crop is now the main guard and a single point of failure.

1. Spark reads each sheet, orders the floors in one joint call, and each floor is registered to the reference floor. The gate needs 25 inliers and a scale within 10% of the median scale of the floors.
2. Spark compares all floors in one call, three runs, medium effort. Every present point is mapped to the reference floor and clustered at 80 px. A cluster is a row. A row is a place, not a name.
3. A row is established when two runs agree on some floor.
4. For each cell of an established row: v is the number of runs claiming it, ink is the ink in a 60 px box at the registered place over the median of the other floors with v of at least 2, blank means under 0.002 ink density in a 100 px box, and crop is a blind yes, no or unclear on a 260 px crop at 3x, low effort. The crop question never mentions other floors.
5. Decision. Unaligned floor: Check. v of at least 2: Check if ink is under 0.40, the place is blank, or the crop says no or unclear; otherwise Found. v under 2: Missing if blank; Check if the crop says yes or unclear, or ink is at least 0.90; otherwise Missing.
6. A row that is not established stays on the matrix under the heading "Seen by one run". A cell with some support is Check. The rest are Not followed. Experiments hid these cells, which made an erased stair vanish without a trace in one audit case, so the product shows them.
7. The marker is the registered row place, not Spark's point, which drifts up to 112 px.
8. A Missing cell above or below every Found floor of its row reads Not on this floor. A Missing cell between Found floors, or on the discharge floor of a stair serving floors above, raises check 9 or 10.

Crops and ink never lift a cell to Found. SAM takes no part in the decision. Its masks are drawn only on Found cells, as a picture of what was found. Stairs only: elevator and shaft claims in the frozen prompt are ignored.

## Architecture

Python 3.12, FastAPI, vanilla JS, no database. Everything for a review sits in one folder. Models: `muse-spark-1.3` for reading and comparing, `sam-3.1` for plan boxes and evidence masks. `muse-wonder-1.0` is not in the runtime path: it gained no accuracy on any reading task at 3 to 5 times the cost, and on routes its verdicts flipped on 3 of 8 repeats. Every runtime model is Meta's.

```
upload
  ingest        PDF or images to sheet PNGs, text layer words, vector or scan   (built)
  read          Spark low effort per sheet: number, title, class, level, scale, plan box. SAM "floor plan" for viewport boxes, Spark's box when SAM returns none
  floors        Joint Spark order, alignment, inlier and scale gate, 960 px stack video
  verticals     Spark comparison three runs, crops, ink, matrix, SAM masks on Found cells
  declared      Text layer tables and index first. Spark strips and tiles where no text exists, two reads
  vector        Walls, arcs, rooms, routes, scale chain. Vector sheets only
  evaluate      Facts plus rule rows to findings. Deterministic and free to rerun
  audit         Optional. Spark low on stair, door and value overlays. Adds a disputed tag, removes nothing
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

`storey/rules/rows.json` holds only the rows the shipped checks use, each with section number, value, unit and jurisdiction overrides. The full research file stays in `data/rules/` and out of git. Titles are in our own words. Unknown sprinkler status uses the stricter value and raises an item asking the examiner. Jurisdictions shipped: Houston, Philadelphia and Seattle. Houston and Philadelphia matched the model code on all 144 checks, so a switch between them changes only the adopted code named on each item. Seattle differs where a shipped check depends on occupancy or load, most visibly in its exception that lets some R-2 buildings up to five stories use a single exit, so the demo set needs a code sheet and a declared occupancy for the switch to change an outcome. Occupant load calculations appear in the item so the examiner can see the rounding.

## Review data contract

The current `review.json` stays and gains these fields.

- `project`: `jurisdiction`, `adopted_code`, `discharge_level`, `facts` with sources.
- `sheets[]`: `number`, `class`, `kind`, `scale`, `reviewed` (bool), `reason`.
- `verticals[]`: `established` (bool). `verticals[].floors[sheet]`: `state` (found, missing, check, not_followed, end), `signals` (`spark` as n of 3, `crop`, `ink`, `aligned`, `reason`), `point`, `box`, `crop` (file name), `mask` on found cells only.
- `findings[]`: `key` (a stable hash of check, sheet number and object), `comment_id` (discipline letter and number, kept across cycles), `check`, `discipline`, `state` for cycle 2.
- `routes[]`: `sheet`, `path`, `length`, `label`.
- `open_items[]`, `not_checked[]`, `cost` (USD and seconds per stage).

## Screens

1. **Start.** Upload, sample reviews, past reviews. Handles `#key=` by posting to `/api/session` and clearing the hash.
2. **Progress.** Stages with counts, thumbnails as sheets are read, spend so far.
3. **Review.** The existing title block and three panes, changed as follows.
   - Matrix cells read the `state` field. Dashed means Check. The legend reads Found, Missing, Check, Not followed, Not on this floor. Rows that are not established sit under "Seen by one run". Clicking a cell shows its crop and the reason for its state.
   - Tabs in the right pane: Findings, Set, Open items, Not checked.
   - Jurisdiction, sprinkler and discharge level selectors in the title block, wired to the free re-evaluate call.
   - Routes drawn on the sheet with the label "Minimum. Examiner to confirm."
   - Keep and Drop decisions saved on the server.
   - A test tool, "Remove this stair from the sheet", which edits a copy of the sheet and reruns the stair stage. It is labelled as a test, and it is offered only on floors where a finding can appear: floors with the stair above and below, and the discharge level. It proves the check is live.
4. **Letter.** Draft in the real letter format, sheet first, imperative wording, section at the end, status per item with no approval box. Editable. Downloads as PDF.
5. **Second review.** Upload a resubmittal against a review. Each comment gets its state, with old and new crops side by side. Unmentioned changes are listed.
6. **How well it works.** The published evaluation, misses included, reading from the eval output.

Design language stays as built: a drawing-office title block, IBM Plex, red for the reviewer's pen.

## Second review

Sheets are matched by sheet number, then registered. Pixel difference on the registered pair finds every changed region: both real sheet pairs tested, in the best case of two renders of one PDF. For each earlier comment, a changed location reads Looks resolved, an unchanged one reads Still open, and a sheet that cannot be matched or registered reads Cannot tell. Spark on 12 overlapping tiles lists revision clouds and delta numbers as a draft for the examiner, 18 of 20 clouds on two runs at about 0.10 USD a sheet, from one PDF only. An empty list never means no change. A changed region outside every comment and every cloud reads New change. SAM does not take part: it scored 0 of 20 clouds and followed the place rather than the object. Comment IDs persist, and items carried forward keep their history, matching how Sonoma's letters mark them.

## Evaluation

`eval/` runs the real pipeline, not lab code.

- Scores in this order: wrong passes, findings per set judged real, false findings per set, cost and minutes per set, with a baseline beside each.
- Truth is written first by the assistant, one labeller, not a licensed examiner, and the report says so.
- Almost every set has been touched by an experiment, so those numbers are development numbers and the report names them as such. Reserved and unopened until Task 14: `queens_royce` Phase 2 and `army_bt_bcof`. Held-out numbers come only from these.
- Planted defects on real sheets with `lab/defects/plant.py`: stair gap, stair cut off above discharge, duplicate sheet, door tag changed, table value changed. Stair plants keep the walls, using the soft plant rule from `lab/audit2`.
- Richmond Appeal 26-02 is the only real examiner record in hand. The method sees packet page 107 only. Pages 84, 86, 87, 101 and 176 hold the answers and are never passed to it. Score whether each of the 10 sheet-only comments was raised for a human, never pass or fail, because the examiner was overturned on appeal. Baseline: a rule that recomputes each area over factor line. It is never shown publicly.
- One full manual audit of `mml_capitol`, sheet by sheet, gives the only miss rate available. It is one labeller and one set, and the report says so.
- A published number comes with its status: frozen or post hoc. The post hoc 93% declared precision is never quoted. The frozen figure is 35 of 137.

## Demo sets

Public on screen: `habs_stelizabeths_center` (public domain, 5 floors, stairs), `mml_capitol`, `mml_baxter` and `mml_abbot` (vector, schedules, real door tag and label findings). `nibs_clinic` supplies numbers only until WBDG gives permission. `bme_f` is not shown, because its date tag cannot be trusted. Replace the sample review, which still uses `bme_f`, with real pipeline output on `habs_stelizabeths_center`.

No demo set has both a code summary and a life safety plan, so no public demo shows checks 8, 11 or 13, or an outcome that changes with jurisdiction. The selector still changes rule values, adopted code names and Not checked items, and the demo says plainly that no sample exercises the R-2 single exit difference. A generated test set would close this gap. It is deferred until one building runs end to end.

`navy_b14` is excluded from the product, the evaluation and all model calls until the user rules on it, because a research agent fetched it by script against SAM.gov's terms. It is also a naval shipyard building, and 245 of its 257 sheets have not been checked for restriction markings. The Second review demo uses a planted revision on a `public_demo: yes` set, labelled as planted.

## Hackathon use of Meta tooling

- Spark multi-image comparison across a whole building, three runs, clustered by place, then checked by a blind crop question. Structured output, grounding on a 0 to 1000 grid, tiled reading of fine text and revision clouds.
- SAM 3.1 concept prompt "floor plan" for viewport boxes at a ninth of the cost of a Spark call, with Spark's box as a required fallback because SAM's silent miss on a life safety sheet repeats on every call.
- SAM 3.1 video tracker over a stack of floor plans, pinned on each confirmed floor, draws the stair on every Found cell. It is evidence for the reader, not a decision.
- A findings schema with no pass value, enforced by constrained decoding.
- The published experiment log: what was tried with each Meta model, what failed and by how much, including SAM as a presence signal, Spark video, and Wonder as a driver. This is the honest account of the tooling and the strongest part of it.
- Stretch: Glimmer running locally so a city's drawings stay on its own machine. Test on the 24 GB laptop before any claim.

## Constraints

- Budget: 148 USD total, 54 spent after Round 2. Reserve 25 USD for the final evaluation and demo rehearsals.
- Repo `tns15june/storey`, private until submission, branch `build`. Commit often, because the platform flags single-commit repos. Scan staged files for secrets.
- Never print the API key. Security rules from the first three reviews stay in force: the API always needs a key, uploads are bounded, PDFs are read in a child process.
- Video at most 3 minutes, in English, link opens without login. README has "Built during the hackathon" and "Pre-existing assets" sections and a licence.
- The public demo is a static site on GitHub Pages holding sample reviews for `public_demo: yes` sets, with one precomputed review per jurisdiction so the selector works with no server and no spend. Keep and Drop stay in the browser there. The full app runs from the repo with the user's own key, and that is what the video and the finals use. A hosted full app is a stretch that needs an account from the user. Sample reviews and their images are committed to `app/web/sample/`, because `data/` is ignored.

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
