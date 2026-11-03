# Response to the external code review, 3 November 2026

An outside review of this repository found five correctness defects, four evaluation defects and five operational gaps. This file lists each one, whether it reproduced, what changed and what is still open.

**The published numbers describe tag `eval-frozen` (commit a45163c), not this code.** `eval/frozen.json` froze `storey/` by hash at 23:06 IST on 1 November, and the 33 sets ran once on that code. Every fix below changes `storey/` after the freeze. The three held-out sets cannot run again as held out ([eval/REPRODUCE.md](../eval/REPRODUCE.md)), so no number on the How well it works page was rerun on the fixed code. The scoring files and truth files that `eval/frozen.json` pins were not edited.

Each code defect was first written as a test that failed on the unchanged code. Then it was fixed, and a second agent that had not written the fix tried to break it. That reviewer's must-fix issues were fixed in a further round, again test first.

## Correctness

| # | Finding | Reproduced | What changed | Commit |
|---|---|---|---|---|
| 1 | An unread value can resolve an earlier comment. `_checked` looked for doubt only under the comment's own check, on its own sheet. | Yes. The 2015 IBC edition comment read Looks resolved after the edition went unread, with the doubt raised under check 4. | Evaluate now lists `compared`: the key of each item whose facts were all read on the new set, whether or not it raised a finding. A comment reads Looks resolved only when its own key is in that list. Otherwise it reads Cannot tell where its place changed, and Still open where it did not. The second review round found the same gap in checks 3, 4 and 16, and those were closed too. | 1ca5b02 |
| 2 | Comments disappear in the third cycle. | Yes. A-01, Still open in cycle 2 and not raised again by cycle 2's checks, was missing from cycle 3. | Each recheck row keeps the comment as it stands on the new set. The next cycle compares the parent's Still open and Cannot tell rows together with its findings. Comment ids stay the same. | 1ca5b02 |
| 3 | The letter's rewrite gates accept reversed measured and required values. | Yes. "Corridor width needs 41 in clear and measures 44 in" passed all five gates. | The model now only joins the instruction to the item, with a colon or one of seven fixed joins, and both must stand word for word. Code writes the measured value, the required value and the section after that sentence. After a first fix, the reviewer's 32 adversarial replies still got 22 through. Now 0 get through. | e0225ac |
| 4 | Changing the discharge floor leaves stale travel routes. | Yes. Under the new level, the old routes were still drawn and a travel Check still ended at a stair on the new discharge floor. | Under a level other than the one routes were traced for, routes whose ends depend on the level are dropped. Travel on those floors reads Cannot tell and names the change. Routes whose ends do not move are kept. | 4a16fae |
| 5 | Rule prerequisites are loose. Sprinklers "yes" was read as NFPA 13, and corridor use was not read. | Yes. A B occupancy building read as sprinklered got Table 1017.2's 300 ft travel limit, and five cases raised no item at all. | Each sprinklered rule case lists the standards it accepts. If the sprinkler standard was not read, the stricter value applies and an item asks the examiner to confirm the standard. A row that needs NFPA 13 is not unlocked by 13R or 13D. A rule case can need a fact no stage reads, such as stretcher traffic in a B corridor. The item then names that case, and a width between the two values reads Cannot tell. | 1ca5b02 |

## Evaluation

| # | Finding | Reproduced | What changed | Commit |
|---|---|---|---|---|
| 6 | The judging was not blind. Copies kept each method's claim ids (A-01, B01) and Storey's severities. | Yes. Storey's sheet was also always numbered first, and the key sat in the judges' folder. | `eval/judge/blind.py` gives neutral ids (C01, ...) in an order shuffled by the sheet's sha256. It keeps only sheet and statement, numbers the copies by sha256 and writes the key beside the folder. `--unblind` maps rulings back, and refuses if a ruling is incomplete or duplicated. The 1 November mappings were made from the old copies, and the README and the submission now say so. | ad0547b |
| 7 | The report can publish with a baseline missing. | Yes | The public file is written only when every set has a scored baseline, and the report prints the reason when it is not written. | ad0547b |
| 8 | Parallel runs can lose `data/eval/runs.json` updates. | Yes | `eval/run.py` is frozen, so `eval/run_all.py` rebuilds `runs.json` from the `eval_run.json` files on disk once every run ends, and exits non-zero if a held-out run is missing from `eval/frozen.json`. | ad0547b |
| 9 | Reproduction depends on the author's machine. | n/a | [eval/REPRODUCE.md](../eval/REPRODUCE.md) lists every input set, its source and sha256, what is not distributed and why, and the commands to remake, rerun, judge and report. | ad0547b |

## Operations

| # | Finding | What changed | Commit |
|---|---|---|---|
| 10 | Image uploads skip the rendered-output disk cap. | Images count toward the same cap as PDF pages, and a failed ingest removes what it wrote. The server also removes rendered sheets when the reader is killed at its time or CPU limit, or crashes. | 86a238f |
| 11 | A failed ffmpeg stack fails the whole floors stage. | The floor order and registration are kept, Not checked says the outlines were not drawn, and the outline step sends no model call without a stack. The README lists ffmpeg. | 86a238f |
| 12 | Parallel calls can overshoot a spend cap. | Each call holds its worst-case cost before it is sent and settles after. A cap can be passed by at most one call's hold, 0.36 USD for the largest call. README Limits states this, and notes the daily cap is per server process. | 86a238f |
| 13 | No CI. | `.github/workflows/test.yml` runs the suite on Ubuntu and macOS on every push and pull request, with no secret. Its first run failed four floor registration tests on Linux, where the same pinned OpenCV gives different match counts at the 25-match gate. Those four now run only on macOS arm64, and README Limits says what that means on Linux. | 86a238f |
| 14 | Letter edits and decisions on the static demo stay in the browser. | README Limits says so. | 86a238f |

## Still open

- No model ran on the fixed code. A live run costs money and its numbers could not be compared with the frozen ones. The sample reviews on the site were made by the frozen code.
- An examiner cannot drop a comment carried into a later cycle until its check raises it again. The decisions endpoint accepts only the current cycle's findings.
- The Sprinklers selector offers Sprinklered and Not sprinklered, but cannot name the standard. Only the code sheet's statement decides it, so when the sheet names no standard the examiner cannot clear the item 'Sprinkler standard is not known' from the screen.
- Some comments that could read Looks resolved before now read Cannot tell. These are the ones whose facts the new set cannot show were read: parts of check 7, and checks 8, 11, 13, 14 and 17. That is the safe direction. On the 8 stored second reviews, all 40 comments that read Looks resolved still do.
- Two errors found during the work predate it and are not wrong passes, so they were left alone:
  - Check 2 can read Fix instead of Cannot tell when an index sheet gave no rows.
  - On an R-2 floor that needs three exits and has two stairs, check 11 reads Cannot tell instead of Check.
