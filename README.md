# Storey

Storey does the first pass of plan review for a city building department. An examiner uploads a permit drawing set. A few minutes later Storey shows what is in the set, follows every stair from floor to floor, checks what the set declares against what it draws, lists what it could not check, and drafts the correction letter in the format cities already send.

It never approves a plan. Every screen reads "Draft for examiner. Not an approval." There is no pass state anywhere in the product, so no model can return one.

Built solo for the Meta Global AI Developer Hackathon (Indie track), from the "AI for City Hall" item on Y Combinator's [Requests for Startups](https://www.ycombinator.com/rfs).

## Why

- Portland's city auditor found in 2021 that more than 80% of standard building permit plans needed corrections before they could be approved. Each correction cycle is weeks of waiting for the applicant and another full read for the examiner.
- Seattle evaluated its own AI plan-check pilot in 2026. On 1,262 checks it passed 6.4% that a human reviewer had sent back for correction ([report](https://seattle.gov/documents/Departments/Performance/Publications/2026CivCheckEvaluationReport.pdf)). A wrong pass is the error that matters in plan review, because nobody looks again at what was passed.

Storey is built around that error. Models read and locate. Code measures, applies the rule rows and sets the severity. When Storey cannot tell, it says Cannot tell on screen, and an item it could not read never turns into an absent fact.

## What it does

| Screen | What the examiner sees |
|---|---|
| Set | Every sheet with its number, title, class and scale, and which sheets were reviewed |
| Stairs | A grid of each stair against each floor: Found, Missing, Check, Not followed, Not on this floor. Each cell opens its crop and the reason for its state |
| Findings | Fix, Check and Cannot tell items. Each one carries the sheet, the place on the drawing, the measured or read value, the rule row and a crop |
| Not checked | What Storey did not review and why: a stage that failed, a scan where nothing can be measured, a discipline it does not cover |
| Letter | The draft comment letter, sheet first, imperative wording, with the section at the end. Editable, prints to PDF |
| Second review | Upload the resubmittal. Each earlier comment reads Looks resolved, Still open or Cannot tell, with old and new crops side by side. A change nobody flagged reads New change |
| Jurisdiction | Houston, Philadelphia or Seattle. The rules run again with no model call, so the switch is instant and free |
| Test tool | "Remove this stair from the sheet" erases a stair on a copy and runs the stair stage again, to show the check is live |

### Checks

1. Required sheets present (code summary, life safety plan, door schedule, a plan per storey, site plan)
2. Sheet index against the sheets in the set
3. Floor named in the title block against the label under the drawing
4. Code summary fields present and consistent
5. Stories declared against floor plans drawn; code edition against the jurisdiction's adopted edition
6. Table arithmetic: room loads, totals, exit width against occupants
7. Door and window tags on plans against schedule rows, both ways; schedule type against the drawn door swing
8. Exit capacity on life safety sheets
9. Stair continuity: a gap between floors where the stair is found
10. A stair that stops above the level of exit discharge
11. Stairs per floor against the exits the declared load needs
12. Every stair's top and bottom floor, listed for the examiner
13. Travel distance on vector sheets, with the route drawn, shown as a minimum
14. Corridor clear width on vector sheets, including the width left beside a stair standing in a hall
15. Second review: each earlier comment at its place, plus changes nobody mentioned
16. Room names in schedules against plan labels
17. Dimension strings that do not add up to their overall

Electrical, plumbing, mechanical, structural, zoning and energy sheets are listed in the inventory and marked as not reviewed.

## How it uses Meta's models

Every model at runtime is Meta's, through the Meta Model API.

- **muse-spark-1.3** reads each sheet's number, title, class, level, scale and plan box at low effort, and orders the floors of the building in one joint call. It compares all floors of a building in one multi-image call, three times, to list where each stair is drawn. It reads schedules and fine text in tiles where a sheet has no text layer, with two reads that must agree. It lists revision clouds on a resubmittal, and rewords the letter under a gate that allows no new word beyond a short list of connecting words, keeps every value, and rejects any status word such as "approved".
- **sam-3.1** finds "floor plan" viewports on a sheet at a ninth of the cost of a Spark call, with Spark's box as the fallback when it returns none. Its video tracker runs over the stack of floor plans and draws each stair on the floors where it was found, as evidence for the examiner.

What decides stair presence was settled by experiment, not by the model's word. Spark alone still claimed 10 of 15 planted stair erasures across three runs. The shipped rule keeps Spark's three runs but also needs ink at the registered place and a yes from a blind crop question that never mentions the other floors. On the same 15 planted erasures it gave no Found. SAM takes no part in that decision: with the treads erased and the walls kept, 14 of 20 SAM masks stayed on the empty stair room at the same confidence. The full experiment record, including what failed, is in [docs/research/build-log.md](docs/research/build-log.md).

On the sample sets a review cost 0.42 to 0.53 USD of model calls and took 2.3 to 3 minutes. Each model call is cached, every call sets a token cap, and each review has a spend cap.

## How well it works

The evaluation runs the same pipeline the server runs, from `eval/`. Truth was written by labellers who saw no Storey output: two independent passes and an adjudicator for every disagreement. Two real sets were held out from all development and run once, on code frozen by hash in `eval/frozen.json`. The baseline is Spark asked directly, with every sheet in one call and a plain prompt.

33 sets ran once each on the frozen code (1 November 2026), and every claim that needed judgement was mapped to truth by two judges blind to which method made it, with an adjudicator on the 11 of 382 claims they split. The full report, misses included, is on the How well it works page (`app/web/eval.html`).

| | Storey | Spark asked directly |
|---|---|---|
| Wrong passes on held-out sets: stairs shown Found on cells truth says are absent or erased | 1 of 16 cells | shows no stair matrix |
| Wrong passes on development sets: 18 stair erasures in 6 real buildings, and the stair defects planted in 22 generated sets | 1 | shows no stair matrix |
| Generated sets: planted defects and rule items raised | 184 of 184 | 81 of 177 |
| Generated sets: findings with no planted cause | 15 | 161 |
| Real stair erasures raised (6 buildings) | 12 of 18 | 0 of 18 |
| Real sets: items of two hand audits and one examiner's letter raised | 6 of 121 | 15 of 121 |
| Median cost and time of one review | 0.51 USD, 3.3 minutes | 0.08 USD, one call |

What the numbers say:

- On the checks Storey makes, it is nearly complete and rarely wrong: every planted defect on the generated sets was raised, with a tenth of the baseline's false findings.
- The two wrong passes are both stairs. On the held-out Queen's set, a striped strip on a roof band that the truth labellers marked absent (likely) was shown Found: Spark's three runs, the ink test and the blind crop all read it as a stair. On the development set Huffman, an erased top-floor stair kept faint ghost lines, and the crop said yes to them.
- The six real erasures not raised are all on a stair's top floor, where an erased stair looks like a stair that ends. That is the first limit below.
- On real sets audited by hand, Storey raised few items. Most audit items are kinds no Storey check covers, such as section callouts that point at sheets missing from the set. Spark asked directly finds more of those, with no evidence attached and far more false findings on the generated sets.

## Run it

Python 3.12.

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
echo "MODEL_API_KEY=<your Meta Model API key>" > .env
.venv/bin/python -m storey
```

The server prints a link that carries this run's access key. Open it, upload a PDF or page images, and the review appears as each stage finishes. Without a server, open `app/web/start.html` from any static file server to browse the sample reviews.

Tests make no network call: `.venv/bin/pytest`.

## Limits

- A stair that ends on its top floor looks the same as one erased from that floor. It is listed for the examiner, not passed.
- Found means a stair is drawn at this place on this floor. It says nothing about whether the stair complies.
- Travel distance and corridor width are measured on vector drawings only. A scan reads Cannot tell.
- The miss rate on real defects is known from one full audit of one set by one labeller.
- One- and two-family dwellings use the International Residential Code, which Storey does not check; those rule checks move to Not checked.
- One upload is one building.

## Built during the hackathon

All code, prompts, tests, rule rows, truth files and the generated test sets in this repository were written between 31 October and 4 November 2026, during the hackathon. The commit history shows the work as it happened. `lab/` holds the experiments that set the design, `storey/` the product, `eval/` the evaluation and `app/web/` the screens.

## Pre-existing assets

- Python libraries listed in `requirements.txt`, and the IBM Plex fonts from Google Fonts.
- Public drawing sets used for development and evaluation. They are not in this repository:
  - Historic American Buildings Survey drawings of St. Elizabeths Hospital, Washington DC (public domain, US federal record). Sample reviews of this set are shown on the demo site.
  - Michigan pattern book duplex and ADU drawings, published as copyright-free construction documents. Sample reviews of The Capitol are shown on the demo site.
  - A US Army Corps of Engineers Fort Worth District sample set and Queen's University's Jean Royce Hall plans, used as held-out sets. Only counts are published for drawings whose terms do not allow showing them.
- Generated test sets are drawn from [Swiss Dwellings v3.0.0](https://doi.org/10.5281/zenodo.7788422) apartment geometry (CC BY 4.0).
- Building code values in `storey/rules/rows.json` come from the model codes the three cities adopt, with titles in our own words.

## Licence

The code is under the MIT licence in [LICENSE](LICENSE). Drawings and datasets keep their own terms.
