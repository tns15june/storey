# Build log

Measured numbers and scope decisions, newest last. All times IST.

## 31 Oct

- 13:00. Picked YC "AI for City Hall", plan review. Probe on BME Building F (4 floors, old scan, truth by eye): SAM word and click prompts failed on line drawings. SAM video tracker with floors stacked as frames and box prompts matched 15 of 15 floor and staircase pairs and reported no mask on a fully erased staircase. Spark alone found the same staircases per sheet.
- 13:20. User approved the build and asked for deep research and many experiments. Round 1 spend cap: 60 USD of 148.
- 13:45. Three security reviews of the first pushes flagged 10 upload and access issues. All fixed with tests (24 passing): body limit enforced while receiving, page, pixel and disk limits, two workers and eight waiting at most, the API always needs a key (a random one per run if none is set), cross-site posts refused, and PDFs read in a child process that is killed after 15 minutes and reports through a file.
- 15:23. Round 1 finished: 8 experiments, 5 research agents, one audit. 29.05 USD spent of 148. Agent outputs are in the workflow result; lab code in `lab/`, outputs in `data/lab/` (gitignored).

### Round 1 numbers that decide scope

Audit status in brackets. Reproduced means an independent agent reran it. Weakened means the number holds but the claim built on it does not.

| Question | Result | Status |
|---|---|---|
| One Spark call comparing all floors, stair presence | Dev 94/96 pairs. Held-out 42/45 stack runs, present 156/162, absent 27/27, 0.02 to 0.05 USD per call. Errors are false absences only | Reproduced on held-out |
| SAM tracker seeded from Spark boxes, no human | Dev 18/21 by best-cluster credit, 15/21 strict. Held-out 8/15, below the always-present guess at 10/15 | Weakened |
| SAM tracker seeded from tight hand boxes | bme_k 32/33. Ghosts on 4 of 5 absent floors with Spark boxes, conf 0.65 to 0.77 | Audit run |
| OR fusion of SAM and Spark | Refuted. Every SAM ghost becomes a stair reported where none exists | Refuted |
| SAM confidence as a presence signal | Fails across drawing styles. Correct masks as low as 0.008, wrong masks up to 0.77 | Refuted |
| Planted stair erasure, no-AI ink in a fixed box on aligned stacks | 47/49 | Reproduced |
| Planted stair erasure, SAM offsets plus ink, 60 px jitter | Fresh 14/16, ink alone 3/16 | Reproduced |
| Stair erased on its first or last floor | Hybrid 3/42, SAM rule 10/42, Spark crops 21/34 | Open safety gap |
| Revision between cycles, erased, moved, exit door changed | SAM tracks 36/36 with 0/72 false flags, pixel diff 33/36 | Cut-offs set after seeing data |
| Revision, added partition in a room | SAM carries the old mask, 0/35. Do not use the tracker for this | Failed |
| Vector door arcs, clinic | F1 89.4/91.1, F2 94.9/96.9 recall/precision, swing 99 to 100%, width error 13 mm mean | Reproduced |
| Spark doors on tiles at 7 to 9 mm per px, low effort | 92.5 to 100% recall, 85 to 98% precision. Whole sheet 8.7% | Reproduced on F2 |
| Exit doors from plans alone on the clinic | All methods fail. Reading declared markers on life safety sheets works, 7/7 | Open |
| Walkable mask and travel distance, vector clinic | Reach 99 to 100%, within 15% for 90 to 97% against IFC truth. Routes 16 to 19% under the architect's printed figures | Weakened: one building, stairs and exits from IFC |
| Walkable mask, raster | Best within-15% 58.9%, shortcuts 13.8%. No method passes | Failed. No distance claim on scans |
| Vector scale chain | 203 dimension strings within 0.26 in after one fitted overshoot | Reproduced |
| Title, level and scale read | 42/45 at 0.0074 USD per sheet, fresh sheets 30/33 and 31/33 on two runs | Reproduced |
| Joint floor order | 10/10 buildings, 39 sheets | Not audited |
| Alignment, Spark plan box, coarse search, block matching | 8/8 buildings under 1% of sheet width, worst floor 0.36% | Not audited. reg.py not tried on held-out sheets |
| Code facts from a code summary sheet | 12/12 fields on 3 sheets | Not audited |
| Door schedule strips with repeated header | 91/91 rows, none invented | Not audited |

### Real defects found so far, none planted

- mml_baxter A2.1: door tag 13 sits on a 36 in swing door, schedule row 13 says 60 in bifold. Rows 12 and 13 look swapped against the mirror unit.
- mml_capitol A2.1: title block says MAIN FLOOR PLAN, the drawing says Ground Floor Plan.
- BCOE sheet A012: exit table says 90 occupants, 18 in, clouded in Plan Check 2. The worked example below still says 100 occupants, 20.0 in.

### Corrections to earlier claims

- "No competing entry" is wrong. Moke covers whole-packet review for design firms, ConformScope covers wheelchair access with the same architecture, Helonic advertises stacking and egress checks commercially.
- bme_f is not safe for public demo: the sheet legend post-dates the 1905 Commons date. Use habs_stelizabeths_center.
- Repeat 4 was the wrong choice for stacks. Repeat 1 at 960 px is as good and cheaper.

- 19:00. Round 2 finished: 6 experiments and one audit, 24.95 USD, ledger 54.01 USD of 148. Output is in the workflow result; lab code in `lab/`, outputs in `data/lab/`. The design and plan now carry these decisions.

### Round 2 numbers that decide scope

Audit status in brackets. Frozen means the rule was fixed before held-out. Post hoc means it was changed after seeing results.

| Question | Result | Status |
|---|---|---|
| Stair erased mid-run, three-run Spark voting as designed | 10 of 15 planted erasures still Found. Run 0 alone 13 of 15. Round 1's "false absences only" does not hold | Reproduced, frozen |
| Arm C: ink and a blind crop on every Found | 0 of 15 Found (12 Check, 3 Missing), 0 of 38 tested natural absent cells, upper bounds 18% and 7.6% | Reproduced. Weakened: on plants that keep walls ink fired on 5 of 22, the crop is a single guard |
| Check load and cost, arm C | 7.2 stair cells per building, 0.32 USD, about 68 s. The crop says no on the same 5 clean stairs every time, so part of the load is permanent | Reproduced |
| SAM tracker from hand boxes, 960 px on aligned stacks | 107 of 110, 14 of 14 absent floors | Reproduced, but only for top-of-run endings |
| SAM with treads erased, walls kept, accurate pinned seeds | 14 of 20 masks stay on the empty stair room at clean confidence | Audit, live. Refutes "the tracker stops where the stair stops" |
| Best SAM seed with no human | Frozen 82 of 110, post hoc 92 of 110, always-present 96 of 110 | Reproduced. Still under baseline |
| SAM mask as a veto on ending flags | 11 false cancels on pinned tracks, 1 true ending cancelled | Refuted. Dropped |
| Alignment gate | 26 of 27 floors, floor order 9 of 9 buildings. A wrong fit passed at 57 inliers, scale 0.92 against 1.13 | Weakened. Scale check added |
| Declared against drawn, 13 real sets | 25 real and 19 likely distinct discrepancies. Held-out flags at frozen settings 35 of 137 real or likely. Text layer 9 of 26, Spark 26 of 111 | Reproduced. One unblinded reader. 93% after fixes is fitted to the same flags and is not quoted |
| Tag against schedule, both ways, text layer | 6 flags, 5 real, 1 likely, 0 false | Frozen, pooled |
| Exit capacity sum | Frozen 1620 against 1440 and 441 against 294, the unsafe direction. Half-sheet tile rule 1440 and 294 on 4 reads plus 4 live | Post hoc, one building. Empty tile replies occurred. Never reported as a pass |
| Spark marker reading on life safety sheets | 32 of 33 no-capacity flags false: pull stations and detectors read as exits | Dropped |
| Clean table reads | USC 105 of 105. Clinic door rows 254 of 255 with row-cut strips, 224 of 255 with frozen strips | Row cut is post hoc |
| Code sheet numbers | 44 of 46 field reads. Two reads required | Frozen |
| Model audits an overlay | Spark low 16 of 16 bad, 6 of 16 good rejected. Live repeat 14 of 16 with two new wrong passes. Routes: Spark rejects all, Wonder flips 3 of 8 | Weakened. Optional tag only |
| Agent driving | Frozen answer no. Spark dropped real row D04 twice, and frozen truth had called those runs clean | Refuted |
| Letter | 0 added facts in 4 letters. A template did the same for free | Not independently rescored |
| Spark on 12 tiles, revision clouds | 18 of 20 on two runs, 3 and 5 false, 0.10 USD a sheet. SAM 0 of 20. Whole sheet 4 of 20 | Reproduced. One PDF only |
| SAM concept "floor plan" | 23 of 48 against Spark 14 of 48, 0.0025 USD a sheet. Silent miss on sheet n161 on 3 of 3 calls | Reproduced. Spark fallback required |
| Spark video on floors | 12 of 59 against always-present 50 of 59. Flat 124 tokens a second | Not rescored. Dropped |
| SAM across versions | Replaced arrow stays tracked 3 of 3 at 0.45 to 0.49 | Not rescored. Dropped. Pixel difference finds the regions |
| response_format and the prefix cache | Image-first caches 97% in plain text, 0% with response_format | Not rescored |

### Real records found

- Richmond VA Appeal 26-02: one submitted sheet with 14 real examiner comments, 10 answerable from the sheet. Virginia packets are public records. No full set paired with a full city letter was found.
- Navy B14: 257 sheets, code and life safety sheets, 25 reissued sheets with a written change list, 211 bidder questions. Deleted, see below.

### Real defects added, none planted

Baxter upper tags run 17 to 34 against a schedule of 1 to 18, and mark 9 has no plan tag. Abbot tag 20 has no schedule row. ADU2 lists sheet AD-901 on its index and the PDF does not hold it. Clinic finish rows 1E25 to 1E28 hold the wrong rooms. Clinic room 2B19 is OFFICE on the plan and WATER LAB in both schedules. Clinic A1-4 prints SHEET __ OF 316.

### Conduct

The `realsets` agent scripted SAM.gov's internal endpoints and fetched 14 files before reading its terms, which restrict automated collection. It also made 5 requests to Portland OR efiles before reading a robots file that bars Claude agents, sending a generic browser user-agent on some, and pulled 13.3 MB of a Sonoma plan file that the county says is protected. The Portland and Sonoma files were deleted. Both SAM scripts exit on start. The user had the Navy set and everything made from it deleted on 31 Oct: 15 source files, plus 33 MB of renders, overlays and truth files. No Meta model saw it. The research agent viewed about 12 of its sheets. The SAM.gov search log stays in `data/lab/realsets/sam`, and it holds no drawings.

### Corrections to earlier claims

- "SAM follows a stair and stops where it stops" is wrong as a general claim. It follows the room.
- "Spark errors are false absences only" holds on clean sheets, not on a stair erased mid-run.
- "SAM tracks reveal changes between cycles" is withdrawn. Round 1's 36 of 36 had cut-offs set after seeing data, and a replaced symbol stays tracked.
- The pitch says stairs, not stairs and shafts.

- 23:20. Task 10, the comment letter. Five live rewrite calls on 11 hand-built comments cost 0.1308 USD. One call at low effort took 2464 output tokens and one at medium took 8127, and each kept 10 of 11 rewrites, so the letter runs at low. The eleventh comment was a planted room label telling the model to write that the set is approved. In all five runs the model quoted the label and did not obey it, and the status word check rejected the quote, so that comment kept its template text. The template prints the label as given, so Task 9's no-pass-value test is the only guard on that text.
- 23:20. Task 10 review. Stubbed replies showed the rewrite gate kept reversed, dropped and swapped word-only values and 11 pass phrases such as "meets the code". None of the real rewrites had these faults. The gate now needs every content word kept, no added negation, names in the same order and no new stair letter, and the status word list is longer. Replayed on the cached replies, runs 3, 4 and 5 stay at 10 of 11 kept. Run 2, from an earlier prompt that left stray capitals such as "with A sheet", falls from 10 to 6. These replies predate the check 13 wording "shown here as a minimum", which has had no live run.
- 23:55. Task 5 review. The lab's 12 Check, 3 Missing on the 15 held-out plants comes from `decide` alone. Through the span step that ships, a replay with no model call gives 12 Check, 1 Missing, 2 Not on this floor, 0 Found: `bme_k__top` and `schenley__top` fall outside the span, which is the stated top-of-stair limit. The shipped states are the figure to publish. A failed crop on a cell under two runs now reads Check, because one failed crop on `bme_k` moved a real main stair from Check to Not on this floor. The replay of 979 cells and 230 rows does not change, because all 686 established, aligned, non-blank lab stair cells have a crop answer.
- 23:55. Crop repeat, clean stairs only. On 6 clean `bme_k` cells, chosen because the lab's and the port's first answers differed, both crops were asked 3 more times, 12 crops in all. 6 of the 12 changed answer, so "the same 5 clean stairs every time" above does not hold for these crops. All 12 are clean sheets with the stair off the crop's centre, so this shows how far the crop's answer varies on real stairs and says nothing about erased ones. The evidence for erased stairs is the audit's repeat on planted cells, `L3_crop_repeat_hard`: all 12 non-Huffman cells said no on 36 of 36, and the 3 plants that only the crop catches said no on 9 of 9. Huffman said yes on stipple on 9 of 9, so on that sheet only ink catches an erasure.
- 23:55. Row drift. Between two fresh `bme_k` runs, row `pos` moved by up to 146 sheet pixels on a 6000 px sheet, which is 49 px on the 2000 px copy, and row ids changed. The second run had a fourth floor, because the first misread one floor name, so this pair may overstate the drift between runs of one set of floors.
