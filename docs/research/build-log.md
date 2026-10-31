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
