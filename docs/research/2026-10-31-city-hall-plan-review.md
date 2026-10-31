# YC list pick: AI for City Hall, 31 Oct 2026

Status: my pick, waiting for the user's approval. No product code exists. The probe code in `spike/plan_probe/` is throwaway.

## The pick

From YC's Winter 2027 Requests for Startups (https://www.ycombinator.com/rfs), I picked "AI for City Hall", starting with plan review. The first version checks one thing a reviewer must check on every building: whether people on each floor can get out.

Reasons:

- Everything is software. It can be built and tested on public drawings, with no filming, pouring or shopping.
- YC names the job directly: "checking drawings against the code, matching applications to zoning rules, sending corrections back and forth."
- YC estimates that approving US permits in 10% of the time would add $16 billion a year in city tax revenue, across 19,000 cities.
- Buyers exist. LA County runs an AI plan-check pilot with Archistar for wildfire rebuilds (https://www.news.caloes.ca.gov/governor-newsom-announces-launch-of-new-ai-tool-to-supercharge-the-approval-of-building-permits-and-speed-recovery-from-los-angeles-fires/).
- A keyword search of 3,454 public hackathon READMEs found no entry that checks drawings against a building code. Private entries are invisible to this search.

Other YC items I ruled out:

- Space, robots, data center supply and physical finance cannot be built here in four days.
- The $15-an-Hour Expert overlaps a crowded lane of health paperwork, claims and legal helpers.
- Plaid for Government Benefits needs government data access that a hackathon entry cannot get.
- Own Your Intelligence and LLMs That Write Well depend on GPU training, and the team would struggle to see the impact.

## Test results

Test sheets: four public-domain floor plans of BME Building F, Budapest, scanned from 1960s-style drawings with Hungarian labels (https://commons.wikimedia.org/wiki/File:BME_Building_F,_level_0_floor_plan.jpg). Copies are in `data/plans/`, which is gitignored. Truth came from my reading of the sheets.

| Test | Result |
|---|---|
| Spark reads one sheet | Found all 5 staircases on levels 0 and 1, within about 2% of the sheet on most, read the title, floor and seat counts, and misread one room area (286 as 226) |
| Spark on all 4 sheets | Found 3 staircases on level 2, which matches the sheet, and 7 on the basement sheet, which also shows a separate block. The 7 were not all checked |
| SAM word prompts on a sheet | "staircase", "stairs", "door", "room", "hallway" and "wall" all returned nothing |
| SAM click prompts on a sheet | Failed: 3 of 5 returned nothing, and the other 2 were a tiny mask and a whole-page mask at 8% confidence |
| SAM video tracker, floors stacked as frames, boxes on the ground floor | 15 of 15 unprompted floor and staircase pairs matched the sheets, including 3 places where a staircase really ends |
| One staircase erased from level 1 | SAM reported no staircase on that floor and resumed on the floor above |
| Spark judging small crops at SAM's boxes | 3 of 4 valid crops right. It missed a real staircase on a low-resolution crop |
| Time and cost | The first test's alignment and two video runs took about 26 seconds in total. Total spend was about US$0.20, by my estimate from token counts and list prices |

The first erase test was invalid. Level 1 sits about 30 px lower on its sheet, so the white-out missed the bottom of the staircase, and both models correctly saw what remained. The figures above use the second run.

The sheet for level 2 did not align well. SAM followed the staircases across an offset of about 90 px anyway.

## What this does and does not show

It shows that Spark can read an old scanned plan and that SAM's video tracker can follow a staircase through a stack of floors. A floor plan set read as a video is the surprising use of SAM.

It does not show:

- That SAM is needed. Spark alone found the same staircases per sheet, more slowly. The honest claim today is two independent methods that can check each other.
- That the method generalises. This is one building and one erased staircase, and I set the truth by eye.
- That Spark's coordinates can seed SAM. I drew the boxes by hand.
- That walls, doors and distances can be read. None of that was tested. Dashed walls and door gaps on these sheets will be hard, and modern CAD sets are untested.
- That a building code check works. No code has been chosen, and the US model code text is copyrighted.

## Risks to answer early

- Archistar and others already sell this, so the entry has to show something they do not advertise.
- Many deadly fires were in buildings that never went through plan review. The pitch should rest on permit delay and on what a reviewer checks. It should not claim to have prevented those fires.
- A wrong pass is the dangerous error. The tool should only raise items for a human reviewer and should never approve a plan.
