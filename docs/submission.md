# Submission text

Paste-ready answers for the hackathon form. Numbers come from `app/web/eval.json` (frozen run, 1 November 2026).

**Project name:** Storey

**Track:** Indie

**One line:** AI plan review for city building departments. Storey reads a permit drawing set, follows every stair floor to floor, and drafts the correction letter. It never approves a plan.

**Links**
- Code: https://github.com/tns15june/storey
- Live demo (no key needed): https://tns15june.github.io/storey/
- Video: https://tns15june.github.io/storey/demo.html (2:33, plays in the browser with no login; the file itself is https://tns15june.github.io/storey/video/storey.mp4)

**What it does**

Most building permits wait weeks for a plan examiner to read hundreds of drawing sheets by hand, and most plans come back for corrections. Storey does the first pass. An examiner uploads a set; about three minutes and half a dollar later Storey shows every sheet, a grid of each stair against each floor, a correction list where every item carries its sheet, its place on the drawing, the value read, the rule and a crop, a list of what it could not check, and a draft comment letter. On a resubmittal it checks each earlier comment at its place and lists changes nobody flagged. Switching between Houston, Philadelphia and Seattle reruns the rules instantly with no model call.

It is built around the error that matters in plan review, the wrong pass: there is no pass state anywhere in the product, and anything Storey cannot read shows as Cannot tell, never as absent.

**How it uses Meta's models**

- Muse Spark 1.3 reads each sheet's number, title, level, scale and plan box, orders the floors in one joint call, compares all floors of a building in one multi-image call three times to find every stair, reads schedules and tables in tiles, lists revision clouds on a resubmittal, and rewords the letter under a gate that allows no new facts and no approval words.
- SAM 3.1 finds floor-plan viewports at a ninth of the cost of a Spark call, and its video tracker runs through the stack of floor plans to outline each stair on every floor where it was found.
- No model decides alone. Experiments showed Spark still claimed 10 of 15 erased stairs, and SAM's masks stayed on 14 of 20 emptied stair rooms. So a stair counts as Found only when Spark's three runs, ink at the registered place and a blind crop question agree. Models read and locate; code measures, applies rule rows and sets severity. The full experiment log, failures included, is in `docs/research/build-log.md`.

**How well it works**

33 sets run once each on code frozen by hash, two of them held out until after the freeze, judged by blind judges. Baseline: Spark asked directly with every sheet in one call.
- Generated sets with planted defects: 184 of 184 items raised, against 81 of 177 for the baseline, with 15 findings that had no cause against the baseline's 161.
- Wrong passes on held-out stair cells: 1 of 16.
- Real sets audited by hand: 6 of 121 items, against 15 for the baseline. Storey's 17 checks cover only part of what an examiner finds, and the report says so.
- Median review: 0.51 USD and 3.3 minutes.

**Built with:** Python, FastAPI, OpenCV, PyMuPDF, vanilla JavaScript; Meta Muse Spark 1.3 and SAM 3.1 through the Meta Model API.

**What's next:** more deterministic checks where the audits show the gap (section callouts that point at missing sheets come first), running against a city's own resubmittal records, and a pilot with one building department.
