# Submission

Deadline: Wed 4 Nov 2026, 9:00 PM ET, which is 07:30 IST on Thu 5 Nov. Aim for 22:30 IST on Wed 4 Nov.

## Checklist

- [x] Product: upload a drawing set, read the sheets, follow the stairs, run the checks, draft the letter; second review; Houston, Philadelphia and Seattle
- [x] Evaluation: 33 sets on code frozen by hash (`eval/frozen.json`), two held out until after the freeze, judged by two Claude judges not told the method (`eval/judge/`)
- [x] Repository public, MIT licence, README with "Built during the hackathon" and "Pre-existing assets"
- [x] No keys in the repository history
- [x] Tests: 1,905 passed at the freeze, 2,091 after the review fixes; GitHub Actions runs them on every push (`.github/workflows/test.yml`)
- [x] Outside code review of 3 Nov: every finding reproduced as a failing test and fixed (`docs/review-response.md`)
- [x] Live demo with no key: https://tns15june.github.io/storey/
- [x] Demo video, 2:52, narrated in the author's voice, plays with no login: https://tns15june.github.io/storey/demo.html (how it was made: `tools/video/`)
- [x] Every runtime model is Meta's: Muse Spark 1.3 and SAM 3.1 through `https://api.meta.ai/v1`, one client in `storey/meta.py`
- [ ] Submit the form (below)
- [ ] After submitting, push nothing to this repository until judging ends, or it may count as late

## How to submit

1. Open https://cerebralvalley.ai/e/meta-ai-developer-hackathon and log in with the email used to apply.
2. Choose Submit project.
3. Copy each answer below into the matching field. Track: Indie. Team: one person. Theme, if asked: Build for Your World.
4. Video: the form's video field shows a YouTube placeholder, so upload `app/web/video/storey.mp4` at studio.youtube.com as Unlisted and paste that link. If you skip YouTube, https://tns15june.github.io/storey/demo.html plays it in any browser.
5. Submit, and keep a screenshot of the confirmation.

Optional, after submitting: share the demo for the Community Vote and Most Viral prizes, for example:

> Built Storey for the Meta AI hackathon: AI plan review for city building departments. It reads a permit drawing set, follows every stair floor to floor and drafts the correction letter, on Muse Spark + SAM 3.1. It never approves. Demo: https://tns15june.github.io/storey/demo.html

## Form answers

These are the questions on the Cerebral Valley form, in its order. Numbers come from `app/web/eval.json` (frozen run, 1 November 2026).

**Team Name:** Storey

**Team Members:** already filled in (you).

**Which track are you submitting to?** Indie Developer Track

**Which problem statement did you build towards?** Build for Your World

**Most Viral Track, X post link (optional):** only if you post on X. The post text is under How to submit.

**Project Description**

Storey is an AI plan review assistant for city building departments. Before a building permit is issued, a plan examiner reads the whole drawing set by hand to check that people can get out of the building: that every stair continues floor to floor to an exit, that exits are wide enough for the occupant load, and that the set agrees with itself. In 2021 Portland's city auditor found that more than 80% of standard permit plans needed corrections, and each correction cycle costs the applicant weeks. Seattle's own AI pilot passed 6.4% of the checks a human examiner later sent back, and nobody looks again at what was passed.

An examiner uploads a PDF set. In about three minutes, for about half a dollar, Storey lists every sheet, follows every stair across every floor in a stair-by-floor grid, and raises a correction list. Each item gives its sheet, its place on the drawing, the value read, the rule for Houston, Philadelphia or Seattle, and a crop. Storey also lists what it could not check and drafts the comment letter. On a resubmittal it checks each earlier comment at its place and lists changes that no revision cloud marks. It has no pass state: whatever it cannot read shows as Cannot tell, and the examiner decides every item.

On 33 evaluation sets run once on code frozen by hash, Storey raised 184 of 184 planted items on generated sets, against 81 of 177 for Muse Spark asked directly, and showed 1 wrong pass in 16 held-out stair cells. On real sets audited by hand it found 6 of 121 items, against 15 for Spark asked directly, because most of those items are kinds its 17 checks do not cover yet. That is the next work.

**Public GitHub Repository:** https://github.com/tns15june/storey

**3-Minute Demo Video:** your YouTube link (see How to submit, step 4), or https://tns15june.github.io/storey/demo.html

**Describe your use of Meta products to build the submitted project**

Every model call Storey makes at runtime goes to the Meta Model API (https://api.meta.ai/v1) through one client, storey/meta.py. It uses two models.

Muse Spark 1.3 reads each sheet's number, title, level, scale and plan box. It orders the floors in one joint call, and compares all floors of a building in one multi-image call, three times, to list where each stair is drawn. Where a sheet has no text layer it reads schedules and fine text in tiles, with two reads that must agree. It answers one blind yes/no question on a crop of every cell where a stair was found, lists revision clouds on a resubmittal, and joins each letter comment's instruction to its item.

SAM 3.1 finds the floor-plan viewports on a sheet, at about a ninth of the cost of a Spark call. Its video tracker runs through the registered floor plans, stacked as the frames of a short video, and outlines each stair on every floor where it was found.

Code does the rest: it registers the floors to each other, measures widths and travel distances on vector drawings, applies rule rows for three cities and sets every severity. A review costs a median of 0.51 USD. Muse Spark asked directly, with every sheet in one call, is also the baseline the evaluation compares against.

**Why are Meta's models important to your solution?**

Following a stair through a building means finding the same stair at the same place on every floor plan. Muse Spark takes all of a building's floor plans in one request, so it compares the floors side by side instead of reading each sheet alone. It is cheap enough to run that comparison three times in every review, and a whole review still costs about half a dollar.

SAM 3.1's video tracker does a job it was not designed for. Storey stacks the registered floor plans as the frames of a short video, and the tracker follows each stair from one floor's frame to the next and outlines it on every floor where it was found. On line drawings SAM's text and click prompts failed, but the tracker with one box per stair matched 15 of 15 floor-and-stair pairs on the building where we first tested it.

We also measured where both models fail and designed around it. Spark alone claimed 10 of 15 planted stair erasures across three runs, and SAM's masks stayed on 14 of 20 emptied stair rooms. So no model decides alone: a stair counts as Found only when Spark's three runs, ink at the registered place and a blind crop question agree, and code sets every severity. On the held-out sets that left 1 wrong pass in 16 stair cells.

**Provide feedback from your experience using partner technologies (optional):** We did not use Fireworks, Modal or Baseten. Every model call went to the Meta Model API.

**The survey questions** ask for your own view, so answer them yourself:
- How likely are you to build with Meta Model API; your overall opinion of it; how likely you are to recommend it (0 to 5).
- How likely you are to recommend Muse Code: Storey was not written with Muse Code, so skip this one if the form allows.
- Biggest obstacle: the measured one was "Code or answers from the model that were wrong or not useful". Spark claimed 10 of 15 erased stairs, which is why a stair needs three agreeing signals.
- Compared with other AI models: "We've used other models but couldn't make a clear comparison". Claude agents audited the same drawings for the truth files, but no model was run head to head with Spark on Storey's task.
- What matters most when choosing a model: for Storey it was quality of the answers, because a wrong pass is the error that matters.
- The consent box lets Meta and Cerebral Valley use and promote the submission. The rules already give the organizers a licence to every submission.
