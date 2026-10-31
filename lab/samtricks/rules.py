"""Decision rules, frozen in the manifest before any model call. A later change must be reported beside this version and marked post hoc.

Item 1, viewports. 30 sheets, 8 sets, split in reg.DEV (10) and reg.HELD (20). SAM concept "floor plan" on the 2000 px sheet.
  Conf floor: the value in CONF_GRID with best F1 on dev. Ties take the lower value. Frozen before held-out is scored.
  Hit: one-to-one greedy match on box IoU >= IOU_V against a req box. A box on an opt box is ignored. Any other box is a false viewport.
  Wrong pass: a false viewport on a sheet with no floor plan. Second danger: a req viewport with no box, because panel cropping then never looks at it.
  Clipped: a hit that covers under CLIP of the truth box, so the crop cuts off part of the plan.
  Spark: lab/align/sheet_read.py P_ONE and schema, default effort, same sheet, one box a sheet.
  Baselines: fixed left 78 percent of the sheet (sheets.title_crops heuristic) and the largest ink component.
  Replace rule: SAM replaces Spark for cropping if, on held-out, its recall is at least Spark's and it has at most 1 false viewport on no-plan sheets.

Item 2, words. Right: the top-conf mask centre falls in a req box (RIGHT). Any: some mask centre does. False fire: a mask centre outside every req and opt box.
  On a sheet where the word is absent, one or more false fires count as a false sheet.
  Keep: right on at least 75 percent of present cases and false on at most 1 absent sheet. Needs more: right on 50 to 75 percent. Otherwise drop.
  Baselines: always-absent for every word, fixed right strip (x >= 900) for title block.

Item 3, circles. Mask box centre within TAG_R crop px of a door tag centre at 150 dpi, one-to-one, and box side within 0.6 to 1.6 of the tag diameter.
  Count: number of masks that pass the size filter, set against the schedule mark count. Baseline: vector circles of the tag size holding a digit.

Item 4, clouds. Hit: one-to-one IoU >= IOU_C with a truth cloud, boxes in sheet points. False: a box with no truth cloud at IoU >= 0.1.
  Delta hit: centre of the predicted box within DELTA_R points of a truth delta centre.
  Baselines: red pixel filter on the raster, and vector arc chains (vector_clouds.candidates with gap 3.5, unchanged).
  Keep: clouds recall >= 0.8 with false boxes <= 1 per sheet. Needs more: recall 0.5 to 0.8. Otherwise drop.

Item 5, Spark video. Scorer: lab/audit/verify.py one-to-one match, on lab/audit/truth.json, buildings bme_k and habs_va_bldg1.
  Comparison: the three cached image runs in data/lab/audit/spark/<b>_floors{0,1,2}.json. Baseline: always-present.
  Keep: pairs right >= image mean, cost per call <= image mean, and frames reported == floors. Needs more: within 3 pairs of image mean. Otherwise drop.

Item 6, prompt order. Sheets CACHE_SHEETS, prompt and schema from lab/bakeoff/e1_meta.py, low effort, 2000 px. Score: truth_meta num, level, scale.
  Arms: text first, image first, two runs each. Holds: image first total >= text first total minus 2 of 36 fields.
  Saving: cached_tokens on a second question on the same image, called only after the first returns.

Item 7, version tracking. Clip of two frames. Same object: mask box IoU >= IOU_T with the prompted box on the other frame.
  Ghost: a mask with area on a frame where the object does not exist. Control: objects away from the change. Baseline: pixel diff of the two frames.
  Keep: all objects held and no ghost. Needs more: objects held but ghosts remain, usable only with a diff gate. Otherwise drop.
"""
CONF_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
IOU_V = 0.5
CLIP = 0.9
TAG_R = 12
IOU_C = 0.3
DELTA_R = 15
IOU_T = 0.5
CACHE_SHEETS = ["n161", "c3", "u2", "af1", "hv1", "a7"]
VIDEO_BUILDINGS = ["bme_k", "habs_va_bldg1"]
