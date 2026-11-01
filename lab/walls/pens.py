"""Can Spark pick the wall pens? One image per pen, ask which show cut walls. usage: pens.py"""
import json, sys
sys.path.insert(0, 'lab'); sys.path.insert(0, 'lab/walls')
import cv2, fitz, numpy as np
import metalab as m, vec
D = 'data/plans/'
O = 'data/lab/walls/'
SHEETS = {'clinic_f1': ('nibs_clinic/2012-03-27-ClinicRedactedDrawings/Clinic_052_Composite First Floor Plan.pdf', 0), 'clinic_f2': ('nibs_clinic/2012-03-27-ClinicRedactedDrawings/Clinic_059_Composite Second Floor Plan.pdf', 0), 'capitol': ('mml_capitol/THE-CAPITOL.pdf', 3), 'baxter': ('mml_baxter/THE-BAXTER.pdf', 3), 'leconte': ('usc_leconte/USC_LeConte_Bid_Drawings.pdf', 2)}
SC = {'type': 'object', 'additionalProperties': False, 'required': ['wall_images'], 'properties': {'wall_images': {'type': 'array', 'items': {'type': 'integer'}}}}
PR = 'Each numbered image shows the strokes of one pen from the same architectural floor plan sheet, in the same position. Images are numbered from 0 in the order given. Which images contain building walls drawn in plan, meaning long pairs of parallel lines that bound rooms and corridors? Include a pen if it draws any such walls, new or existing. Leave out pens that only draw text, grid lines, dimensions, furniture, fixtures, stairs or symbols. Return the image numbers.'
res = {}
for k, (f, pg) in SHEETS.items():
    g = fitz.open(D + f)[pg]
    s = vec.strokes(g)
    W, H = g.rect.width, g.rect.height
    sc = 1300 / max(W, H)
    pens = sorted((p for p in s if np.hypot(s[p][:, 2] - s[p][:, 0], s[p][:, 3] - s[p][:, 1]).sum() > 500), reverse=True)
    ims = []
    for p in pens:
        im = np.full((int(H * sc), int(W * sc), 3), 255, np.uint8)
        im[vec.draw(s[p], (0, 0, W, H), sc, 1)] = 0
        ims.append(im)
    r, i = m.spark(PR, ims, schema=SC, effort='low')
    res[k] = dict(pens=pens, picked=[pens[j] for j in r['wall_images'] if j < len(pens)], usd=i['usd'])
    print(k, res[k])
json.dump(res, open(O + 'pens.json', 'w'), indent=1)
