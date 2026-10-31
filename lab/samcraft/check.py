"""Self-check with no API call: prompt grammar, scoring, the snap and the mask paste."""
import numpy as np

import core as c
from snap import snap

assert c.obj_prompt([dict(box=(1, 2, 30, 40), pos=[(5, 6)], neg=[(7, 8)])], 100, 50) == "<0f>1<|box;x1=1;y1=2;x2=30;y2=40;w=100;h=50|><|point;x=5;y=6;w=100;h=50|>-<|point;x=7;y=8;w=100;h=50|>"
a = np.zeros((10, 10), bool)
a[:5] = True
assert c.iou(a, a) == 1 and c.iou(a, ~a) == 0
ink = np.zeros((200, 200), bool)
ink[40:44, 30:170] = ink[140:144, 30:170] = True
ink[40:144, 30:34] = ink[40:144, 160:164] = True
x1, y1, x2, y2 = snap((45, 55, 150, 130), ink)
assert (x1, y1, x2, y2) == (33, 43, 160, 140), (x1, y1, x2, y2)
p = c.paste((2, 2, 4, 4), np.ones((2, 2), bool), (6, 6))
assert p.sum() == 4 and p[2, 2] and not p[0, 0]
print("ok")
