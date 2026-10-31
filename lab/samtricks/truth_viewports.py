"""Hand truth, written from grid views before any model call on these sheets. One labeller, not a licensed reviewer. All boxes are 0-1000 grid [x1, y1, x2, y2].

req: every floor plan drawn on the sheet, walls plus attached stairs and porches, no dimension strings or grid bubbles.
opt: real plan views that are not floor plans (roof, reflected ceiling) and enlarged partial plans. A box on one neither scores nor costs.
Key plans are not viewports. An empty req is a sheet with no floor plan.
kind: single, several, none (no floor plan on the sheet).
"""
V = {
    # dev
    "n52": dict(kind="single", opt=[], req=[[307, 137, 640, 672]]),
    "c3": dict(kind="single", req=[[672, 97, 785, 720]], opt=[[77, 101, 247, 379], [287, 49, 457, 379]]),
    "u3": dict(kind="single", opt=[], req=[[97, 101, 650, 712]]),
    "a7": dict(kind="several", req=[[347, 135, 490, 349], [97, 607, 240, 821]], opt=[[335, 607, 365, 821], [457, 607, 487, 821]]),
    "af1": dict(kind="single", opt=[], req=[[120, 359, 787, 757]]),
    "hv1": dict(kind="single", opt=[], req=[[175, 202, 860, 670]]),
    "c14": dict(kind="none", req=[], opt=[]),
    "n91": dict(kind="none", req=[], opt=[]),
    "n122b": dict(kind="several", opt=[], req=[[135, 112, 285, 420], [135, 547, 280, 873]]),
    "bk1": dict(kind="single", opt=[], req=[[87, 97, 915, 887]]),
    # held out
    "n161": dict(kind="single", opt=[], req=[[315, 139, 645, 671]]),
    "n79": dict(kind="none", req=[], opt=[]),
    "n133": dict(kind="several", opt=[], req=[
        [117, 116, 210, 356], [271, 103, 327, 180], [357, 101, 420, 180], [472, 45, 532, 176], [577, 97, 650, 176], [702, 90, 768, 176],
        [260, 262, 320, 405], [360, 255, 418, 416], [440, 289, 500, 412], [557, 311, 620, 416], [682, 337, 745, 420], [790, 276, 872, 424], [895, 337, 960, 416],
        [107, 581, 200, 656], [212, 547, 307, 652], [600, 577, 672, 652], [685, 600, 757, 652], [790, 547, 882, 652], [915, 543, 950, 652],
        [330, 615, 440, 896], [472, 645, 585, 900], [110, 810, 180, 892], [215, 802, 292, 892], [620, 795, 672, 892]]),
    "c11": dict(kind="several", opt=[], req=[[172, 201, 285, 825], [387, 202, 499, 825], [602, 202, 717, 795]]),
    "c16": dict(kind="several", opt=[], req=[[147, 176, 258, 802], [342, 176, 452, 802], [537, 176, 647, 802], [730, 176, 840, 802]]),
    "c7": dict(kind="none", req=[], opt=[]),
    "b2": dict(kind="single", req=[[605, 116, 829, 575]], opt=[[75, 22, 210, 401], [270, 22, 485, 397]]),
    "b9": dict(kind="several", opt=[], req=[[191, 332, 414, 706], [545, 300, 767, 745]]),
    "b14": dict(kind="several", opt=[], req=[[105, 420, 328, 810], [390, 420, 612, 810], [668, 425, 892, 798]]),
    "b18": dict(kind="none", req=[], opt=[]),
    "u2": dict(kind="single", opt=[], req=[[97, 101, 650, 712]]),
    "u9": dict(kind="single", opt=[], req=[[97, 101, 650, 712]]),
    "u11": dict(kind="several", opt=[], req=[[145, 556, 420, 862], [515, 556, 791, 862]]),
    "u7": dict(kind="none", req=[], opt=[]),
    "a8": dict(kind="several", opt=[], req=[[467, 142, 642, 356], [467, 611, 642, 825]]),
    "a9": dict(kind="none", req=[], opt=[[144, 131, 304, 369], [412, 131, 512, 367], [312, 613, 472, 851], [597, 613, 672, 851]]),
    "a11": dict(kind="none", req=[], opt=[]),
    "af2": dict(kind="several", opt=[], req=[[90, 100, 547, 341], [679, 100, 782, 341]]),
    "hv2": dict(kind="single", req=[[155, 186, 880, 637]], opt=[[622, 235, 696, 292]]),
    "bk2": dict(kind="single", opt=[], req=[[90, 114, 897, 895]]),
}
