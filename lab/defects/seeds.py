"""Fixed box list on the aligned 2000 px stacks. kind: stair or shaft. Boxes read by eye from Spark overlays and checked on crops."""
SEEDS = {
 'bme_f': {'floor': '0', 'objects': {
   'A_topleft': ('stair', (310, 158, 440, 228)), 'B_topright': ('stair', (1270, 170, 1392, 245)),
   'III': ('stair', (418, 640, 548, 738)), 'II': ('stair', (1162, 655, 1275, 730)), 'I': ('stair', (795, 925, 905, 1040)),
   }},
 'habs_stelizabeths_center': {'floor': '2', 'objects': {
   'central': ('stair', (818, 782, 902, 828)), 'north': ('stair', (1003, 325, 1042, 422)),
   'west_curve': ('stair', (265, 875, 332, 992)), 'east_curve': ('stair', (1628, 878, 1697, 992)),
   'elevator': ('shaft', (805, 880, 860, 920)), 'xshaft_w': ('shaft', (100, 283, 134, 312)), 'xshaft_e': ('shaft', (1752, 310, 1790, 346)), 'xshaft_c': ('shaft', (652, 702, 682, 730)), }},
 'habs_va_bldg1': {'floor': '1', 'objects': {
   'west': ('stair', (400, 510, 496, 568)), 'east': ('stair', (1535, 515, 1618, 578)), 'central': ('stair', (1020, 495, 1088, 548)),
   'elev1': ('shaft', (920, 492, 966, 526)), 'elev2': ('shaft', (920, 528, 966, 562)), }},
 'schenley': {'floor': '1', 'objects': {
   'sw': ('stair', (160, 830, 270, 935)), 'se': ('stair', (1722, 830, 1820, 935)), 'n': ('stair', (945, 60, 1050, 185)),
   'aud_nw': ('stair', (857, 325, 895, 395)), 'aud_ne': ('stair', (1135, 325, 1170, 395)), 'aud_sw': ('stair', (862, 695, 910, 735)), 'aud_se': ('stair', (1112, 695, 1160, 735)),
   'vent_w': ('shaft', (345, 700, 395, 738)), 'vent_e': ('shaft', (1600, 700, 1650, 738)), }},
 'mml_capitol': {'floor': 'main', 'objects': {
   'hall': ('stair', (445, 1265, 548, 1470)), 'rear': ('stair', (760, 160, 890, 458)), 'front': ('stair', (760, 1700, 890, 1895)), }},
}

# Floors where each object really exists, read by eye from the clean crop grids. Ordered top to bottom.
TRUTH = {
 'bme_f': {'A_topleft': ['1', '0', '-1'], 'B_topright': ['1', '0', '-1'], 'III': ['2', '1', '0', '-1'], 'II': ['2', '1', '0', '-1'], 'I': ['2', '1', '0', '-1']},
 'habs_stelizabeths_center': {'central': ['4', '3', '2', '1', '-1'], 'north': ['2', '1'], 'west_curve': ['4', '3', '2', '1', '-1'], 'east_curve': ['4', '3', '2', '1', '-1'],
                              'elevator': ['4', '3', '2', '1', '-1'], 'xshaft_w': ['3', '2', '1'], 'xshaft_e': ['4', '3', '2', '1'], 'xshaft_c': ['3', '2', '1', '-1']},
 'habs_va_bldg1': {'west': ['4', '3', '2', '1', '0'], 'east': ['3', '2', '1', '0'], 'central': ['4', '3', '2', '1', '0'], 'elev1': ['4', '3', '2', '1', '0'], 'elev2': ['4', '3', '2', '1', '0']},
 'schenley': {'sw': ['2', '1', '0', '-1'], 'se': ['2', '1', '0', '-1'], 'n': ['2', '1', '0', '-1'], 'aud_nw': ['2', '1'], 'aud_ne': ['2', '1'], 'aud_sw': ['2', '1'], 'aud_se': ['2', '1'],
              'vent_w': ['2', '1', '0'], 'vent_e': ['2', '1', '0']},
 'mml_capitol': {'hall': ['upper', 'main'], 'rear': ['upper', 'main', 'ground'], 'front': ['upper', 'main', 'ground']},
}
# Corridors: (floor, axis, across_low, across_high, along_at, along_from, along_to), 'h' runs along x, read by eye from crops of the corridors.
CORRIDORS = {
 'bme_f': [('0', 'v', 334, 402, 860, 790, 960), ('0', 'h', 1054, 1109, 1100, 1040, 1200)],
 'habs_stelizabeths_center': [('1', 'h', 778, 856, 470, 420, 560), ('1', 'h', 778, 856, 1500, 1450, 1590)],
 'habs_va_bldg1': [('1', 'h', 572, 618, 700, 650, 780), ('1', 'h', 572, 618, 1300, 1250, 1380)],
 'schenley': [('1', 'h', 748, 822, 500, 440, 600), ('1', 'h', 748, 822, 1450, 1390, 1540)],
}
# Exit openings to ink shut: (floor, x1, y1, x2, y2, wall thickness)
EXITS = {
 'habs_stelizabeths_center': [('1', 278, 989, 306, 989, 6), ('1', 1656, 989, 1684, 989, 6)],
 'habs_va_bldg1': [('1', 895, 738, 1080, 738, 5)],
 'mml_capitol': [('main', 450, 1598, 537, 1598, 9), ('main', 568, 563, 652, 563, 9)],
}
