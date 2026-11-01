"""Hand paths on two scanned non-residential sheets. Waypoints read by eye from the 1500-wide overlay, working image is 2 times that for bme_f."""
import json, sys
sys.path.insert(0, 'lab/walls')
import numpy as np, cv2, truepath_lib as T
O = 'data/lab/walls/'
PATHS = {'bme_f': (2.0, {'P1 west wing room to stair': [(285, 620), (335, 620), (350, 520), (390, 525)], 'P2 south wing room to stair': [(400, 850), (400, 805), (620, 790), (670, 755)], 'P3 east wing room to stair': [(1060, 720), (1010, 720), (1010, 560), (960, 530)], 'P4 west wing to south stair, crosses wings': [(285, 620), (335, 620), (350, 780), (620, 790), (670, 755)]}),
         'schenley': (2227 / 1500, {'P1 classroom 103 to main entry': [(300, 690), (300, 628), (700, 628), (750, 690), (750, 770)], 'P2 classroom 110 to main entry': [(1120, 690), (1120, 628), (800, 628), (750, 690), (750, 770)], 'P3 auditorium to main entry': [(750, 500), (880, 500), (880, 615), (800, 628), (750, 690), (750, 770)]})}
out = {}
for k, (s, ps) in PATHS.items():
    z = np.load(O + f'scan_{k}.npz'); walk = z['walk']
    for n, p in ps.items():
        q = np.array(p, float) * s
        L = float(np.hypot(*np.diff(q, axis=0).T).sum())
        g8, t, pts, sn = T.route(walk, q[0], q[-1])
        snap = [float(np.hypot(sn[0][1] - q[0][0], sn[0][0] - q[0][1])), float(np.hypot(sn[1][1] - q[-1][0], sn[1][0] - q[-1][1]))]
        out[f'{k} {n}'] = dict(hand_px=round(L), geo_px=None if t is None else round(t), err=None if t is None else round((t - L) / L, 3), snap_px=[round(x) for x in snap])
        print(k, n, out[f'{k} {n}'])
json.dump(out, open(O + 'handpaths.json', 'w'), indent=1)
