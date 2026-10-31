"""Clinic, both floors. Walls from the heavy pen, doors from vector arcs. Truth from the IFC model of the same building (independent of the sheet)."""
import json, sys
sys.path.insert(0, '/Users/tsharma/Github/Meta/lab/walls')
import collections
import cv2, fitz, numpy as np
import ifc, plan as P, score as S, vec
D = '/Users/tsharma/Github/Meta/data/plans/nibs_clinic/'
O = '/Users/tsharma/Github/Meta/data/lab/walls/'
PPM = 1000 / (150 * 0.352778)
PXM = 40.0                 # raster pixels per metre
SC = PXM / PPM             # pixels per point
U = 0.9 * PXM              # one door width in pixels
BOX = (500, 100, 2400, 1800)
SHEETS = {1: ('2012-03-27-ClinicRedactedDrawings/Clinic_052_Composite First Floor Plan.pdf', '2012-03-27-ClinicRedactedDrawings/Clinic_161_Life Safety Composite First Floor Plan.pdf'),
          2: ('2012-03-27-ClinicRedactedDrawings/Clinic_059_Composite Second Floor Plan.pdf', '2012-03-27-ClinicRedactedDrawings/Clinic_162_Life Safety Composite Second Floor Plan.pdf')}


def px(p):
    return (np.asarray(p, float) - np.array(BOX[:2])) * SC


def fit_rigid(a, b, rot):
    """ICP with known scale; a in IFC metres, b sheet points. rot: 2x2 metres-to-points including the y flip. Returns t, median residual in metres."""
    A = a @ rot.T * PPM
    t = np.median(b, 0) - np.median(A, 0)
    for it in range(40):
        d = np.hypot(*((A + t)[:, None] - b[None]).transpose(2, 0, 1))
        nn = d.argmin(1)
        keep = d.min(1) < max(3 * PPM, 60 - it * 2)
        if keep.sum() < 5:
            return t, 1e9, 0
        t = np.median(b[nn[keep]] - A[keep], 0)
    d = np.hypot(*((A + t)[:, None] - b[None]).transpose(2, 0, 1)).min(1)
    return t, float(np.median(d) / PPM), int((d < 0.6 * PPM).sum())


PEN2 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0  # also take this pen's segments longer than this many metres
STAIR_RING = len(sys.argv) > 2
DBL = int(sys.argv[3]) if len(sys.argv) > 3 else 0
NO2AS1 = len(sys.argv) > 4
SHIFT = 1.0  # the IFC door origin sits on one jamb; residual to the sheet arcs drops from 0.48 m to 0.09 m with this half-width shift
m = ifc.Model(D + 'Clinic_A_20110906_optimized.ifc')
# spaces by storey (first character of the room number)
spaces = {}
for s in m.of('IFCSPACE'):
    a = m.a(s)
    no, nm = a[2][2:], (a[7][2:] if isinstance(a[7], str) else '')
    p = m.space_poly(s)
    spaces[s] = dict(no=no, name=nm, poly=p, floor=int(no[0]) if no[:1] in '12' else 0)
# door to spaces graph
d2s = collections.defaultdict(set)
ext = collections.Counter()
for b in m.of('IFCRELSPACEBOUNDARY'):
    a = m.a(b)
    e = ifc.ref(a[5])
    if e and m.r[e][0] == 'IFCDOOR':
        d2s[e].add(ifc.ref(a[4]))
        if '.EXTERNAL' in a[8]:
            ext[e] += 1
doors = {}
for d in m.of('IFCDOOR'):
    a = m.a(d)
    M = m.place(ifc.ref(a[5]))
    w = float(a[9]) if a[9] not in ('$',) else 0.9
    doors[d] = dict(xy=M[:2, 3] + float(a[9] if a[9] != '$' else 0.9) / 2 * SHIFT * M[:2, 0], z=M[2, 3], w=w, dirx=M[:2, 0], spaces=d2s.get(d, set()), name=a[2][2:30], ext=ext[d] > 0)
print('spaces', len(spaces), 'polys', sum(v['poly'] is not None for v in spaces.values()), 'floors', collections.Counter(v['floor'] for v in spaces.values()))
print('door z', collections.Counter(round(v['z'], 1) for v in doors.values()).most_common(5), 'ext', sum(v['ext'] for v in doors.values()))

res = {}
for fl, (f, fls) in SHEETS.items():
    g = fitz.open(D + f)[0]
    sg = vec.strokes(g)
    a36 = sg[0.36]
    a36 = a36[np.hypot(a36[:, 2] - a36[:, 0], a36[:, 3] - a36[:, 1]) >= PEN2 * PPM]
    if DBL:
        import dbl
        allseg = np.vstack([v for k, v in sg.items() if k != 0.29 or DBL == 2])
        allseg = allseg[dbl.wall_segments(allseg, PPM)]
        wall = vec.draw(allseg, BOX, SC, 1)
    else:
        wall = vec.draw(np.vstack([sg[0.79], a36]) if PEN2 else sg[0.79], BOX, SC, 1)
    h, w_ = wall.shape
    arcs = vec.door_arcs(g, PPM)
    # choose the arc end that lies on the wall axis as the opening end
    wd = cv2.dilate(wall.astype(np.uint8), P.SQ(5)) > 0

    def on_wall(H, E):
        d = (E - H) / np.hypot(*(E - H))
        pts = [px(E + d * t * PPM) for t in (0.12, 0.25)] + [px(H - d * t * PPM) for t in (0.12, 0.25)]
        return sum(0 <= int(q[1]) < h and 0 <= int(q[0]) < w_ and wd[int(q[1]), int(q[0])] for q in pts)
    cand = []
    for c, a, b, r, sw in arcs:
        s1, s2 = on_wall(c, a), on_wall(c, b)
        E, Lf = (a, b) if s1 >= s2 else (b, a)
        cand.append((px(c), px(E), px(Lf), float(max(s1, s2))))
    mids = np.array([(c[0] + c[1]) / 2 / SC + np.array(BOX[:2]) for c in cand])
    # align IFC doors of this storey to the sheet
    zs = 0.0 if fl == 1 else 4.57
    ds = {k: v for k, v in doors.items() if abs(v['z'] - zs) < 1.5}
    A = np.array([v['xy'] for v in ds.values()])
    best = None
    for th in (0, 90, 180, 270):
        t = np.deg2rad(th)
        for fy in (-1, 1):
            R = np.array([[np.cos(t), -np.sin(t)], [fy * np.sin(t), fy * np.cos(t)]])
            tt, e, n = fit_rigid(A, mids, R)
            if best is None or n > best[0]:
                best = (n, e, th, fy, R, tt)
    n, e, th, fy, R, tt = best
    print(f'floor {fl}: arcs {len(cand)} IFC doors {len(ds)} fit rot {th} flip {fy} matched {n} residual {e:.2f} m')
    toxy = lambda q: (np.asarray(q) @ R.T * PPM + tt - np.array(BOX[:2])) * SC
    # truth mask: IFC space polygons plus door rectangles
    lab = np.zeros((h, w_), np.int32)
    names = {}
    sids = [s for s, v in spaces.items() if v['floor'] == fl and v['poly'] is not None and len(v['poly']) >= 3]
    for i, s in enumerate(sorted(sids, key=lambda s: -cv2.contourArea(np.float32(spaces[s]['poly']))), 1):
        cv2.fillPoly(lab, [np.round(toxy(spaces[s]['poly'])).astype(np.int32)], i)
        names[i] = s
    door_rect = np.zeros((h, w_), np.uint8)
    for k, v in ds.items():
        c = toxy(v['xy']); dd = v['dirx'] @ R.T * np.array([1, 1])
        dn = np.array([-dd[1], dd[0]])
        hw, hd = v['w'] / 2 * PXM, 0.35 * PXM
        q = np.array([c + dd * hw + dn * hd, c + dd * hw - dn * hd, c - dd * hw - dn * hd, c - dd * hw + dn * hd]).round().astype(np.int32)
        cv2.fillConvexPoly(door_rect, q, 1)
    gt = (lab > 0) | (door_rect > 0)
    gt = cv2.morphologyEx(gt.astype(np.uint8), cv2.MORPH_CLOSE, P.SQ(3)) > 0  # only seals rounding gaps; the door rectangles are what span the walls
    # prediction
    solid = cv2.morphologyEx(wall.astype(np.uint8), cv2.MORPH_CLOSE, P.SQ(0.28 * PXM)) > 0
    bars = np.zeros((h, w_), np.uint8)
    for c, E, Lf, s in cand:
        cv2.line(bars, tuple(np.round(c).astype(int).tolist()), tuple(np.round(E).astype(int).tolist()), 1, 3)
    closed = P.close_doors(solid | (bars > 0), U)
    fp = P.footprint(closed, U, fp_gap=9.0)
    rooms = P.rooms(closed, fp, U)
    # A stair is a closed shaft entered only by its doors. Its walls can be on a lighter pen, so close its outline except at door arcs.
    # shortcut: the outline comes from the IFC here as a stand-in for the staircase box that Spark or SAM supplies in the product.
    stair_ring = np.zeros((h, w_), np.uint8)
    if STAIR_RING:
        for s0 in sids:
            if 'STAIR' in spaces[s0]['name'].upper():
                mk = np.zeros((h, w_), np.uint8)
                cv2.fillPoly(mk, [np.round(toxy(spaces[s0]['poly'])).astype(np.int32)], 1)
                stair_ring |= (cv2.dilate(mk, P.SQ(0.3 * PXM)) - mk).astype(np.uint8)
        for c, E, Lf, sc0 in cand:
            cv2.circle(stair_ring, tuple(np.round((c + E) / 2).astype(int).tolist()), int(0.55 * PXM), 0, -1)
    walk = P.walkable(solid | (stair_ring > 0), fp, U, clear=0.3)
    out = dict(arcs=len(cand), ifc_doors=len(ds))
    # door recall against the IFC
    B = toxy(A)
    C = np.array([(c[0] + c[1]) / 2 for c in cand])
    dist = np.hypot(*(B[:, None] - C[None]).transpose(2, 0, 1)) / PXM
    W = np.array([v['w'] for v in ds.values()])
    out['ifc_door_matched_0.8m'] = int((dist.min(1) < 0.8).sum())
    out['arc_with_ifc_door_0.8m'] = int((dist.min(0) < 0.8).sum())
    # room IoU
    ious = []
    pt = {}
    for i, s in names.items():
        mk = lab == i
        if mk.sum() < 0.5 * PXM ** 2:
            continue
        dt = cv2.distanceTransform(mk.astype(np.uint8), cv2.DIST_L2, 3)
        y, x = np.unravel_index(int(dt.argmax()), dt.shape)
        pt[s] = (x, y)
        ids, cnt = np.unique(rooms[mk], return_counts=True)
        cnt, ids = cnt[ids > 0], ids[ids > 0]
        if len(ids):
            pm = rooms == ids[cnt.argmax()]
            ious.append(float((mk & pm).sum() / (mk | pm).sum()))
        else:
            ious.append(0.0)
    out['rooms_scored_iou'] = len(ious); out['room_iou'] = round(float(np.mean(ious)), 3); out['room_iou50'] = round(float(np.mean(np.array(ious) >= .5)), 3)
    # exits: exterior doors on floor 1, stair rooms on both
    # 2AS1 has no door in the IFC or on the sheet, so it can be left out as an exit to see what that one stair does to floor 2
    stair = [s for s in sids if 'STAIR' in spaces[s]['name'].upper() and not (NO2AS1 and spaces[s]['no'] == '2AS1')]
    print(' stairs', [(spaces[s]['no'], spaces[s]['name']) for s in stair])
    # a stair counts as reached at any point of its polygon: treads must not hide it
    seeds = []
    for s in stair:
        i = [k for k, v in names.items() if v == s]
        if i:
            ys, xs = np.nonzero(lab == i[0])
            seeds += [(float(x), float(y)) for x, y in zip(xs[::9], ys[::9])]
    if fl == 1:
        seeds += [tuple(toxy(v['xy'])) for v in ds.values() if v['ext']]
    out['seeds'] = len(seeds)
    # a stair pixel that is not free would snap through a wall to the far side, so only free pixels seed
    nst = len(seeds) - (sum(v['ext'] for v in ds.values()) if fl == 1 else 0)
    keep = lambda mk: [q for i, q in enumerate(seeds) if i >= nst or mk[int(q[1]), int(q[0])]]
    dg = P.geodesic(gt, keep(gt))
    dp = P.geodesic(walk, keep(walk))
    rows = []
    for s, (x, y) in pt.items():
        a = dg[y, x]
        if not np.isfinite(a) or a <= 0:
            continue
        i = lab == [k for k, v in names.items() if v == s][0]
        mf = i & np.isfinite(dp)
        if mf.any():
            ys, xs = np.nonzero(mf); k = ((ys - y) ** 2 + (xs - x) ** 2).argmin(); b = dp[ys[k], xs[k]]
        else:
            b = np.inf
        rows.append((spaces[s]['no'], spaces[s]['name'], float(a) / PXM, float(b) / PXM))
    a = np.array([r[2] for r in rows]); b = np.array([r[3] for r in rows])
    fin = np.isfinite(b); rel = (b - a) / a
    out.update(rooms_truth_reachable=len(rows), reach=round(float(fin.mean()), 3), within15=round(float((np.abs(rel[fin]) <= .15).sum() / len(a)), 3),
               shortcut=round(float((rel[fin] < -.15).sum() / len(a)), 3), detour=round(float((rel[fin] > .15).sum() / len(a)), 3), med_abs=round(float(np.median(np.abs(rel[fin]))), 3))
    out['truth_rooms_total_floor'] = len(sids); out['truth_unreached'] = sum(1 for s in sids if s in pt and not np.isfinite(dg[pt[s][1], pt[s][0]]))
    res[fl] = out
    print(json.dumps(out))
    np.savez_compressed(O + f'clinic_f{fl}.npz', walk=walk, solid=solid, gt=gt, lab=lab, rooms=rooms, dp=dp, dg=dg)
    json.dump(dict(rows=rows, seeds=[(float(x), float(y)) for x, y in seeds]), open(O + f'clinic_f{fl}_rows.json', 'w'))
    v = np.full((h, w_, 3), 255, np.uint8)
    v[gt] = (225, 225, 170); v[walk & gt] = (170, 235, 170); v[walk & ~gt] = (170, 200, 255)
    v[solid] = (40, 40, 40)
    for r, (s, (x, y)) in zip(rows, list(pt.items())):
        pass
    bad = [r for r in rows if not np.isfinite(r[3]) or abs(r[3] - r[2]) / r[2] > .15]
    no2s = {spaces[s]['no']: pt[s] for s in pt}
    for r in bad:
        cv2.circle(v, tuple(int(q) for q in no2s[r[0]]), 10, (255, 0, 255), -1)
    cv2.imwrite(O + f'clinic_f{fl}.jpg', cv2.resize(v, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA))
    print(' worst', sorted([(r[0], r[1][:18], round(r[2], 1), round(r[3], 1)) for r in bad], key=lambda r: -r[2])[:10])
json.dump(res, open(O + 'clinic.json', 'w'), indent=1)
