"""Scores a predicted (wall, rooms, walkable) against CubiCasa ground truth."""
import cv2
import numpy as np
import plan as P


def door_axes(p):
    """Centre, unit normal, half thickness of a door polygon (min-area rect)."""
    (cx, cy), (a, b), ang = cv2.minAreaRect(p.astype(np.float32))
    t = np.deg2rad(ang)
    along_a = np.array([np.cos(t), np.sin(t)])
    across = np.array([-along_a[1], along_a[0]]) if a >= b else along_a
    return np.array([cx, cy]), across, min(a, b) / 2, max(a, b)


def gt_setup(d):
    h, w = d['h'], d['w']
    lab = np.zeros((h, w), np.int32)
    inside = []
    for i, (n, p) in enumerate(d['rooms'], 1):
        cv2.fillPoly(lab, [np.round(p).astype(np.int32)], i)
        inside.append(not n.startswith('Outdoor'))
    in_room = np.array([False] + inside)[lab]
    door = d['door']
    gw = (in_room | door) & ~d['wall']
    gw = cv2.morphologyEx(gw.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    gw &= ~d['wall']
    exits, sides, widths = [], [], []
    for p in d['doors']:
        c, n, ht, L = door_axes(p)
        widths.append(L)
        a = c + n * (ht + 8)
        b = c - n * (ht + 8)
        def at(q):
            x, y = int(np.clip(q[0], 0, w - 1)), int(np.clip(q[1], 0, h - 1))
            return lab[y, x]
        la, lb = at(a), at(b)
        sides.append((a, b, la, lb, L))
        oa, ob = la == 0 or not inside[la - 1], lb == 0 or not inside[lb - 1]
        if oa != ob:
            exits.append(tuple(c))
    return dict(lab=lab, inside=inside, gw=gw, exits=exits, sides=sides, u=float(np.median(widths)) if widths else 0.0)


def room_points(lab, n, inside):
    pts = []
    for i in range(1, n + 1):
        if not inside[i - 1]:
            continue
        m = (lab == i).astype(np.uint8)
        if m.sum() < 400:
            continue
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        y, x = np.unravel_index(int(dt.argmax()), dt.shape)
        pts.append((i, int(x), int(y)))
    return pts


def score(d, g, pw, proom, walk, near_scale=0.5):
    """Everything is per plan; summaries are taken later."""
    out = {}
    gtw = d['wall']
    out['wall_iou'] = float((gtw & pw).sum() / max(1, (gtw | pw).sum()))
    out['wall_prec'] = float((gtw & pw).sum() / max(1, pw.sum()))
    out['wall_rec'] = float((gtw & pw).sum() / max(1, gtw.sum()))
    # room IoU: best predicted region for each interior GT room
    ious = []
    pts = room_points(g['lab'], len(d['rooms']), g['inside'])
    for i, x, y in pts:
        m = g['lab'] == i
        ids, cnt = np.unique(proom[m], return_counts=True)
        cnt = cnt[ids > 0]; ids = ids[ids > 0]
        if not len(ids):
            ious.append(0.0)
            continue
        j = ids[cnt.argmax()]
        pm = proom == j
        ious.append(float((m & pm).sum() / (m | pm).sum()))
    out['room_iou'] = ious
    # door recall: both door sides meet in walkable inside a local window, with a path under 3 door widths
    u = g['u']
    op = []
    for a, b, la, lb, L in g['sides']:
        x0, x1 = int(min(a[0], b[0]) - u), int(max(a[0], b[0]) + u)
        y0, y1 = int(min(a[1], b[1]) - u), int(max(a[1], b[1]) + u)
        x0, y0 = max(0, x0), max(0, y0)
        crop = walk[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        s = np.array([a[0] - x0, a[1] - y0]), np.array([b[0] - x0, b[1] - y0])
        dd = P.geodesic(crop, [tuple(s[0])], 1.0)
        yy, xx = int(np.clip(s[1][1], 0, crop.shape[0] - 1)), int(np.clip(s[1][0], 0, crop.shape[1] - 1))
        near = dd[max(0, yy - 6):yy + 7, max(0, xx - 6):xx + 7]
        op.append(bool(np.isfinite(near).any() and near.min() < 3 * u))
    out['door_open'] = op
    # distance from exits
    rows = []
    if g['exits']:
        dg = P.geodesic(g['gw'], g['exits'])
        dp = P.geodesic(walk, g['exits'])
        for i, x, y in pts:
            a = dg[y, x]
            if not np.isfinite(a) or a <= 0:
                continue
            m = (g['lab'] == i) & np.isfinite(dp)
            # snap: room's reachable pixel nearest the GT point
            if m.any():
                ys, xs = np.nonzero(m)
                k = ((ys - y) ** 2 + (xs - x) ** 2).argmin()
                b = dp[ys[k], xs[k]]
            else:
                b = np.inf
            rows.append((float(a), float(b)))
    out['dist'] = rows
    return out


def summary(res):
    r = [x for rs in res for x in rs['room_iou']]
    o = [x for rs in res for x in rs['door_open']]
    dd = np.array([x for rs in res for x in rs['dist']], float).reshape(-1, 2)
    fin = np.isfinite(dd[:, 1])
    rel = (dd[:, 1] - dd[:, 0]) / dd[:, 0]
    plan_ok = [np.mean([np.isfinite(b) and abs(b - a) / a <= .15 for a, b in rs['dist']]) for rs in res if rs['dist']]
    return dict(plans=len(res), wall_iou=round(float(np.mean([x['wall_iou'] for x in res])), 3), wall_prec=round(float(np.mean([x['wall_prec'] for x in res])), 3),
                wall_rec=round(float(np.mean([x['wall_rec'] for x in res])), 3),
                room_iou=round(float(np.mean(r)), 3) if r else None, room_iou50=round(float(np.mean(np.array(r) >= .5)), 3) if r else None,
                door_recall=round(float(np.mean(o)), 3) if o else None, rooms_scored=len(dd),
                reach=round(float(fin.mean()), 3) if len(dd) else None,
                within15=round(float((np.abs(rel[fin]) <= .15).sum() / max(1, len(dd))), 3),
                shortcut=round(float((rel[fin] < -.15).sum() / max(1, len(dd))), 3),
                detour=round(float((rel[fin] > .15).sum() / max(1, len(dd))), 3),
                med_abs_err=round(float(np.median(np.abs(rel[fin]))), 3) if fin.any() else None,
                plans_all_within15=round(float(np.mean(np.array(plan_ok) >= .9)), 3) if plan_ok else None)
