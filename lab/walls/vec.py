"""Vector sheets: exact door arcs from PDF paths, and a clean render. Page geometry in points."""
import fitz, cv2, numpy as np, warnings
warnings.filterwarnings('ignore')


def polylines(g):
    """Connected runs of line segments from every stroked path."""
    out = []
    for x in g.get_drawings():
        run = []
        for it in x['items']:
            if it[0] == 'l':
                a, b = (it[1].x, it[1].y), (it[2].x, it[2].y)
                if run and abs(run[-1][0] - a[0]) + abs(run[-1][1] - a[1]) < 0.02:
                    run.append(b)
                else:
                    if len(run) >= 2:
                        out.append(run)
                    run = [a, b]
            elif it[0] == 'c':  # Bezier arcs: sample them so one circle fit serves both encodings
                p = np.array([(q.x, q.y) for q in it[1:5]])
                t = np.linspace(0, 1, 7)[:, None]
                c = (1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3]
                c = [tuple(v) for v in c]
                if run and abs(run[-1][0] - c[0][0]) + abs(run[-1][1] - c[0][1]) < 0.02:
                    run += c[1:]
                else:
                    if len(run) >= 2:
                        out.append(run)
                    run = c
        if len(run) >= 2:
            out.append(run)
    return out


def circle_fit(p):
    A = np.c_[2 * p, np.ones(len(p))]
    b = (p ** 2).sum(1)
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    r2 = c[2] + c[0] ** 2 + c[1] ** 2
    return c[:2], np.sqrt(max(r2, 0))


def door_arcs(g, pts_per_m, rmin=0.6, rmax=1.35):
    """Quarter-ish arcs of door radius. Returns [(hinge, open end, leaf-side end, radius_m)] with both arc ends; leaf end unknown at this stage."""
    arcs = []
    for run in polylines(g):
        p = np.array(run)
        if len(p) < 5:
            continue
        L = np.hypot(*np.diff(p, axis=0).T).sum()
        c, r = circle_fit(p)
        if not (rmin * pts_per_m <= r <= rmax * pts_per_m):
            continue
        d = np.hypot(*(p - c).T)
        if abs(d - r).max() > 0.02 * r + 0.15:
            continue
        ang = np.unwrap(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))
        sweep = abs(ang[-1] - ang[0])
        if not np.deg2rad(60) <= sweep <= np.deg2rad(105):
            continue
        arcs.append((c, p[0], p[-1], r / pts_per_m, np.rad2deg(sweep)))
    return arcs


def render(g, dpi):
    pix = g.get_pixmap(dpi=dpi)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3][:, :, ::-1].copy()


def strokes(g):
    """Segments by pen: {(width): array [x0,y0,x1,y1]} for stroked paths, rectangles expanded."""
    import collections
    segs = collections.defaultdict(list)
    for x in g.get_drawings():
        if x['type'] == 'f':
            continue
        w = round(x.get('width') or 0, 2)
        for it in x['items']:
            if it[0] == 'l':
                segs[w].append((it[1].x, it[1].y, it[2].x, it[2].y))
            elif it[0] == 're':
                r = it[1]
                segs[w] += [(r.x0, r.y0, r.x1, r.y0), (r.x1, r.y0, r.x1, r.y1), (r.x1, r.y1, r.x0, r.y1), (r.x0, r.y1, r.x0, r.y0)]
    return {k: np.array(v) for k, v in segs.items()}


def draw(segs, box, s, thick=1):
    """Rasterise segments to a bool mask. box in points, s pixels per point."""
    x0, y0, x1, y1 = box
    im = np.zeros((int((y1 - y0) * s), int((x1 - x0) * s)), np.uint8)
    p = np.round((segs - np.array([x0, y0, x0, y0])) * s).astype(np.int32)
    for a in p:
        cv2.line(im, (int(a[0]), int(a[1])), (int(a[2]), int(a[3])), 1, thick)
    return im > 0
