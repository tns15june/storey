"""Fixes after reading overlays: Abbot attic by its section bubble (blocks gave 23 inliers and a wrong fit), Leonard ground with a fainter-ink threshold and a finer angle search. No API."""
import warnings; warnings.filterwarnings("ignore")
import json, sys
import fitz, cv2, numpy as np
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed"); sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/align")
import reg, stacks as st
from stacks4 import page, plan_rect, P
from stacks5 import gray_ink


def abbot_attic():
    pg = fitz.open(P / "mml_abbot/THE-ABBOT.pdf")[1]
    z = 1.744
    rs = pg.search_for("A3.0")
    print([tuple(round(v) for v in r) for r in rs])
    pan = {"foundation": (84, 872), "main": (1241, 872), "upper": (1241, 40), "attic": (84, 40)}
    bub = {}
    for k, (x, y) in pan.items():
        # the section bubble sits left of the plan: the leftmost A3.0 inside this panel
        c = [r for r in rs if x <= r.x0 <= x + 1147 and y <= r.y0 <= y + 823]
        r = min(c, key=lambda r: r.x0)
        bub[k] = ((r.x0 + r.x1) / 2 - x, (r.y0 + r.y1) / 2 - y)
    print({k: tuple(round(v, 1) for v in b) for k, b in bub.items()})
    d = st.OUT / "mml_abbot"; h = json.load(open(d / "H.json"))
    ims = []
    for k in ("foundation", "main", "upper", "attic"):
        x, y = pan[k]
        ims.append(page(pg, z, clip=fitz.Rect(x, y, x + 1147, y + 823)))
    hh = min(i.shape[0] for i in ims); ww = min(i.shape[1] for i in ims)
    ims = [i[:hh, :ww] for i in ims]
    Hs = {int(k): np.array(v) for k, v in h["H"].items()}; info = {int(k): v for k, v in h["info"].items()}
    # panel-local bubble offsets give every panel's translation to main; keep blocks for f0 and f2 only if they agree within 12 px
    for i, k in enumerate(("foundation", "main", "upper", "attic")):
        t = np.eye(3); t[0, 2] = (bub["main"][0] - bub[k][0]) * z; t[1, 2] = (bub["main"][1] - bub[k][1]) * z
        print(k, "bubble shift", t[:2, 2].round(1), "blocks shift", Hs[i][:2, 2].round(1), "scale", round(float(np.hypot(Hs[i][0, 0], Hs[i][1, 0])), 4))
        if k == "attic" or np.abs(t[:2, 2] - Hs[i][:2, 2]).max() > 12:
            Hs[i] = t; info[i] = dict(info[i], via="bubble")
    st.save("mml_abbot", ims, Hs, 1, info)


def leonard_ground():
    d = fitz.open(P / "queens_leonard/leonard_hall.pdf")
    z = 2000 / 612
    ims = [page(d[i], z) for i in range(1, 6)]
    rects = [plan_rect(d[i]) for i in range(1, 6)]
    boxes = [tuple(int(v * z) for v in (r.x0, r.y0, r.x1, r.y1)) for r in rects]
    old = json.load(open(st.OUT / "queens_leonard/H.json"))
    Hs = {int(k): np.array(v) for k, v in old["H"].items()}; info = {int(k): v for k, v in old["info"].items()}
    def faint(im, top):
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        return ((hsv[..., 1] < 40) & (hsv[..., 2] < top)).astype(np.float32)
    best = None
    for top in (225, 238, 246, 250):
        a, b = faint(ims[0], top), faint(ims[1], 238)
        c = dict(mov_ink=a, ref_ink=b, mov_box=boxes[0], ref_box=boxes[1])
        c["mov_soft"], c["ref_soft"] = cv2.GaussianBlur(a, (0, 0), 1.2), cv2.GaussianBlur(b, (0, 0), 1.2)
        H0 = reg.coarse(c, scales=np.linspace(.94, 1.06, 25), angles=(-2.5, -2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 2.5))
        H = reg.blocks(c, H0=H0)
        print("top", top, c.get("inliers"), c.get("matches"), H.round(4).tolist())
        if best is None or (c.get("inliers") or 0) > best[0]:
            best = (c.get("inliers") or 0, H, c.get("matches"), top)
    Hs[0] = Hs[1] @ best[1]; info[0] = dict(inliers=best[0], matches=best[2], via="f1", ink=f"grey{best[3]}")
    st.save("queens_leonard", ims, Hs, 2, info)


if __name__ == "__main__":
    if "abbot" in sys.argv: abbot_attic()
    if "leonard" in sys.argv: leonard_ground()
