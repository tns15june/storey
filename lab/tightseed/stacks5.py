"""mml_abbot panels, Leonard's two podium floors, dev stacks from the align run, canvas pad for wfu. No API."""
import warnings; warnings.filterwarnings("ignore")
import json, sys
import fitz, cv2, numpy as np
sys.path.insert(0, "lab/tightseed"); sys.path.insert(0, "lab/align")
import reg, stacks as st
from stacks4 import page, plan_rect, P


def gray_ink(im):
    """Grey linework only: Leonard's walls sit above reg.ink's threshold, and colour fills and red arrows must not count."""
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 1] < 40) & (hsv[..., 2] < 238)
    return m.astype(np.float32)


def ctx_gray(mov, ref, mbox, rbox):
    a, b = gray_ink(mov), gray_ink(ref)
    c = dict(mov_ink=a, ref_ink=b, mov_box=mbox, ref_box=rbox)
    c["mov_soft"], c["ref_soft"] = cv2.GaussianBlur(a, (0, 0), 1.2), cv2.GaussianBlur(b, (0, 0), 1.2)
    return c


def abbot():
    d = fitz.open(P / "mml_abbot/THE-ABBOT.pdf")
    pg = d[1]
    z = 1.744
    # panel rules read off the sheet at 1.296 pt per display px; right-hand column is plans, left is foundation and attic
    pan = {"foundation": (84, 872, 1231, 1695), "main": (1241, 872, 2386, 1695), "upper": (1241, 40, 2386, 862), "attic": (84, 40, 1231, 862)}
    ims = [page(pg, z, clip=fitz.Rect(*pan[k])) for k in ("foundation", "main", "upper", "attic")]
    h = min(i.shape[0] for i in ims); w = min(i.shape[1] for i in ims)
    ims = [i[:h, :w] for i in ims]
    # plan only: drop the panel title strip and the red disclaimer at the top of the upper panel
    b = (120, 120, w - 40, h - 150)
    Hs, info = {}, {}
    Hs[1] = np.eye(3); info[1] = dict(inliers=None, matches=None)
    for i, r in ((0, 1), (2, 1), (3, 2)):
        c = st.ctx(ims[i], ims[r], b, b)
        H = reg.blocks(c)
        Hs[i] = Hs[r] @ H; info[i] = dict(inliers=c.get("inliers"), matches=c.get("matches"), via=f"f{r}")
    st.save("mml_abbot", ims, Hs, 1, info)


def leonard():
    d = fitz.open(P / "queens_leonard/leonard_hall.pdf")
    z = 2000 / 612
    ims = [page(d[i], z) for i in range(1, 6)]
    rects = [plan_rect(d[i]) for i in range(1, 6)]
    boxes = [tuple(int(v * z) for v in (r.x0, r.y0, r.x1, r.y1)) for r in rects]
    old = json.load(open(st.OUT / "queens_leonard/H.json"))
    Hs = {int(k): np.array(v) for k, v in old["H"].items()}; info = {int(k): v for k, v in old["info"].items()}
    for i, r, sc, an in ((1, 2, np.linspace(.9, 1.1, 21), (-2, -1, 0, 1, 2)), (0, 1, np.linspace(.9, 1.1, 21), (-2, -1, 0, 1, 2))):
        c = ctx_gray(ims[i], ims[r], boxes[i], boxes[r])
        H0 = reg.coarse(c, scales=sc, angles=an)
        H = reg.blocks(c, H0=H0)
        print(i, "->", r, c.get("inliers"), c.get("matches"), H.round(3).tolist())
        Hs[i] = Hs[r] @ H; info[i] = dict(inliers=c.get("inliers"), matches=c.get("matches"), via=f"f{r}", ink="grey")
    st.save("queens_leonard", ims, Hs, 2, info)


def dev(name, n, ref):
    Hb = json.load(open(reg.OUT / "H_blocks.json"))
    ims = [reg.sheet(name, i) for i in range(n)]
    Hs = {i: (np.eye(3) if i == ref else np.array(Hb[f"{name}/{i}"])) for i in range(n)}
    st.save(name, ims, Hs, ref, {i: {} for i in range(n)})


def capitol_seed():
    """Seed panel frames are already aligned and much larger than the whole-sheet frames."""
    ims = [cv2.imread(str(st.ROOT / f"data/lab/seed/mml_capitol/f{i}.png")) for i in range(3)]
    st.save("mml_capitol", ims, {i: np.eye(3) for i in range(3)}, 1, {i: {} for i in range(3)})


def pad_wfu(extra=300):
    d = st.OUT / "wfu_huffman"
    h = json.load(open(d / "H.json"))
    ims = [cv2.imread(str(st.AUD / "wfu_huffman" / f"f{i}.png")) for i in range(4)]
    Hs = {int(k): np.array(v) for k, v in h["H"].items()}
    shape = (ims[1].shape[0] + extra, ims[1].shape[1], 3)
    for i, im in enumerate(ims):
        cv2.imwrite(str(d / f"f{i}.png"), reg.warp(im, Hs[i], shape, 255))
    print("wfu canvas", shape)


if __name__ == "__main__":
    what = sys.argv[1:]
    if "abbot" in what: abbot()
    if "leonard" in what: leonard()
    if "dev" in what:
        dev("bme_f", 4, 1); dev("habs_stelizabeths_center", 5, 1); capitol_seed()
    if "wfu" in what: pad_wfu()
