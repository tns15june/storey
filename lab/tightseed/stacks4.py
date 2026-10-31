"""New stacks (no model has seen them) and dev stacks. Plan box comes from the PDF's image rect or a fixed panel, never from a model."""
import warnings; warnings.filterwarnings("ignore")
import json, sys
import fitz, cv2, numpy as np
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed"); sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/align")
import reg, stacks as st
P = st.ROOT / "data/plans"


def page(pg, z, clip=None):
    px = pg.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False, clip=clip)
    return cv2.cvtColor(np.frombuffer(px.samples, np.uint8).reshape(px.h, px.w, 3), cv2.COLOR_RGB2BGR).copy()


def plan_rect(pg):
    """Largest embedded image rect, in page points."""
    return max((r for x in pg.get_images(full=True) for r in pg.get_image_rects(x[0])), key=lambda r: r.width * r.height)


def queens(name, fn, pages, ref):
    d = fitz.open(P / fn)
    z = 2000 / 612
    rects = [plan_rect(d[i]) for i in pages]
    ims = [page(d[i], z) for i in pages]
    boxes = [tuple(int(v * z) for v in (r.x0, r.y0, r.x1, r.y1)) for r in rects]
    print(name, boxes)
    Hs, info = st.align_stack(ims, ref, boxes)
    st.save(name, ims, Hs, ref, info)


if __name__ == "__main__":
    what = sys.argv[1:]
    if "queens_leonard" in what:
        queens("queens_leonard", "queens_leonard/leonard_hall.pdf", [1, 2, 3, 4, 5], 2)
    if "queens_royce1" in what:
        queens("queens_royce1", "queens_royce/jean_royce_hall.pdf", [1, 2, 3, 4], 1)
    if "queens_royce2" in what:
        queens("queens_royce2", "queens_royce/jean_royce_hall.pdf", [5, 6, 7, 8], 2)
