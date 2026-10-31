import json, sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/modern"); sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/agent")
import cv2, numpy as np, draw, vec
O = "/Users/tsharma/Github/Meta/data/lab/agent/mine/"
d = json.load(open(O + "doors.json"))
for r in d:
    pg = vec.open_page(r["id"], r["file"], r["page"])
    px = [p for p in r["pairs"] if p["d"] < 75]
    if not px: continue
    ims = []
    for p in px:
        t, a = p["tag"], p["arc"]
        cx, cy = (t[0] + a[0]) / 2, (t[1] + a[1]) / 2; R = 110
        im, P, s = draw.render(pg, (cx - R, cy - R, cx + R, cy + R), long=640, max_dpi=240)
        cv2.circle(im, P(*t), 16, (255, 0, 0), 2); cv2.circle(im, P(*a), 8, (0, 160, 0), 2); cv2.line(im, P(*t), P(*a), (0, 0, 255), 1)
        cv2.putText(im, f"tag {p['mark']} sch{p['sched']} arc{p['r_in']} d{p['d']}", (4, 16), 0, .5, (0, 0, 255), 1)
        im = cv2.copyMakeBorder(im, 0, max(0, 640 - im.shape[0]), 0, max(0, 640 - im.shape[1]), cv2.BORDER_CONSTANT, value=(255, 255, 255))[:640, :640]
        ims.append(im)
    while len(ims) % 3: ims.append(np.full((640, 640, 3), 255, np.uint8))
    rows = [np.hstack(ims[i:i + 3]) for i in range(0, len(ims), 3)]
    cv2.imwrite(O + f"doorcrops_{r['id']}_{r['level']}.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
