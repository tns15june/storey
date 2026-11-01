"""Contact sheet of sampled flag crops with their flag text. No API."""
import json, cv2, numpy as np
D = "data/lab/declared/crops/"
ids = json.load(open("data/lab/audit2/declared_pick.json"))
tiles = []
for i in ids:
    im = cv2.imread(D + i + ".png")
    if im is None:
        print("no crop", i); im = np.full((80, 400, 3), 255, np.uint8)
    print(i, im.shape)
    z = min(760 / im.shape[1], 420 / im.shape[0])
    im = cv2.resize(im, None, fx=z, fy=z, interpolation=cv2.INTER_AREA if z < 1 else cv2.INTER_CUBIC)
    c = np.full((450, 770, 3), 255, np.uint8)
    c[24:24 + im.shape[0], 4:4 + im.shape[1]] = im
    cv2.putText(c, i, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 200), 1, cv2.LINE_AA)
    tiles.append(cv2.copyMakeBorder(c, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
for k in range(2):
    part = tiles[k * 6:(k + 1) * 6]
    cv2.imwrite(f"data/lab/audit2/declared_crops{k}.jpg", np.vstack([np.hstack(part[j:j + 2]) for j in range(0, 6, 2)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
