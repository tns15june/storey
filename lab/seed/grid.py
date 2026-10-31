"""Draw a 100 px coordinate grid on aligned floors, for reading truth boxes by eye. Usage: grid.py building [x1 y1 x2 y2]"""
import sys; sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/seed")
from common import *
bid = sys.argv[1]
d = OUT / bid; (d / "grid").mkdir(exist_ok=True)
for i in range(len(CAT[bid]["floors"])):
    im = cv2.imread(str(d / f"f{i}.png"))
    for x in range(0, im.shape[1], 100):
        cv2.line(im, (x, 0), (x, im.shape[0]), (0, 170, 0) if x % 500 else (0, 0, 255), 1)
        for y0 in range(10, im.shape[0], 500):
            cv2.putText(im, str(x), (x + 2, y0 + 12), 0, 0.4, (0, 0, 255), 1)
    for y in range(0, im.shape[0], 100):
        cv2.line(im, (0, y), (im.shape[1], y), (0, 170, 0) if y % 500 else (0, 0, 255), 1)
        for x0 in range(10, im.shape[1], 500):
            cv2.putText(im, str(y), (x0 + 2, y - 2), 0, 0.4, (255, 0, 0), 1)
    cv2.imwrite(str(d / "grid" / f"f{i}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
