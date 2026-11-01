"""CubiCasa5K reader: image plus ground truth masks, straight from the zip (data only, nothing executed)."""
import re
import threading
import zipfile
import xml.etree.ElementTree as ET
import cv2
import numpy as np

PATH = 'data/plans/cubicasa5k/cubicasa5k.zip'
Z = zipfile.ZipFile(PATH)
LOCK = threading.Lock()


def read(n):
    with LOCK:  # ZipFile reads are not safe across threads
        return Z.read(n)
NS = '{http://www.w3.org/2000/svg}'
OUT = ('Outdoor', 'Garage')  # treated as not interior when looking for exits


def split(name='test'):
    return ['cubicasa5k' + x for x in Z.read(f'cubicasa5k/{name}.txt').decode().split()]


def _poly(e):
    p = e.find(NS + 'polygon')
    if p is None or not p.get('points'):
        return None
    v = np.array([float(x) for x in re.split(r'[ ,]+', p.get('points').strip())], float)
    return v[:len(v) // 2 * 2].reshape(-1, 2)


def load(house):
    """Returns dict: img, wall (polygons incl. windows, doors cut), door_polys, rooms [(name, poly)], stairs."""
    img = cv2.imdecode(np.frombuffer(read(house + 'F1_scaled.png'), np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    root = ET.fromstring(read(house + 'model.svg'))
    walls, doors, windows, rooms, railings = [], [], [], [], []

    def walk(e):
        c = (e.get('class') or '').split(' ')
        if c[0] == 'Wall':
            walls.append(_poly(e))
        elif c[0] == 'Door':
            doors.append(_poly(e))
        elif c[0] == 'Window':
            windows.append(_poly(e))
        elif c[0] == 'Railing':
            railings.append(_poly(e))
        elif c[0] == 'Space':
            rooms.append((' '.join(c[1:]), _poly(e)))
        for x in e:
            walk(x)
    walk(root)

    def fill(ps, m=None):
        m = np.zeros((h, w), np.uint8) if m is None else m
        for p in ps:
            if p is not None and len(p) >= 3:
                cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
        return m
    door_m = fill(doors)
    wall_m = fill(windows, fill(walls))
    wall_m[door_m > 0] = 0
    return dict(img=img, wall=wall_m > 0, door=door_m > 0, doors=[d for d in doors if d is not None],
                windows=[d for d in windows if d is not None], rooms=[(n, p) for n, p in rooms if p is not None], fill=fill, h=h, w=w)
