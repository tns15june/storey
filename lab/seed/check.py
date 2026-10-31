"""Self-check for the geometry helpers. No API calls, no data needed beyond the catalog and truth files."""
import sys; sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/seed")
from common import iou, hit, to_px, TRUTH
assert to_px({"x_min": 100, "y_min": 200, "x_max": 300, "y_max": 400}, 2000, 1000) == [200, 200, 600, 400]  # x then y, the bug that gave 0/5 on the first score
assert to_px({"x_min": 300, "y_min": 1001, "x_max": 100, "y_max": 200}, 2000, 1000) == [200, 200, 600, 999]  # swapped and out of range are clamped
assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1 and iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0
assert hit([0, 0, 10, 10], [0, 0, 10, 10]) and not hit([50, 50, 60, 60], [0, 0, 10, 10]) and not hit([-50, -50, 60, 60], [0, 0, 10, 10])
assert all(0 <= f for b, v in TRUTH.items() if not b.startswith("_") for t in v for f in t["floors"])
print("ok")
