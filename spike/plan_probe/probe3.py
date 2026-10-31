"""Throwaway: repeat the erase test with the whole of staircase II removed from level 1, on the same aligned stack."""
import subprocess, sys
import cv2, numpy as np
sys.argv = ["probe.py", "none"]
from probe import stack, track

def frame(n):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", "floors.mp4", "-vf", f"select=eq(n\\,{n})", "-fps_mode", "passthrough", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

frames = [frame(i * 4 + 3) for i in range(4)]
names = ["level 2", "level 1", "level 0", "level -1"]
x1, y1, x2, y2 = 1160, 672, 1296, 790
frames[1][y1:y2, x1:x2] = np.median(frames[1].reshape(-1, 3), axis=0)
cv2.imwrite("erased_full_check.jpg", frames[1][600:860, 1080:1380])
w, h = stack(frames, "floors_erased2.mp4")
res, _ = track("floors_erased2.mp4", {1: (848, 900, 945, 1015), 2: (1175, 660, 1275, 740)}, w, h, names)
for k in sorted(res):
    print(k, res[k])
