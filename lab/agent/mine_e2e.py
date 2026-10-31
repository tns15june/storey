"""Views of the clinic stairs and life safety sheets for building the end-to-end fixture. No API."""
import sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/agent")
import cv2, numpy as np, pymupdf as fitz, draw
D = "/Users/tsharma/Github/Meta/data/plans/nibs_clinic/2012-03-27-ClinicRedactedDrawings/"
O = "/Users/tsharma/Github/Meta/data/lab/agent/mine/"
GT = {"1CS3": {"1": [1196, 286, 1249, 399], "2": [1194, 280, 1250, 400]}, "1AS1": {"1": [1581, 548, 1631, 644], "2": [1579, 547, 1634, 645]}, "1BS2": {"1": [1604, 1104, 1707, 1159], "2": [1582, 1103, 1702, 1153]}}
S = {"1": "Clinic_052_Composite First Floor Plan.pdf", "2": "Clinic_059_Composite Second Floor Plan.pdf"}
ims = []
for k, v in GT.items():
    for fl, b in v.items():
        pg = fitz.open(D + S[fl])[0]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; R = 130
        im, P, s = draw.render(pg, (cx - R, cy - R, cx + R, cy + R), long=640)
        cv2.rectangle(im, P(b[0], b[1]), P(b[2], b[3]), (0, 200, 0), 1)
        cv2.putText(im, f"{k} F{fl}", (4, 16), 0, .55, (0, 0, 255), 1)
        ims.append(cv2.copyMakeBorder(im, 0, 640 - im.shape[0], 0, 640 - im.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)))
cv2.imwrite(O + "e2e_stairs.jpg", np.vstack([np.hstack(ims[i:i + 3]) for i in (0, 3)][:2]) if False else np.vstack([np.hstack(ims[:2] + ims[2:3]), np.hstack(ims[3:6])]), [cv2.IMWRITE_JPEG_QUALITY, 85])
pg = fitz.open(D + "Clinic_161_Life Safety Composite First Floor Plan.pdf")[0]
print(pg.rect)
im, P, s = draw.render(pg, (0, 0, pg.rect.width, pg.rect.height), long=2000)
cv2.imwrite(O + "e2e_ls161.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
