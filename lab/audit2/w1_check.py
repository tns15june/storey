"""What width does the set give window W1? Text layer and render of the window schedule sheets. No API."""
import fitz, re, glob
D = "/Users/tsharma/Github/Meta/data/plans/nibs_clinic/2012-03-27-ClinicRedactedDrawings/"
for f in ("Clinic_116_Exterior Window Elevation Schedule And Details.pdf", "Clinic_117_Exterior Window Elevation Schedule And Details.pdf"):
    d = fitz.open(D + f); p = d[0]; t = p.get_text()
    print(f, p.rect, "chars", len(t), "W1 hits", len(re.findall(r"\bW-?1\b", t)))
    for w in p.get_text("words"):
        if re.fullmatch(r"W-?1", w[4]):
            print("   W1 at", [round(x) for x in w[:4]])
    pix = p.get_pixmap(dpi=60); pix.save("/Users/tsharma/Github/Meta/data/lab/audit2/w_" + f[7:10] + ".png")
# every R.O. dimension near a W1 tag on the block plans
for f in sorted(glob.glob(D + "Clinic_05[3-8]_*.pdf")):
    d = fitz.open(f); p = d[0]; ws = p.get_text("words")
    tags = [w for w in ws if w[4] == "W1"]
    ros = [w for w in ws if w[4] in ("R.O.", "R.O")]
    near = []
    for r in ros:
        cand = [w for w in ws if abs((w[0] + w[2]) / 2 - (r[0] + r[2]) / 2) < 40 and abs(w[1] - r[1]) < 22 and re.fullmatch(r"\d{3,4}", w[4])]
        near.append([c[4] for c in cand])
    print(f.split("/")[-1][:34], "W1 tags", len(tags), "R.O. labels", len(ros), "printed near R.O.", near[:14])
