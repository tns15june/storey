import fitz
D = "data/plans/nibs_clinic/2012-03-27-ClinicRedactedDrawings/"
p = fitz.open(D + "Clinic_116_Exterior Window Elevation Schedule And Details.pdf")[0]
pix = p.get_pixmap(dpi=170, clip=fitz.Rect(180, 120, 800, 560)); pix.save("data/lab/audit2/w1_type.png")
