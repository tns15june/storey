"""Render a crop of a sheet to the scratchpad for reading by eye. No model.
Usage: view.py <pdf> <page> <x0> <y0> <x1> <y1> <out.png> [px_long]  (coordinates are 0-1000 of the rendered page)"""
import sys, warnings
warnings.filterwarnings("ignore")
import fitz
f, p, x0, y0, x1, y1, out = sys.argv[1], int(sys.argv[2]), *map(float, sys.argv[3:7]), sys.argv[7]
px = int(sys.argv[8]) if len(sys.argv) > 8 else 2000
pg = fitz.open(f)[p]
W, H = pg.rect.width, pg.rect.height
clip = fitz.Rect(x0 / 1000 * W, y0 / 1000 * H, x1 / 1000 * W, y1 / 1000 * H)
z = px / max(clip.width, clip.height)
pg.get_pixmap(matrix=fitz.Matrix(z, z), clip=clip, alpha=False).save(out)
