"""Write data/plans/catalog.json and check that every floor entry points at a real file and page.

Run: .venv/bin/python -I lab/gather_data/build_catalog.py
page is 0-based. public_demo is yes, ask or no, and is my reading of the license text, not legal advice.
"""
import json
from pathlib import Path

import cv2
import fitz

PLANS = Path(__file__).resolve().parent.parent.parent / "data/plans"
CLINIC = "2012-03-27-ClinicRedactedDrawings/Clinic_"
WBDG_BIM = "https://legacy.wbdg.org/bim/cobie/common-bim-files"
MML = "https://mml.org/resources-research/books-and-briefs/pattern-book-homes-for-21st-century-michigan/volume-ii-the-missing-middle-mixtape/"
MML_LIC = 'Publisher page calls these "copyright-free construction documents". No named license. Sheets are marked NOT FOR CONSTRUCTION.'
HABS = "Commons tag: Public domain. Historic American Buildings Survey, US federal record."


def COMMONS_PAGE(bid):
    """First sheet's Commons page. commons_meta.json in the folder has the page and license of every sheet."""
    return json.loads((PLANS / bid / "commons_meta.json").read_text())[0]["page_url"]


def fl(level, file, page=0):
    return {"level": level, "file": file, "page": page}


def entry(id, name, url, lic, demo, kind, btype, floors, stairs, code, notes, stackable=True, truth=None):
    return {"id": id, "name": name, "source_url": url, "license": lic, "public_demo": demo, "kind": kind, "building_type": btype,
            "floors": floors, "stackable": stackable, "stairs_note": stairs, "has_code_summary_sheet": code, "ground_truth": truth, "notes": notes}


CATALOG = [
    entry("nibs_clinic", "Medical and dental clinic (Common BIM File, location redacted)", WBDG_BIM,
          "No license statement for the files. WBDG site footer reads (c) 2026 National Institute of Building Sciences, All rights reserved. Page offers the files to extend for research. Drawings are redacted.", "ask", "vector",
          "medical clinic, 2 floors",
          [fl(1, CLINIC + "052_Composite First Floor Plan.pdf"), fl(2, CLINIC + "059_Composite Second Floor Plan.pdf")],
          "3 stairs. Enlarged plans on sheets 122 to 124 (A11-1 to A11-3). IFC has 3 IfcStair.", True,
          "Best set we have. 316 single-sheet PDFs. 42x28 in, 1:150, metric dimensions. Sheets 161 and 162 are life safety plans drawn by the architect. Sheet 161: travel distance 137 FT / 41.75 M, exit widths and capacities (915 mm, 180 people; 1830 mm, 360 people), Business occupancy, Type II(N) under the UBC, fully sprinklered. Sheet 162: travel distance 153 FT / 46.65 M, two boxes reading STAIR WELL EXIT CAPACITY = 147 PEOPLE. The third stair has no such box and appears to be the open glass stair, so a tool that counts it as an exit would be wrong. No text layer: 0 characters on sheet 52, so labels must be read from the image. Block plans 053-058 and 060-063 are larger scale.",
          truth="Clinic_A_20110906_optimized.ifc is the same building: room ids 1AS1, 2AS1, 1AC2, 2AC2 on sheet 122 are IfcSpace names. 269 IfcSpace, 254 IfcDoor, 3 IfcStair, 6 IfcStairFlight."),
    entry("wfu_efird", "Efird Residence Hall, Wake Forest University", "https://prod.wp.cdn.aws.wfu.edu/sites/303/2017/07/Efird.pdf",
          "No license stated. University copyright assumed.", "no", "vector", "university residence hall, 4 levels",
          [fl(0, "Efird.pdf", 0), fl(1, "Efird.pdf", 1), fl(2, "Efird.pdf", 2), fl(3, "Efird.pdf", 3)],
          "One central stair beside the lobby on the first floor sheet. Other floors not traced.", False,
          "11x17 in, vector, colored by room type, no dimensions and no scale, so distances cannot be measured from it. Same footprint as Huffman, mirrored, so the two are not independent."),
    entry("wfu_huffman", "Huffman Residence Hall, Wake Forest University", "https://prod.wp.cdn.aws.wfu.edu/sites/303/2017/07/HUFFMAN-FLOOR-PLAN-G-3.pdf",
          "No license stated. University copyright assumed.", "no", "vector", "university residence hall, 4 levels",
          [fl(0, "HUFFMAN-FLOOR-PLAN-G-3.pdf", 0), fl(1, "HUFFMAN-FLOOR-PLAN-G-3.pdf", 1), fl(2, "HUFFMAN-FLOOR-PLAN-G-3.pdf", 2), fl(3, "HUFFMAN-FLOOR-PLAN-G-3.pdf", 3)],
          "One central stair beside the lobby on the first floor sheet. Other floors not traced.", False,
          "11x17 in, vector, thin walls, no dimensions and no scale. Mirrored twin of Efird."),
    entry("mml_capitol", "The Capitol, Michigan pattern book duplex", MML, MML_LIC, "yes", "vector", "small residential, 3 levels",
          [fl("ground", "THE-CAPITOL.pdf", 2), fl("main", "THE-CAPITOL.pdf", 3), fl("upper", "THE-CAPITOL.pdf", 4)],
          "Interior stair in a stair hall, plus exterior stairs on the upper plan.", False,
          "36x24 in, 1/4 in = 1 ft, dimensioned, door and window schedules, real text layer. BUILDING DATA block is left blank for local completion. Best candidate for a clean public vector demo."),
    entry("mml_baxter", "The Baxter, Michigan pattern book duplex", MML, MML_LIC, "yes", "vector", "small residential, 2 levels",
          [fl("main", "THE-BAXTER.pdf", 2), fl("upper", "THE-BAXTER.pdf", 3)],
          "Two stair halls side by side, one per unit.", False, "36x24 in, 1/4 in = 1 ft, dimensioned, schedules, text layer."),
    entry("mml_abbot", "The Abbot, Michigan pattern book ADU", MML, MML_LIC, "yes", "vector", "small residential, 2 levels, ADU over garage",
          [fl("main", "THE-ABBOT.pdf", 1), fl("upper", "THE-ABBOT.pdf", 1)],
          "Exterior stair only, 16 risers noted on sheet A2.0.", False,
          "Both levels sit on one sheet (A2.0), so the sheet must be cropped into panels before stacking. Module B plans start at page 6. Volume2_manual.pdf is the pattern book."),
    entry("habs_stelizabeths_center", "St. Elizabeths Hospital, Center Building, Washington DC",
          COMMONS_PAGE("habs_stelizabeths_center"), HABS + " HABS DC-349-W.", "yes", "scan", "hospital, basement plus 4 floors",
          [fl(-1, "00.jpg"), fl(1, "02.jpg"), fl(2, "03.jpg"), fl(3, "04.jpg"), fl(4, "01.jpg")],
          "Several stairs, including curved stairs at the ward wing ends. Not counted.", False,
          "Raster of clean CAD-style measured drawings, printed scale bars and dimensions, same sheet frame on every floor. Saved at 5000 px wide from 17600 px originals; re-fetch for fine text. Commons calls 01.jpg 'Fifth Floor Plan' but the sheet says FOURTH FLOOR PLAN with a fifth floor tower inset, so level 4 is right. Top public demo pick for the stacking result."),
    entry("habs_stelizabeths_atkins", "St. Elizabeths Hospital, Atkins Hall, Washington DC",
          COMMONS_PAGE("habs_stelizabeths_atkins"), HABS + " HABS DC-349-T.", "yes", "scan", "hospital ward building, basement, 3 floors, attic",
          [fl(-1, "01.jpg"), fl(1, "02.jpg"), fl(2, "03.jpg"), fl(3, "04.jpg"), fl("attic", "00.jpg")],
          "One main stair running basement to attic, plus exterior stairs.", False,
          "Same drawing style as Center Building, small and simple. Good first test. Saved at 5000 px wide from 14400 px."),
    entry("habs_va_bldg1", "VA Medical Center Building No. 1, Huntington WV",
          "https://commons.wikimedia.org/wiki/File:Photocopy_of_original_drawing,_May_20,_1931._First_floor_plan_as_built._-_Veterans_Administration_Medical_Center,_Building_No._1,_Spring_Valley_Drive,_Huntington,_Cabell_County,_HABS_WVA,50-HUNT,1A-28.tif",
          HABS + " Originals by the US Veterans Bureau, 1931. HABS WVA,50-HUNT,1A.", "yes", "scan", "hospital, ground plus 4 floors",
          [fl(0, "02.jpg"), fl(1, "00.jpg"), fl(2, "03.jpg"), fl(3, "04.jpg"), fl(4, "01.jpg")],
          "Not traced.", False,
          "Worst case on purpose: photographs of 1931 as-built sheets, film edges, tape, skew, different placement on every sheet. Stress test for loose alignment."),
    entry("bme_k", "BME Building K, Budapest", "https://commons.wikimedia.org/wiki/File:BME_Building_K,_level_0_floor_plan.jpg",
          "PD-old-auto-expired, deathyear 1951. Author György Klösz, Pál Klösz, dated 1909.", "yes", "scan", "university, 4 levels",
          [fl(0, "00.jpg"), fl(0.5, "01.jpg"), fl(1, "02.jpg"), fl(2, "03.jpg")],
          "Many stairs. Not traced.", False, "Large sheets, 11300 px wide. Aged paper tint. Hungarian labels."),
    entry("bme_f", "BME Building F, Budapest", "https://commons.wikimedia.org/wiki/File:BME_Building_F,_level_0_floor_plan.jpg",
          "PD-anon-expired. Commons gives date 1905, author unknown.", "ask", "scan", "university, 4 levels",
          [fl(-1, "bme_-1.jpg"), fl(0, "bme_0.jpg"), fl(1, "bme_1.jpg"), fl(2, "bme_2.jpg")],
          "5 staircases on the ground floor by the earlier reading. The earlier test found 3 floor and staircase pairs where a staircase really ends.", False,
          "Used in the stacking test. The 1905 date is wrong for these sheets: the level 0 legend lists ATOMFIZIKA TSZ. (atomic physics department), which cannot be 1905, so 1905 is probably the building date and the PD-anon-expired tag may not hold. Do not lead the public demo with this set until the archive (HU BMEL XV.602.a) confirms the date."),
    entry("schenley", "Schenley High School, Pittsburgh", "https://commons.wikimedia.org/wiki/File:Schenley_High_School,_1916,_First_Floor_Plan.png",
          "PD-US. Published 1916, Edward Stotz architect.", "yes", "scan", "high school, 4 levels",
          [fl(-1, "sch_base.png"), fl(0, "sch_ground.png"), fl(1, "sch_first.png"), fl(2, "sch_second.png")],
          "Stairs at the three corners and by the entrance on the first floor sheet.", False,
          "Book plates about 2100 px wide. First floor file is byte-identical to the Commons file (sha1 43862b1d). Triangular plan, wide corridors, a good contrast to the others."),
    entry("queens_leonard", "Leonard Hall, Queen's University", "https://www.queensu.ca/residences/sites/residencewww/files/2026-06/Leonard%20Hall%20Floor%20Plans%2001.2026.pdf",
          "No license stated. University copyright assumed.", "no", "scan", "university residence hall, 5 levels",
          [fl(0, "leonard_hall.pdf", 1), fl(1, "leonard_hall.pdf", 2), fl(2, "leonard_hall.pdf", 3), fl(3, "leonard_hall.pdf", 4), fl(4, "leonard_hall.pdf", 5)],
          "Stair symbols visible but small.", False,
          "Letter-size housing map. The plan is a low-resolution embedded image, 1016x535 px on the first floor, under colored fills and arrows. Canadian. Weak for measuring."),
    entry("queens_royce", "Jean Royce Hall, Queen's University", "https://www.queensu.ca/residences/sites/residencewww/files/2025-06/Jean%20Royce%20Hall%20Floor%20Plans%2005.2025.pdf",
          "No license stated. University copyright assumed.", "no", "scan", "university residence hall, 4 levels, 2 phases",
          [fl(f"phase1_{n}", "jean_royce_hall.pdf", n) for n in (1, 2, 3, 4)] + [fl(f"phase2_{n}", "jean_royce_hall.pdf", n + 4) for n in (1, 2, 3, 4)],
          "Many small stairs at wing ends.", False, "Embedded images up to 1967 px wide. Each house wing is small on the page."),
    entry("carpinteria_adu1", "Carpinteria ADU Plan 1, 400 sf", "https://carpinteriaca.gov/city-hall/community-development/planning/accessory-dwelling-unit-program/preapproved-adus/",
          'City page: "For public use, provided by the City of Carpinteria". Title block: "only to be used within City of Carpinteria jurisdiction". Designer RRM Design Group.', "ask", "vector", "small residential, 1 floor",
          [fl(1, "plan1.pdf", 7)], "None, single story.", True,
          "Complete permit set, 34 pages, 36x24 in, Revit export with text. G-001 has occupancy R-3, Type VB, sprinkler notes under the 2022 CBC. Real set structure, but nothing to stack.", stackable=False),
    entry("carpinteria_adu2", "Carpinteria ADU Plan 2, 480 sf", "https://carpinteriaca.gov/city-hall/community-development/planning/accessory-dwelling-unit-program/preapproved-adus/",
          'City page: "For public use, provided by the City of Carpinteria". Title block: "only to be used within City of Carpinteria jurisdiction". Designer RRM Design Group.', "ask", "vector", "small residential, 1 floor",
          [fl(1, "plan2.pdf", 7)], "None, single story.", True, "33 pages. Same structure as Plan 1.", stackable=False),
    entry("usc_leconte", "LeConte College 3rd floor renovation, University of South Carolina", "https://sc.edu/purchasing/solicitations/documents/USC%20LeConte%20Bid%20Drawings.pdf",
          "(c) 2014 Compass 5 Partners, LLC, printed on the sheets. Public bid document. Sheets carry architect seals.", "no", "vector", "university, one floor of a larger building",
          [fl(3, "USC_LeConte_Bid_Drawings.pdf", 2)], "STAIR ST01 on the life safety plan, outside the work area.", True,
          "19 sheets, 42x30 in. G002 CODE SHEET, G003 LIFE SAFETY PLAN at 1/4 in = 1 ft with remote travel distance 27'-8\" drawn and labeled and exit locations marked. Good model of what a reviewer looks at. One floor only, and only part of it. Do not show publicly.", stackable=False),
    entry("af_ces_admin", "USAF Civil Engineer Squadron Administration Facility", "https://legacy.wbdg.org/node/22571",
          "No license statement for the files, and the page has no access limit. WBDG site footer reads (c) 2026 NIBS. PDF metadata author is Air Force - AFCEC. Probably a US government work; contractor role unknown.", "yes", "scan", "office, 1 floor",
          [fl(1, "CES_SD_drawings_admin_bldg.pdf", 1)], "None, single story.", False,
          "Sheets are 24 raster tiles with a real text layer and a few overlay lines, so walls are not vector. 6 sheets, marked NOT FOR CONSTRUCTION, approximate dimensions, no wall thickness or structure by its own note. standard_design_2018.pdf is the 68-page narrative. Nothing to stack.", stackable=False),
    entry("af_ces_shops", "USAF Civil Engineer Squadron Shops and Warehouse", "https://legacy.wbdg.org/node/22576",
          "No license statement for the files, and the page has no access limit. WBDG site footer reads (c) 2026 NIBS. Probably a US government work; contractor role unknown.", "yes", "scan", "industrial, 1 floor",
          [fl(1, "CES_sd_drawings_warehouse_bldg.pdf", 1)], "None, single story.", False, "Same raster-tile format as the admin set, at half the tile resolution. Nothing to stack.", stackable=False),
    entry("brandeis_hall", "New Residence Hall, Brandeis University, level 1 only", "https://www.brandeis.edu/new-residence-hall/pdfs/2017-03-17-floor-plan-level-1.pdf",
          "No license stated. Copyright assumed to rest with William Rawn Associates and Brandeis.", "no", "vector", "university residence hall, other levels not found",
          [fl(1, "level-1.pdf")], "STAIR 1, 2, 3 labeled.", False,
          "40x30 in, 1/8 in = 1 ft, clean presentation plan with vector rooms and labels. Other levels were not obtained: the guessed level-2 to level-5 URLs gave a copy of level 1 once and then 403.", stackable=False),
    entry("nibs_office", "Office model (Common BIM File)", WBDG_BIM, "No license statement on the WBDG page.", "ask", "ifc", "office, 2 levels",
          [], "2 IfcStair, 4 IfcStairFlight.", False,
          "IFC only. The downloaded PDF is a 3D PDF with one flat preview page, not a drawing. Could be cut into synthetic plans with exact truth; not done.", stackable=False, truth="IFC model: 99 IfcSpace, 102 IfcDoor."),
    entry("nibs_duplex", "Duplex apartment model (Common BIM File)", WBDG_BIM, "No license statement on the WBDG page. Created by a student for a design competition, per the page.", "ask", "ifc", "small residential, 2 levels",
          [], "2 IfcStair, 2 IfcStairFlight.", False, "IFC only, same as the office model.", stackable=False, truth="IFC model: 21 IfcSpace, 14 IfcDoor."),
    entry("cubicasa5k", "CubiCasa5K", "https://zenodo.org/records/2613548", "CC BY-NC-SA 4.0 per the Zenodo record, not plain CC BY-NC.", "no", "dataset", "residential floor plans, annotated",
          [], "Not applicable.", False, "Complete, md5 matches Zenodo. Kept as the zip: 5000 SVG, 12342 PNG, 6.0 GB unpacked. Read it from the zip, because unpacking alongside it would pass the 12 GB cap. NC plus ShareAlike: evaluation only, no training for the product, no images in the demo.", stackable=False, truth="Annotated walls, doors, windows, rooms as SVG."),
    entry("swiss_dwellings", "Swiss Dwellings v3.0.0", "https://zenodo.org/records/7788422", "CC BY 4.0 per the Zenodo record.", "yes", "dataset", "residential buildings, geometry",
          [], "Stairs, shafts, elevators are labeled per floor. See swiss_stats.json.", False,
          "Complete, md5 matches Zenodo. Kept as the zip. Geometry tables, not drawings: 3184 buildings, 2872 with 3 or more floors, 2624 of those with stair records on every floor. MSD is derived from it and was not fetched.", stackable=False, truth="Exact polygons for walls, doors, rooms, stairs, shafts, elevators, by floor."),
    entry("floorplancad", "FloorPlanCAD, test split", "https://floorplancad.github.io/", 'Annotations CC BY-NC 4.0. Site says: "We do not own the copyright of the drawings."', "no", "dataset", "CAD plans with symbol labels",
          [], "Not applicable.", False, "test-00.tar.xz only, 5502 SVG and 11004 PNG by archive listing. train-00 (1.46 GB) and train-01 (2.13 GB) not fetched. Not unpacked.", stackable=False, truth="Symbol-level labels on vector primitives."),
]


def check(e):
    for f in e["floors"]:
        path = PLANS / e["id"] / f["file"]
        assert path.exists(), path
        if path.suffix == ".pdf":
            assert f["page"] < len(fitz.open(path)), path
        else:
            assert cv2.imread(str(path)) is not None, path


for e in CATALOG:
    check(e)
(PLANS / "catalog.json").write_text(json.dumps(CATALOG, indent=1, ensure_ascii=False))
stack = [e for e in CATALOG if e["stackable"]]
print(len(CATALOG), "entries,", len(stack), "stackable:",
      sum(e["kind"] == "vector" for e in stack), "vector,", sum(e["kind"] == "scan" for e in stack), "scan,",
      sum(e["public_demo"] == "yes" for e in stack), "public demo yes")
