"""Hand-read truth, written to data/lab/declared/truth.json before any model call on these sheets.

Sources: the three known defects from Round 1, my reading of rendered crops, the text layer, and earlier hand truth
in lab/bakeoff/e2_tasks.py (DOORS_N79, SPECIAL_N79, EXITS, CODE), copied here so this file stands alone.
"""
import hashlib, json, time
from pathlib import Path

OUT = Path("/Users/tsharma/Github/Meta/data/lab/declared/truth.json")

D079 = """1A03 1A04 1A05 1A06 1A07 1A08 1A09 1A10 1A11 1A12 1A13 1A15 1A15A 1AC1 1AC2 1AC3 1AC4 1AC5 1AC5A 1B01 1B02 1B03 1B04 1B04A 1B05 1B06 1B07 1B08 1B09
1B10 1B11 1B12 1B13 1B14 1B15 1B16 1B17 1B19 1B19A 1B21 1B23 1BC1 1BC2 1BC3 1BC4 1BC4A 1BS2 1BS2A 1C02 1C03 1C04 1C05 1C06 1C07 1C08 1C09 1C10 1C11 1C12 1C13 1C14 1C15 1C16
1C17 1C18 1C18A 1C19 1CC1 1CC3 1CS3 1D01 1D02 1D04 1D05 1D06 1D06A 1D07 1D08 1D09 1D10 1D11 1D12 1D13 1D14 1D15 1D16 1D17 1D18 1D19 1D20 1D21""".split()
D080 = """1D22 1D22A 1D23 1D24 1D25 1D26 1D27 1D28 1D30 1D31 1D32 1D33 1D33A 1D34 1D35 1D36 1D38 1D39 1D40 1D40A 1DC2 1DC4 1DC5 1DC5A 1DC5B 1DC8
1E01 1E02 1E03 1E04 1E05 1E06 1E07 1E07A 1E08 1E09 1E10 1E11 1E12 1E14 1E15 1E15A 1E16 1E17 1E18 1E19 1E20 1E21 1E22 1E23 1E24 1E24A 1E25 1E26 1E27 1E28 1E29 1E30 1E33
1EC1 1EC1A 1EC3 1EC4 2A01 2A02 2A03 2A04 2A05 2A06 2A07 2A08 2A09 2A11 2A11A 2A12 2A13 2A14 2AC2 2AC2A 2AC5 2B01 2B02 2B03 2B04 2B05 2B06 2B07 2B08 2B09 2B10 2B12""".split()
D081 = """2B13 2B14 2B15 2B16 2B17 2B18 2B19 2B20 2B21 2B22 2B23 2B24 2B25 2BC1 2BC2 2BC3 2BC4 2BC6 2BS2 2C01 2C02 2C03 2C04 2C05 2C06 2C07 2C08 2C09 2C10 2C11 2C12 2C12A 2C13 2C14 2C15
2C16 2C17 2C18 2C19 2C22 2C23 2C24 2CC1 2CC3 2CC3A 2CS3 2D01 2D02 2D03 2D04 2D04A 2D05 2D05A 2D06 2D06A 2D07 2D08 2D09 2D10 2D11 2D13 2D14 2D15 2D16 2D17 2D19 2D20 2D20A 2D20B
2D21 2D21A 2D21B 2DC2""".split()

TRUTH = {
    "known_defects": [
        {"id": "K1", "set": "mml_baxter", "sheet": "A2.1", "check": 5, "text": "door tag 13 sits on a 36 in swing door, schedule row 13 says 60 in bifold"},
        {"id": "K2", "set": "mml_capitol", "sheet": "A2.1", "check": 6, "text": "title block MAIN FLOOR PLAN, drawing label Ground Floor Plan"},
        {"id": "K3", "set": "bcoe_addendum3", "sheet": "A012", "check": 4, "text": "exit table Exit 1 90 occupants 18 in, worked example 100 occupants 20.0 in"},
    ],
    "usc_leconte": {
        "index": {"G001": "COVER SHEET & INDEX", "G002": "CODE SHEET, GENERAL NOTES, & LEGEND", "G003": "LIFE SAFETY PLAN", "AD101": "DEMOLITION FLOOR PLAN",
                  "A101": "RENOVATION FLOOR PLAN", "A121": "REFLECTED CEILING PLAN", "A300": "INTERIOR ELEVATIONS AND DETAIL", "A400": "DOOR SCHEDULE & DETAILS",
                  "A500": "CEILING DETAILS", "A700": "FINISH PLAN, LEGEND & SCHEDULE", "A721": "FURNITURE PLAN (FOR REFERENCE ONLY)", "P100": "FLOOR PLAN, NOTES AND SCHEDULE",
                  "MD100": "DEMOLITION FLOOR PLAN", "M100": "RENOVATION FLOOR PLAN", "M200": "DETAILS, NOTES, SCHEDULES, AND LEGENDS", "E001": "ELECTRICAL SYMBOLS, LEGEND, AND SCHEDULES",
                  "ED101": "ELECTRICAL DEMOLITION PLANS", "E101": "LIGHTING RENOVATION PLAN", "E201": "POWER/SYSTEMS RENOVATION PLAN"},
        "title_blocks_pages_11_18": {"11": ["P100", "FLOOR PLANS, NOTES, AND SCHEDULE"], "12": ["MD100", "DEMOLITION FLOOR PLAN"], "13": ["M100", "RENOVATION FLOOR PLAN"],
                                     "14": ["M200", "DETAILS, NOTES, SCHEDULES, AND LEGEND"], "15": ["E001", "ELECTRICAL SYMBOL LEGEND & SCHEDULES"], "16": ["ED101", "ELECTRICAL DEMOLITION PLAN"],
                                     "17": ["E101", "LIGHTING RENOVATION PLAN"], "18": ["E201", "POWER/SYSTEMS RENOVATION PLAN"]},
        "code": {"code_edition": "International Building Code, 2012 Edition", "occupancy": "BUSINESS, Group B", "occupancy_type_table_5_6_5": "Institutional Outpatient B", "construction": "IIB",
                 "sprinklered": "no", "sprinklers_required_table_5_6_7": "Yes box ticked, comment Not for this project", "standpipes_required_table_5_6_7": "Yes box ticked, comment Not for this project",
                 "opening_protectives_required_table_5_6_7": "No box ticked", "area_of_work_sf": 4351, "third_floor_total_sf": 16950, "factor_sf_per_occ": 100, "occupant_load": 43, "stories_declared": "N/A"},
        "doors": {"317A1": ["3-0x7-0", "SET 3", "-"], "317B1": ["3-0x7-0", "SET 2", "1HR"], "317C1": ["3-0x7-0", "SET 1", "1HR"], "317D1": ["3-0x7-0", "SET 1", "1HR"],
                  "317E1": ["3-0x7-0", "SET 1", "1HR"], "317F1": ["3-0x7-0", "SET 1", "1HR"], "317G1": ["3-0x7-0", "SET 1", "1HR"], "317G2": ["3-0x7-0", "SET 1", "-"],
                  "317H1": ["3-0x7-0", "SET 1", "1HR"], "317I1": ["3-0x7-0", "SET 1", "1HR"], "317J1": ["3-0x7-0", "SET 1", "1HR"], "317K1": ["3-0x7-0", "SET 1", "1HR"],
                  "317L1": ["3-0x7-0", "SET 1", "1HR"], "317M1": ["3-0x7-0", "SET 1", "1HR"], "317N1": ["3-0x7-0", "SET 1", "1HR"], "317O1": ["3-0x7-0", "SET 1", "1HR"],
                  "317P1": ["3-0x7-0", "SET 1", "1HR"], "317Q1": ["3-0x7-0", "SET 1", "1HR"], "317R1": ["(2)3-0x7-0", "SET 4", "1HR"], "317S1": ["3-0x7-0", "SET 1", "1HR"],
                  "317T1": ["3-0x7-0", "SET 3", "-"]},
        "hardware_sets": {"SET1": "317C1 317D1 317E1 317F1 317G1 317G2 317H1 317I1 317J1 317K1 317L1 317M1 317N1 317O1 317P1 317Q1 317S1".split(), "SET2": ["317B1"], "SET3": ["317A1", "317T1"], "SET4": ["317R1"]},
        "tags_on_a101": "317A1 317B1 317C1 317D1 317E1 317F1 317G1 317G2 317H1 317I1 317J1 317K1 317L1 317M1 317N1 317O1 317P1 317Q1 317R1 317S1 317T1".split(),
        "finish_rooms": {"317": "ENTRY", "317A": "ELECTRICAL CLOSET", "H307": "CORRIDOR", "H308": "CORRIDOR", "H309": "CORRIDOR", "317B": "BREAK/WORK ROOM", "317C": "OFFICE 4", "317D": "OFFICE 5",
                         "317E": "OFFICE 6", "317F": "OFFICE 7", "317G": "ADMIN 8", "317H": "COEE OFFICE 9", "317I": "OFFICE 10", "317J": "OFFICE 11", "317K": "OFFICE 12", "317L": "OFFICE 13",
                         "317M": "OFFICE 14", "317N": "GRAD 15", "317O": "OFFICE 1", "317P": "OFFICE 2", "317Q": "OFFICE 3", "317R": "SEMINAR", "317S": "COLLABORATION ROOM", "317T": "CLOSET"},
        "exit_markers_g003": [{"note": "one EXIT LOCATIONS arrow in corridor H307, no width and no capacity printed"}],
        "travel_distance_g003": "RTD: 27'-8\"",
    },
    "nibs_clinic": {
        "index_sample": {"T1-3": "INDEX OF DRAWINGS", "A2-1": "COMPOSITE FIRST FLOOR PLAN", "A2-8": "COMPOSITE SECOND FLOOR PLAN", "A2-13": "UPPER LOBBY/MECHANICAL PENTHOUSE PLAN", "A4-1": "FINISH SCHEDULE NOTES",
                         "A4-2": "FINISH SCHEDULE", "A4-3": "FINISH SCHEDULE", "A4-4": "FINISH SCHEDULE", "A5-1": "DOOR SCHEDULE", "A5-2": "DOOR SCHEDULE (CONT.)", "A5-3": "DOOR SCHEDULE (CONT.)",
                         "A18-1": "LIFE SAFETY - COMPOSITE FIRST FLOOR PLAN", "A18-2": "LIFE SAFETY - COMPOSITE SECOND FLOOR PLAN", "P9-1": "FIRE PROTECTION - FIRST FLOOR COMPOSITE PLAN", "P9-2": "FIRE PROTECTION - SECOND FLOOR COMPOSITE PLAN"},
        "index_not_used_rows": ["A17-6", "EL2-8", "EP2-1", "EP2-8", "EC3-1", "EC3-2"],
        "index_rows_with_sequence": 316,
        "exit_markers_161": [{"where": "STAFF ENTRY top", "width_mm": 915, "capacity": 180}, {"where": "north side", "width_mm": 915, "capacity": 180}, {"where": "north east", "width_mm": 915, "capacity": 180},
                             {"where": "MAIN ENTRY east", "width_mm": 1830, "capacity": 360}, {"where": "STAFF ENTRY west", "width_mm": 915, "capacity": 180},
                             {"where": "south 1", "width_mm": 915, "capacity": 180}, {"where": "south 2", "width_mm": 915, "capacity": 180}],
        "exit_capacity_sum_161": 1440,
        "stair_boxes_162": [{"text": "STAIR WELL EXIT CAPACITY = 147 PEOPLE"}, {"text": "STAIR WELL EXIT CAPACITY = 147 PEOPLE"}],
        "exit_capacity_sum_162": 294,
        "travel_161": "137 FT/41.75 M",
        "travel_162": "153 FT / 46.65 M",
        "code": {"code_edition": "Uniform Building Code, no edition printed", "occupancy": "BUSINESS", "construction": "TYPE II (N)", "sprinklered": "yes, fully sprinklered", "occupant_load": "not printed", "area": "not printed", "stories_declared": "not printed"},
        "doors_079": D079, "doors_080": D080, "doors_081": D081,
        "door_specials": {"1AC1": ["PR 915", ""], "1B19": ["1220", ""], "1BC3": ["1065", ""], "1C11": ["PR 915", ""], "1BS2": ["915", "60"], "1CS3": ["915", "60"],
                          "1E15": ["PR 915", ""], "1E15A": ["PR 760", ""], "1E24": ["PR 915", ""], "1E24A": ["PR 865", ""], "1EC4": ["PR 1220 / 305", ""],
                          "2BS2": ["915", "60"], "2CS3": ["915", "60"], "2C12A": ["PR 610", ""], "2D05": ["PR 915", ""], "2D05A": ["PR 915", ""]},
    },
    "code_sheets": {
        "mml_capitol": {"code_edition": "2015 Michigan Residential Code", "occupancy": "not printed", "construction": "not printed", "sprinklered": "not printed", "levels_in_area_table": ["Ground", "Main", "Upper"], "area_total_sf": 1728},
        "mml_baxter": {"code_edition": "2015 Michigan Residential Code", "occupancy": "not printed", "construction": "not printed", "sprinklered": "not printed", "levels_in_area_table": ["Main", "Upper"], "area_total_sf": 2304},
        "mml_abbot": {"code_edition": "2015 Michigan Residential Code", "occupancy": "not printed", "construction": "not printed", "sprinklered": "not printed", "levels_in_area_table": [], "area_total_sf": None},
        "carpinteria_adu1": {"code_edition": "2022 CBC", "occupancy": "R-3", "construction": "VB", "sprinklered": "conditional", "stories": 1, "area_sf": 400},
        "carpinteria_adu2": {"code_edition": "2022 CBC", "occupancy": "R-3", "construction": "VB", "sprinklered": "conditional", "stories": 1, "area_sf": 480},
        "bcoe_addendum3": {"code_edition": "CBC 2019", "occupancy": "B", "sprinklered": "yes", "occupant_load": 357, "area_sf": 9273},
    },
}

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    assert len(set(D079)) == len(D079) and len(TRUTH["usc_leconte"]["doors"]) == 21 and len(TRUTH["usc_leconte"]["index"]) == 19
    OUT.write_text(json.dumps(TRUTH, indent=1))
    print("truth", hashlib.sha256(OUT.read_bytes()).hexdigest(), time.strftime("%Y-%m-%d %H:%M:%S %Z"), len(D079), len(D080), len(D081))
