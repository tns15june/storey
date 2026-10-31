"""My verdict on each flag after opening its crop. Keys are flag ids. Values: (verdict, root cause id, reading step for a false alarm or note)."""
DEV = {
    "mml_capitol.1.title_mismatch.0": ("real", "capitol-a21-title", "known defect K2: index and drawing say Ground, title block says MAIN"),
    "mml_capitol.1.title_mismatch.1": ("real", "capitol-p61-index", "index row P6.1 repeats HVAC Cover Sheet, the sheet is PLUMBING SCHEDULES"),
    "mml_capitol.6.floor_label.0": ("real", "capitol-a21-title", "known defect K2, same root as the index flag"),
    "mml_capitol.5.schedule_cell.0": ("real", "capitol-row8", "row 8 SWING reads STEEL and MATERIAL reads WOOD; row 1, the matching entry door, reads LEFT and STEEL"),
    "mml_capitol.7.schedule_room_not_on_plan.0": ("real", "capitol-bedroom2-201", "schedule 205 BEDROOM 2; the plan labels Bedroom 2 as 201, the number of Bedroom 1"),
    "mml_capitol.3.sprinkler_missing.0": ("real", "capitol-sprinkler", "no sprinkler statement anywhere but an electrical legend. Residential prototype, so a Check at most"),
    "mml_baxter.1.title_mismatch.0": ("likely", "baxter-a40-title", "index Building Elevations; title block FRONT & RIGHT; the sheet draws all four elevations"),
    "mml_baxter.1.title_mismatch.1": ("real", "baxter-x10-title", "index Triplex Front Elevation + Plan, title block Townhouse PLANS & ELEVATIONS"),
    "mml_baxter.1.index_duplicate.0": ("real", "baxter-e51-index", "E2.1 listed twice; second row is the E5.1 sheet"),
    "mml_baxter.1.index_missing_sheet.0": ("real", "baxter-p61-index", "index says P1.6, the sheet is P6.1"),
    "mml_baxter.1.unindexed_sheet.0": ("real", "baxter-e51-index", "same root as the E2.1 duplicate"),
    "mml_baxter.1.unindexed_sheet.1": ("real", "baxter-p61-index", "same root as P1.6"),
    "mml_baxter.5.row_not_on_plan.0": ("real", "baxter-tag9", "Unit 2 rear door has no tag; row 9 is unused"),
    "mml_baxter.5.type_vs_arc.0": ("real", "baxter-unit2-reversed", "tag 15 sits on a swing door, row 15 says POCKET. Unit 2 tags run in reverse of the schedule"),
    "mml_baxter.5.type_vs_arc.1": ("real", "baxter-unit2-reversed", "known defect K1. Same reversal, wider than rows 12 and 13"),
    "mml_baxter.5.window_qty.0": ("real", "baxter-window-a", "schedule A QTY 14; plan draws and tags 12 type A windows, plus 2 type C"),
    "mml_baxter.5.row_not_on_plan.1": ("real", "baxter-upper-tags", "upper schedule marks 1 to 18, plan tags 17 to 34"),
    "mml_baxter.5.tag_not_in_schedule.0": ("real", "baxter-upper-tags", "same root"),
    "mml_baxter.3.sprinkler_missing.0": ("real", "baxter-sprinkler", "no sprinkler statement. Residential prototype, so a Check at most"),
    "usc_leconte.3.sprinkler_conflict.0": ("real", "usc-sprinkler", "G002 says no system; its ceiling note I places sprinkler heads and Table 5.6-7 ticks Yes with Not for this project. A121 has no heads"),
    "usc_leconte.3.occupancy_conflict.0": ("likely", "usc-occupancy", "Group B, but Table 5.6-5 says Institutional Outpatient on a math department office floor"),
    "usc_leconte.4.occupant_calc.0": ("likely", "usc-occ-43", "4,351 / 100 = 43.51, printed 43. Rounded down, which the code does not support"),
    "usc_leconte.8.marker_no_capacity.0": ("likely", "usc-exit-capacity", "exit location prints no width or capacity, so capacity cannot be checked; also it sits at a corridor door, not an exit"),
    "carpinteria_adu1.1.unindexed_sheet.0": ("real", "adu1-s1312", "S1-312 CONCRETE DETAILS is missing from the cover index; the structural index on S1-101 lists it"),
    "bcoe_addendum3.4.occupant_calc.0": ("likely", "bcoe-119", "150 SF / 150 prints 2 on the table and the plan. The area is probably rounded for display"),
    "bcoe_addendum3.4.worked_example.0": ("real", "bcoe-example", "known defect K3"),
    "bcoe_addendum3.4.exit_sum_vs_total.0": ("likely", "bcoe-exit-sum", "exit table carries 363, occupant total 357. Over by 6, the safe direction, but the numbers do not close"),
    "bcoe_addendum3.4.worked_example.1": ("real", "bcoe-net-area", "note 9,273 SF; 11,538 less the 2,463 SF of rows with no load is 9,075 SF"),
}
# v1 flags that v2 removed, with the reading step that caused each. All are dev flags.
DEV_V1_FALSE = {
    "width_vs_arc Capitol tags 1, 1, 5, 6": (4, "tolerance: door blocks draw 2.3 to 2.5 in under the schedule"),
    "width_vs_arc ADU tag 13": (1, "arc pairing: global assignment gave door 12's arc to sliding door 13"),
    "type_vs_arc Capitol tag 6, Baxter tag 11": (2, "arc detector took a dashed optional swing for a door"),
    "plan_room_not_in_schedule USC A101": (8, "room parser: 3 sheet references plus 5 existing rooms beyond the area of work"),
    "plan_room_not_in_schedule Capitol 910": (1, "room parser: elevation marker digits read as a room number"),
    "area_sum BCOE": (1, "tolerance: 50 rows rounded for display sum 4 SF over the total"),
    "exit_label_vs_table BCOE exit 6": (1, "text line merge: the count shares a line with a revision number"),
}

# Sprinkler silence on a Michigan duplex is explained by convention, so these are likely, not real.
for _k in ("mml_capitol.3.sprinkler_missing.0", "mml_baxter.3.sprinkler_missing.0"):
    DEV[_k] = ("likely", _k.split(".")[0] + "-sprinkler", "no sprinkler statement. Residential prototype, the code may not ask for one")
DEV["bcoe_addendum3.4.worked_example.1"] = ("likely", "bcoe-net-area", "note 9,273 SF; 11,538 less the 2,463 SF of rows with no load is 9,075 SF. The note does not list which rows it removes")

NOT_USED = "rule: NOT USED rows are not sheets"
NORM = "title normaliser"
PROXY = "proxy: the IFC model stands in for the plan and is not the drawing"
GHOST = "strip crop: the header clip leaks a cut first row, read as a ghost row at the top of each strip"
MARK = "Spark marker read: pull stations, exit signs and detectors read as exit markers"
HELD = [
    # set, sub, text substring, verdict, root, step or note
    ("mml_abbot", "index_missing_sheet", "B1.0", "real", "abbot-b10", "index lists B1.0, the set has one cover, A1.0"),
    ("mml_abbot", "title_mismatch", "A1.0", "false", "", NORM + ": Info against INFORMATION"),
    ("mml_abbot", "title_mismatch", "A2.0", "false", "", "index reader dropped the wrapped second line, title block reader kept the project name"),
    ("mml_abbot", "title_mismatch", "X4.1", "false", "", NORM + ": same content in other words"),
    ("mml_abbot", "title_mismatch", "X5.1", "likely", "abbot-x51", "title block TRIPLE, index Triplex under a Townhouse heading"),
    ("mml_abbot", "title_mismatch", "B2.0", "false", "", "title block reader kept the project name"),
    ("mml_abbot", "title_mismatch", "L1.0", "false", "", NORM + ": same content in other words"),
    ("mml_abbot", "sprinkler_missing", "", "likely", "abbot-sprinkler", "no sprinkler statement. Residential prototype, the code may not ask for one"),
    ("mml_abbot", "tag_not_in_schedule", "20", "real", "abbot-tag20", "two garage openings carry tag 20, the door schedule stops at row 11"),
    ("mml_abbot", "width_vs_arc", "tag 11", "likely", "abbot-door11", "row 11 says 32 in, the plan and the alternate plan draw 36 in. Sheets say do not scale"),
    ("mml_abbot", "schedule_cell", "", "false", "", "rule: SWING vocabulary was set on dev, PAIR, PAIRED and OPEN are valid here"),
    ("mml_abbot", "row_not_on_plan", "row C", "likely", "abbot-window-c", "window row C has no tag on any plan sheet. Could be an unused option"),
    ("carpinteria_adu2", "index_missing_sheet", "AD-901", "real", "adu2-ad901", "AD-901 is on the index and not in the PDF. Plan 1 has it, with the flashing and vent details"),
    ("carpinteria_adu2", "unindexed_sheet", "S2-312", "real", "adu2-s2312", "same omission as Plan 1"),
    ("carpinteria_adu2", "tag_not_in_schedule", "24", "real", "adu2-tag24", "optional door tag 24 on A2-101, schedule rows 21, 22, 23, 25"),
    ("af_ces_shops", "plan_key_not_in_legend", "", "false", "", "text line merge: legend row key and name came as one line"),
    ("nibs_clinic", "index_missing_sheet", "NOT USED", "false", "", NOT_USED),
    ("nibs_clinic", "index_missing_sheet", "S4-6", "real", "clinic-s4-s5", "index S4-6 lobby roof, the sheet with the same NAVFAC number is S5-1, and S5-7 is unindexed"),
    ("nibs_clinic", "index_missing_sheet", "P3-2", "false", "", "Spark title block: one read gave the drawing number, so the sheet was dropped and its index row looked orphaned"),
    ("nibs_clinic", "unindexed_sheet", "S5-7", "real", "clinic-s4-s5", "same root as S4-6"),
    ("nibs_clinic", "title_mismatch", "GT-", "likely", "clinic-gt-title", "index LOGS, title block DATA. Trivial"),
    ("nibs_clinic", "title_mismatch", "A2-13", "false", "", NORM + ": MECH. against MECHANICAL"),
    ("nibs_clinic", "title_mismatch", "A5-4", "false", "", NORM + ": THRESH HOLD against THRESHOLD"),
    ("nibs_clinic", "title_mismatch", "A8-10", "real", "clinic-a8-atrium", "index BUILDING SECTION AT ATRIUM, title block WALL SECTIONS"),
    ("nibs_clinic", "title_mismatch", "A8-11", "real", "clinic-a8-atrium", "same"),
    ("nibs_clinic", "title_mismatch", "A8-1", "false", "", NORM + ": generic index title DETAILS"),
    ("nibs_clinic", "title_mismatch", "A8-2", "false", "", NORM + ": generic index title DETAILS"),
    ("nibs_clinic", "title_mismatch", "A10-2", "likely", "clinic-a10-2", "index says SCHEDULE, the sheet is details only. Trivial"),
    ("nibs_clinic", "title_mismatch", "A11-7", "likely", "clinic-a11-7", "index misspells V-WHEEL as V-WEIL. Trivial"),
    ("nibs_clinic", "title_mismatch", "P2-", "false", "", NORM + ": waste piping against drain, waste and vent"),
    ("nibs_clinic", "title_mismatch", "P3-3", "real", "clinic-p33-block", "title block says FIRST FLOOR PLAN - BLOCK 2C, index and sequence say 1C"),
    ("nibs_clinic", "title_mismatch", "P3-9", "likely", "clinic-p39-typo", "title block misspells PIPIING. Trivial"),
    ("nibs_clinic", "title_mismatch", "P5-1", "real", "clinic-p51-title", "index WATER RISER DIAGRAMS, title block WASTE RISER DIAGRAM"),
    ("nibs_clinic", "title_mismatch", "P9-", "false", "", NORM + ": fire protection against fire suppression"),
    ("nibs_clinic", "edition_missing", "", "likely", "clinic-edition", "notes on 161 and 162 cite U.B.C. with no edition, the cover has none. Other sheets were not read"),
    ("nibs_clinic", "door_room_not_in_finish", "", "false", "", GHOST),
    ("nibs_clinic", "room_name_mismatch", "door 1B04", "false", "", GHOST),
    ("nibs_clinic", "room_name_mismatch", "door 1E20", "false", "", GHOST),
    ("nibs_clinic", "room_name_mismatch", "door 1E15A", "false", "", "rule: door suffix stripped although room 1E15A has its own finish row"),
    ("nibs_clinic", "room_name_mismatch", "door 2D04A", "false", "", "rule: door suffix stripped although room 2D04A has its own finish row"),
    ("nibs_clinic", "room_name_mismatch", "door 1E24", "likely", "clinic-1e24", "finish RECEIV./PROCESS., door schedule and plan RECEIVING/STORAGE"),
    ("nibs_clinic", "room_name_mismatch", "room 1E24", "likely", "clinic-1e24", "same"),
    ("nibs_clinic", "room_name_mismatch", "door 1E2", "real", "clinic-1e25-28", "finish rows 1E25 to 1E28 hold the wrong rooms. Plan, door schedule and model agree, so the finishes attach to the wrong numbers"),
    ("nibs_clinic", "room_name_mismatch", "room 1E2", "real", "clinic-1e25-28", "same"),
    ("nibs_clinic", "room_name_mismatch", "room 2B19", "real", "clinic-2b19", "finish and door schedule WATER LAB, plan 162 OFFICE"),
    ("nibs_clinic", "room_name_mismatch", "room 1D39", "false", "", PROXY),
    ("nibs_clinic", "room_name_mismatch", "room 1E14", "false", "", PROXY),
    ("nibs_clinic", "room_name_mismatch", "room 2C24", "false", "", "room name matcher: WOMENS TLT. against W. TOILET"),
    ("nibs_clinic", "plan_room_not_in_schedule", "1D20", "false", "", "two-read agreement: reads gave APPPMTS and APPMTS, so the row was dropped"),
    ("nibs_clinic", "plan_room_not_in_schedule", "", "false", "", PROXY),
    ("nibs_clinic", "schedule_room_not_on_plan", "", "false", "", PROXY),
    ("nibs_clinic", "marker_no_capacity", "", "false", "", MARK),
]
TRIVIAL_ROOTS = {"clinic-gt-title", "clinic-a10-2", "clinic-a11-7", "clinic-p39-typo", "abbot-x51"}


def held_verdict(f):
    """First matching row wins, so specific rows sit above general ones."""
    for s, sub, key, v, root, note in HELD:
        if s == f["set"] and sub == f["sub"] and key in f["sheet"] + " " + f["text"]:
            return v, root, note
    raise KeyError(f["id"])
