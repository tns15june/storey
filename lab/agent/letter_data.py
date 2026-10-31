"""Eight structured findings for the letter test. Sheets and values come from the Capitol and clinic results; two fields are left empty on purpose. No API."""
HEADER = {"subject": "Plan Check Comment Letter #1", "project": "Medical and Dental Clinic"}
FINDINGS = [
    {"id": 1, "sheet": "A18-1", "location": "life safety plan, travel path to Main Entry", "kind": "travel_distance", "printed": "137 FT./41.75 M", "measured": "35.11 m", "correction": "Reconcile the marked travel distance with the drawn route", "section": "CBC 1017.2"},
    {"id": 2, "sheet": "A18-2", "location": "life safety plan, travel path across corridor 2AC1", "kind": "travel_distance", "printed": "153 FT./46.65 M", "measured": "37.79 m", "correction": "Reconcile the marked travel distance with the drawn route", "section": "CBC 1017.2"},
    {"id": 3, "sheet": "A-059", "location": "stair 2BS2", "kind": "stair_continuity", "printed": "", "measured": "stair shown on floor 1, no treads drawn on floor 2", "correction": "Show the stair continuing to the second floor, or state how floor 2 is served", "section": "CBC 1006.3"},
    {"id": 4, "sheet": "A5.1", "location": "door 4, Closet 103", "kind": "door_mismatch", "printed": "42 in in door schedule", "measured": "36 in swing on plan", "correction": "Make the door schedule and the plan agree", "section": "CBC 1010.1.1"},
    {"id": 5, "sheet": "A2.1", "location": "Stair Hall 100, dimension string", "kind": "dimension_mismatch", "printed": "8'-7\"", "measured": "8'-1\"", "correction": "Correct the dimension or the drawing", "section": None},
    {"id": 6, "sheet": "A2.1", "location": "Hall 200", "kind": "corridor_width", "printed": "3'-9 1/2\"", "measured": "3'-5\" clear", "correction": "Dimension the clear width of the corridor", "section": "CBC 1020.3"},
    {"id": 7, "sheet": "A5.0", "location": "door 13, Closet 103, Unit 2", "kind": "door_mismatch", "printed": "60 in bifold in door schedule", "measured": "36 in swing on plan", "correction": "Make the door schedule and the plan agree", "section": "CBC 1010.1.1"},
    {"id": 8, "sheet": "A1.0", "location": "code summary", "kind": "occupant_load", "printed": "357 occupants", "measured": "357 occupants by room sum; room 119 gives 1 where 2 is declared", "correction": "Correct the declared occupant load of room 119", "section": "CBC 1004.5"},
]
EXAMPLE = """14. Sheet A0.01
 a. Provide following information on plan: parcel number, occupancy ratings, construction type, area tabulations, applicable codes, deferred approvals, north arrow.
15. Sheet A1.01 and A1.02
 a. Show all exterior windows.
 c. Demonstrate entrances and exterior exits comply with CBC 11B-202.4, 11B-404, CBC Section 1005.4 & 1022.2.2.
 e. "Gallery" room occupant load factor should be under Assembly without fixed seats, occupant load factor should be 5 per CBC Table 1004.1.2.
 f. Retail occupant load factor should be 60. (CBC 2016, Table 1004.1.2)
17. Sheet A2.11
 c. Provide dimensions of rooms, partitions and built-in furniture and stairways' width.
 f. Provide landing and path of travel for 2nd Exit near Kitchen. (CBC1009.1)
 g. Provide handrail at stair to Dishwash."""
PROMPT = """Draft a building plan check comment letter from the findings below, in the style of the excerpt. Group comments by sheet, sheet first. Write each comment as an instruction in the imperative. End a comment with its code section in parentheses when the finding gives one. Use only facts and numbers from the findings: add no code section, value, date, name, permit number or requirement that is not in them. Do not approve anything and do not say that anything complies. Start with the subject line and project name given, then the comments. Add no closing, signature or contact block.

Subject: {subject}
Project: {project}

Style excerpt from a real letter, for wording only. Nothing in it applies to this project:
{example}

Findings:
{findings}"""
