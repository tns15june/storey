"""Count how many jurisdictions print each egress and life-safety check in their public lists.

Run: .venv/bin/python -I lab/gather_practice/rank.py
Reads data/lab/gather_practice/text/*.txt (made by fetch.py), writes rank.csv and rank_samples.txt.
This ranks how many cities print an item, not how often it is issued. No public data gives that for commercial egress.
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TXT = ROOT / "data/lab/gather_practice/text"
OUT = ROOT / "data/lab/gather_practice"

# correction = lists a plan checker issues; checklist = what an applicant must show. Same template counts once.
GROUPS = {
    "Burbank CA": ("correction", ["burbank_comm_mixed_corrlist_2022"]),
    "Orange County CA": ("correction", ["orange_county_ca_commercial_corrlist_2019"]),
    "Newport Beach CA": ("correction", ["newport_beach_11113", "newport_beach_11137", "newport_beach_6744"]),
    "Anaheim CA": ("correction", ["anaheim_commercial_comments"]),
    "Michigan template (Grand Rapids, Battle Creek)": ("checklist", ["grand_rapids_commercial_checklist", "battle_creek_commercial_checklist"]),
    "Philadelphia PA": ("checklist", ["philadelphia_p013_checklist_2023"]),
    "St. Paul MN": ("checklist", ["stpaul_commercial_checklist_2024"]),
    "St. Petersburg FL": ("checklist", ["stpete_commercial_checklist"]),
    "Orange County FL": ("checklist", ["orange_county_fl_commercial_guide"]),
    "Miami-Dade FL": ("checklist", ["miami_dade_building_commercial"]),
    "North Carolina (OSFM guideline, Appendix B form)": ("checklist", ["ncosfm_life_safety_guidelines", "gates_nc_appendix_b"]),
    "Seattle WA": ("checklist", ["seattle_ti_standards", "seattle_ti_checklist"]),
    "Clark County WA": ("checklist", ["clark_wa_code_summary"]),
    "Sammamish WA": ("checklist", ["sammamish_code_summary_2021"]),
    "Scott County MN": ("checklist", ["scott_co_mn_code_record"]),
    "Bend OR": ("checklist", ["bend_or_prescreen"]),
}

ITEMS = {
    "Occupant load shown or calculated, room by room": r"occupant load|occ\. load|occupant loading",
    "Occupancy classification stated": r"occupancy classification|occupancy group|occ\. group|type of occupancy|occupancy\(ies\)|occupancy:|use and occupancy|occupant use/category",
    "Code analysis or code summary on the cover sheet": r"code analysis|code summary|code record|code compliance summary|code analysis",
    "Sprinkler status stated": r"sprinkler",
    "Exit access travel distance": r"travel distance|distance of travel|travel distances",
    "Exit or egress plan submitted": r"exit (analysis )?plan|egress (analysis/)?plan|life safety plan|exiting plan|egress/exiting analysis|egress analysis",
    "Exit signs located": r"exit sign",
    "Rated corridors, stairs and shafts labelled and rated": r"(rated|fire.resistive|fire.resist\w*|one.hour|1.hour) (corridor|shaft|stair)|rating of corridors|corridors? . widths, fire rating|fire.resistive corridors|shafts/exit enclosures|stair/elevator shaft",
    "Exit width and capacity": r"exit width|egress width|width of (the )?exits?|required width|clear width|size of (required )?exits?|egress capacity|0\.2 inch|width of means of egress|size of exits",
    "Number of exits": r"number of (required )?exits|required number of exits|exits required|number of exit|how many exits|number, location",
    "Door hardware, panic and locking": r"panic hardware|fire exit hardware|door hardware|delayed egress|special locking|locking",
    "Egress lighting": r"egress lighting|means of egress (system )?(shall be )?illuminat|exit lighting|emergency lighting|exit pathway lighting|emergency egress lighting",
    "Stair or exit enclosure": r"exit enclosure|stairs? shall be enclosed|interior exit stair|exit passageway|fire rated shafts|enclosure",
    "Common path of egress travel": r"common path",
    "Handrails and guards": r"handrail|guard(s|rail)?\b",
    "Exit discharge and path to public way": r"exit discharge|public way|discharge directly",
    "Door swing direction": r"door swing|swing in (the )?direction|swings? of|swing of",
    "Exit separation or remoteness": r"diagonal|exit separation|separation of exits|remoteness|egress separation|exits are remote|placed a distance apart",
    "Dead-end corridor length": r"dead.?end",
    "Stair width": r"stair(way)?s?.{0,3} widths?|width of (stairs|stairways|each stairway)|corridors and stairways|stairs shall have a minimum width",
    "Corridor width": r"corridor width|width of corridors|corridors? . widths",
    "Stair tread, riser and landing geometry": r"tread|riser",
    "Accessible means of egress or area of refuge": r"accessible means of egress|areas? of (refuge|rescue|evacuation)|refuge",
    "Door schedule provided": r"door (and window |and hardware )?schedule|door/window schedule",
    "Egress through intervening rooms": r"intervening",
}


def text_of(stem):
    return " ".join((TXT / f"{stem}.pdf.txt").read_text().split())


def main():
    rows, samples = [], []
    corpus = {g: (kind, " ".join(text_of(s) for s in stems)) for g, (kind, stems) in GROUPS.items()}
    for item, pat in ITEMS.items():
        hits, corr = [], 0
        rx = re.compile(pat, re.I)
        for g, (kind, text) in corpus.items():
            m = rx.search(text)
            if m:
                hits.append(g)
                corr += kind == "correction"
                samples.append(f"{item} | {g} | ...{text[max(0, m.start() - 70):m.end() + 70]}...")
        rows.append({"item": item, "jurisdictions": len(hits), "correction_lists": corr, "of": len(GROUPS), "where": "; ".join(hits)})
    rows.sort(key=lambda r: (-r["jurisdictions"], -r["correction_lists"], r["item"]))
    assert len(rows) == 25
    # self-check: a phrase read by eye in Burbank must be found, and a phrase absent from Bend must stay absent
    burbank = corpus["Burbank CA"][1]
    assert re.search(ITEMS["Dead-end corridor length"], burbank, re.I)
    assert not re.search(ITEMS["Dead-end corridor length"], corpus["Bend OR"][1], re.I)
    with open(OUT / "rank.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "rank_samples.txt").write_text("\n".join(samples))
    for i, r in enumerate(rows, 1):
        print(f"{i:2d} {r['jurisdictions']:2d}/{r['of']} corr {r['correction_lists']}/4  {r['item']}")


if __name__ == "__main__":
    main()
