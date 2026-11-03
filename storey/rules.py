"""Rule lookups by jurisdiction. Every value comes from rules/rows.json, so a section or number on screen traces to one row."""
import json
import re
from pathlib import Path

JURISDICTIONS = {
    "houston": "Houston Building Code 2021",
    "philadelphia": "Philadelphia Building Code 2021",
    "seattle": "2021 Seattle Building Code",
}
# shortcut: Table 1020.3's equipment access and dwelling unit rows ask less and are left out, so such a corridor can raise a false item, upgrade when a stage reads a corridor's use
# shortcut: every R-2 reads Table 1006.3.4(1), which asks a person to confirm dwelling units, and Table 1006.3.4(2)'s travel limit is a condition no route is compared with, upgrade when a check reads them
# shortcut: Seattle item 7 stops at story 5, so R-2 over a podium reads NP, upgrade when stages read occupancy per story
# shortcut: R-3 and R-4 read NP though 1006.3.4 item 4 lets them use one exit, so check 11 can raise a false item, upgrade when a stage tells them from dwellings
ROWS = json.loads((Path(__file__).parent / "rules" / "rows.json").read_text())
SYSTEMS = ("no", "NFPA 13R", "NFPA 13")   # every sprinkler status a row names. NFPA 13D, or a system whose standard was not read, fits no sprinklered case


def _fits(want, got) -> bool:
    if got is None:
        return False   # a fact nobody read must never unlock a more lenient case
    if isinstance(want, dict):
        return isinstance(got, (int, float)) and want.get("min", got) <= got <= want.get("max", got)
    if isinstance(want, list):
        return got in want or (re.fullmatch(r"[A-Z]-\d", got) is not None and got[0] in want)   # R-2 is in group R, but I-2 is not in I-1, and a mixed A-3/I-4 is in no group
    return got == want


def rule(id: str, jurisdiction: str, **inputs) -> dict:
    """The first case of the row that fits: value, unit, section, adopted, title, plus conditions and confirm where a person must check them,
    and unless, the earlier cases that every fact read fits but that need a fact nobody read, so the value is wrong if one of them holds.

    Inputs a row may read: occupancy (one IBC group such as R-2), sprinklers ("no", or the standard of the system: "NFPA 13", "NFPA 13R" or "NFPA 13D"), voice_alarm,
    occupant_load (int), story (int above grade plane), construction, use (a corridor's use, which no stage reads).
    A caller splits a mixed occupancy and converts numbers: a mixed string or a number as text counts as unread and fits no case that needs it. A sprinklered case names the standards it takes,
    so a system whose standard was not read gets the case without sprinklers.
    The value "NP" means not permitted. Raises KeyError for an unknown id or jurisdiction, or when no case fits.
    """
    row, adopted = ROWS[id], JURISDICTIONS[jurisdiction]
    unless = []
    for case in row.get("overrides", {}).get(jurisdiction, row["cases"]):
        when, found = case.get("when", {}), {k: v for k, v in case.items() if k != "when"}
        if all(_fits(want, inputs.get(key)) for key, want in when.items()):
            return {"unit": row["unit"], "section": row["section"], "adopted": adopted, "title": row["title"], **found, **({"unless": unless} if unless else {})}
        if all(_fits(want, inputs[key]) for key, want in when.items() if inputs.get(key) is not None):
            unless.append(found)
    raise KeyError(f"{id}: no case fits {inputs}")


def stricter_when_unknown(id: str, jurisdiction: str, **inputs) -> dict:
    """The strictest row over every sprinkler status. Use it when the status, or the standard of a system, was not read. NP is stricter than any number."""
    inputs.pop("sprinklers", None)
    found = [rule(id, jurisdiction, sprinklers=status, **inputs) for status in SYSTEMS]
    return next((row for row in found if row["value"] == "NP"), None) or (max if ROWS[id].get("stricter") == "higher" else min)(found, key=lambda row: row["value"])
