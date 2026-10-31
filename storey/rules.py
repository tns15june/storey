"""Rule lookups by jurisdiction. Every value comes from rules/rows.json, so a section or number on screen traces to one row."""
import json
import re
from pathlib import Path

JURISDICTIONS = {
    "houston": "Houston Building Code 2021",
    "philadelphia": "Philadelphia Building Code 2021",
    "seattle": "2021 Seattle Building Code",
}
# shortcut: Table 1020.3 has no equipment access, dwelling unit or stretcher rows, so stretcher corridors read under their 72 in, upgrade when a stage reads those uses
# shortcut: Table 1006.3.4(2) leaves out R-2 sleeping units and the travel column, upgrade when a check reads them
# shortcut: Seattle item 7 stops at story 5, so R-2 over a podium reads NP, upgrade when stages read occupancy per story
# shortcut: R-3 and R-4 read NP though 1006.3.4 item 4 lets them use one exit, so check 11 can raise a false item, upgrade when a stage tells them from dwellings
ROWS = json.loads((Path(__file__).parent / "rules" / "rows.json").read_text())


def _fits(want, got) -> bool:
    if got is None:
        return False   # a fact nobody read must never unlock a more lenient case
    if isinstance(want, dict):
        return isinstance(got, (int, float)) and want.get("min", got) <= got <= want.get("max", got)
    if isinstance(want, list):
        return got in want or (re.fullmatch(r"[A-Z]-\d", got) is not None and got[0] in want)   # R-2 is in group R, but I-2 is not in I-1, and a mixed A-3/I-4 is in no group
    return got == want


def rule(id: str, jurisdiction: str, **inputs) -> dict:
    """The first case of the row that fits: value, unit, section, adopted, title, plus conditions and confirm where a person must check them.

    Inputs a row may read: occupancy (one IBC group such as R-2), sprinklers ("yes" or "no"), voice_alarm, occupant_load (int), story (int above grade plane), construction.
    A caller splits a mixed occupancy and converts numbers: a mixed string or a number as text counts as unread and fits no case that needs it.
    The value "NP" means not permitted. Raises KeyError for an unknown id or jurisdiction, or when no case fits.
    """
    # shortcut: sprinklers "yes" is read as NFPA 13 throughout, so a 13R building gets NFPA 13 values, upgrade when the stages read the standard
    row, adopted = ROWS[id], JURISDICTIONS[jurisdiction]
    for case in row.get("overrides", {}).get(jurisdiction, row["cases"]):
        if all(_fits(want, inputs.get(key)) for key, want in case.get("when", {}).items()):
            return {"unit": row["unit"], "section": row["section"], "adopted": adopted, "title": row["title"], **{k: v for k, v in case.items() if k != "when"}}
    raise KeyError(f"{id}: no case fits {inputs}")


def stricter_when_unknown(id: str, jurisdiction: str, **inputs) -> dict:
    """The stricter of the sprinklered and unsprinklered rows. Use it when sprinklers is anything but "yes" or "no". NP is stricter than any number."""
    inputs.pop("sprinklers", None)
    yes = rule(id, jurisdiction, sprinklers="yes", **inputs)
    no = rule(id, jurisdiction, sprinklers="no", **inputs)
    if "NP" in (yes["value"], no["value"]):
        return no if no["value"] == "NP" else yes
    return (max if ROWS[id].get("stricter") == "higher" else min)(yes, no, key=lambda found: found["value"])
