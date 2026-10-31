"""Decision rules for the declared-against-drawn checks. Frozen by sha256 in data/lab/declared/stamps.json.

v1 was written before any check ran. Any change to this file after flags have been seen gets a new VERSION and a
line in CHANGES, and both counts are reported. Changes made after held-out results are marked post hoc.
"""
import math
import re

VERSION = "v2"
CHANGES = ["v1: first version, before any check ran.",
           "v2: tuned on dev only, before any held-out read. Door width tolerance 1.5 to 3 in (door blocks draw 2.3 to 2.5 in under the schedule, every MML sheet says do not scale). "
           "Area sum tolerance 1 SF to 0.2 percent (rows are rounded for display). LEVEL n counts as a floor word. In the code: dashed arcs are no longer doors, an arc pairs with its nearest tag, "
           "plan rooms need a name and must sit inside the area the schedule covers, sheet numbers are not rooms, exit labels search 130 pt on words. "
           "A trial subcheck, a swing door with no arc beside its tag, raised 6 dev flags with 1 real, because the arc detector misses solid arcs. It was removed before held-out."]

# Dev sets tune the readers and rules. Held-out sets run once on frozen code.
# Siblings: abbot~capitol/baxter, adu2~adu1, shops~admin, huffman~efird. Only nibs_clinic and brandeis_hall are independent.
DEV = ("mml_capitol", "mml_baxter", "usc_leconte", "carpinteria_adu1", "af_ces_admin", "wfu_efird", "bcoe_addendum3")
HELD = ("mml_abbot", "carpinteria_adu2", "af_ces_shops", "wfu_huffman", "brandeis_hall", "nibs_clinic")

# Not submissions: true "absent" findings on these sets go in a separate row and never count as yield.
NOT_A_SUBMISSION = ("af_ces_admin", "af_ces_shops", "wfu_efird", "wfu_huffman", "brandeis_hall")
RESIDENTIAL = ("mml_capitol", "mml_baxter", "mml_abbot", "carpinteria_adu1", "carpinteria_adu2")  # check 2 list is for commercial sets

DOOR_WIDTH_TOL_IN = 3.0      # v1 was 1.5, the value in lab/modern/doors.py
EXIT_IN_PER_OCC = 0.2        # 5.08 mm per occupant, doors
STAIR_IN_PER_OCC = 0.3       # 7.62 mm per occupant, stairs
MM_PER_IN = 25.4
ROUNDING_TOL_IN = 0.05       # printed width against occupants times factor
TILE_DEDUP_PERMILLE = 12     # two exit markers this close on a 0-1000 sheet with the same width and capacity are one marker
SPARK_READS_TO_FLAG = 2      # a flag that rests on a Spark read must appear in 2 of 2 reads
AREA_SUM_TOL_FRAC = 0.002    # printed total against the sum of rows, v1 was 1 SF
FT_TO_M_TOL = 0.05           # printed metric distance against printed feet

STOP = {"AND", "THE", "OF", "FOR", "A", "ONLY", "REFERENCE", "SHEET"}
FLOOR_WORDS = {"GROUND": "GROUND", "MAIN": "MAIN", "FIRST": "1", "1ST": "1", "SECOND": "2", "2ND": "2", "THIRD": "3", "3RD": "3", "3D": "3",
               "FOURTH": "4", "4TH": "4", "UPPER": "UPPER", "LOWER": "LOWER", "BASEMENT": "BASEMENT", "ATTIC": "ATTIC", "PENTHOUSE": "PENTHOUSE"}


def norm(s):
    """Uppercase, & to AND, punctuation to space, collapsed whitespace."""
    s = (s or "").upper().replace("&", " AND ")
    return " ".join(re.sub(r"[^A-Z0-9/.]+", " ", s).replace(".", " ").split())


def tokens(s):
    """Normalised words with stop words dropped and a trailing plural S folded."""
    out = set()
    for t in norm(s).replace("/", " ").split():
        if t in STOP:
            continue
        out.add(t[:-1] if len(t) > 3 and t.endswith("S") and not t.endswith("SS") else t)
    return out


def floors(s):
    """Floor words in a label, as canonical keys."""
    return {FLOOR_WORDS[t] for t in norm(s).split() if t in FLOOR_WORDS} | set(re.findall(r"LEVEL (\d)", norm(s)))


def title_same(a, b):
    """Two titles name the same sheet: equal word sets, or one holds the other with at least two shared words, and no floor word conflict."""
    ta, tb = tokens(a), tokens(b)
    fa, fb = floors(a), floors(b)
    if fa and fb and not fa & fb:
        return False
    if ta == tb:
        return True
    return len(ta & tb) >= 2 and (ta <= tb or tb <= ta)


def occupants(area, factor):
    """Occupant load rounded up. IBC 1004 gives no fraction, so ceiling is the safe reading."""
    return math.ceil(area / factor - 1e-9)


def exit_required_in(occ, per=EXIT_IN_PER_OCC):
    return occ * per


def capacity_from_mm(width_mm, per=EXIT_IN_PER_OCC):
    """People a width carries, rounded down."""
    return math.floor(width_mm / MM_PER_IN / per + 1e-9)


# Verdicts, set by opening a crop of each flag.
# real: the crop shows two places that disagree in substance, printed arithmetic that is wrong, or a required item absent.
# likely: the crop shows the disagreement, but a convention or an option could explain it.
# false: the crop shows no disagreement. The verdict names the reading step that caused the flag.
# cannot_tell: the crop does not settle it.
#
# Ship rule, fixed before held-out. p = (real + likely) / flags, pooled over dev and held-out, cannot_tell counted as not real.
#   Fix:  flags >= 3, p >= 0.80, and no false alarm in the held-out sets.
#   Check: flags >= 3 and p >= 0.50, or flags < 3 with at least one real.
#   Drop: p < 0.50 with flags >= 3, or no real flag anywhere.
#   Cannot tell is the fixed outcome wherever a reading step did not complete, so a missing read never reads as a pass.
if __name__ == "__main__":
    assert norm("A2.1  Ground Floor Plan") == "A2 1 GROUND FLOOR PLAN"
    assert title_same("Attic & Roof Plan", "ATTIC & ROOF PLANS")
    assert title_same("Building Section & Wall Section", "BUILDING AND WALL SECTION")
    assert not title_same("Ground Floor Plan", "MAIN FLOOR PLAN")
    assert not title_same("HVAC Cover Sheet", "DOMESTIC WATER PIPING SCHEMATIC")
    assert occupants(4351, 100) == 44 and occupants(150, 150) == 1
    assert capacity_from_mm(915) == 180 and capacity_from_mm(1830) == 360
    print("ok", VERSION)
