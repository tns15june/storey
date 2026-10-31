"""The decision rule for one cell of the stair and shaft matrix. No model call and no truth here. Frozen before any held-out stack was run: see data/lab/presence/rule_meta.json.

Evidence per cell: v, runs out of 3 with a present point at this place on this floor; crop, the blind crop answer or None; ink, ink in a 60 px box at the registered place over the median of the
floors at least two runs called present, or None; blank, nothing drawn at the place; aligned, the floor passed the inlier gate. A row is established when two runs agree on some floor.

Three arms, all declared here before held-out:
  A  the design as specified: a 3 of 3 cell is Found with no check; crop and ink run only where runs disagree or say absent.
  B  A plus the ink check, which costs nothing, on every Found candidate. Stairs only.
  C  B plus a crop on every Found candidate. This is the arm proposed for the product if held-out agrees with dev.
Checks only ever move a cell toward Check. They never lift a cell to Found. Nothing is shown as passed: Found means a stair is drawn at this place on this floor, no more."""

INK_LOW = 0.40  # dev: real stairs 0.48 and up, erased stairs 0.27 and down, n 6 erased
INK_HIGH = 0.90  # dev: a Missing candidate with this much ink is worth a look; stairs absent on dev rarely reach it
ARMS = ("A", "B", "C")


def decide(r, c, arm):
    """r: row dict with established, fam, ink30. c: its cell dict. Returns (state, reason); state None means the cell is not shown."""
    v, crop, blank = c["v"], c.get("crop"), c.get("blank")
    ink = r["ink"] if r["fam"] == "stair" else None  # a 60 px box says nothing about a 25 px shaft
    if not r["established"]:
        if v == 0 or blank or crop == "no":
            return None, "single run, not confirmed"
        return "Check", "seen by one run only"
    if not c["aligned"]:
        return "Check", "floor unaligned"
    if v >= 2:
        guarded = v < 3 or arm in ("B", "C")
        if guarded and ink is not None and ink < INK_LOW:
            return "Check", "ink gone at the registered place"
        asks = v < 3 or arm == "C"
        if asks and blank:
            return "Check", "runs claim it, nothing is drawn"
        if asks and crop in ("no", "unclear"):
            return "Check", f"crop says {crop}"
        return "Found", f"{v} of 3 runs"
    if blank:
        return "Missing", "nothing drawn here"
    if crop in ("yes", "unclear"):
        return "Check", f"crop says {crop}"
    if ink is not None and ink >= INK_HIGH:
        return "Check", "ink as dense as on floors with the stair"
    return "Missing", f"{v} of 3 runs"


def apply_rule(E, arm):
    """E: evidence file for one stack. Returns decided cells, each with floor, marker, state, reason, row id."""
    out = []
    for r in E["rows"]:
        for f, c in r["cells"].items():
            q = dict(r, ink=r["ink30"].get(f))
            st, why = decide(q, c, arm)
            out.append(dict(f=int(f), loc=c["loc"], chk=c["chk"], state=st, why=why, row=r["id"], fam=r["fam"], v=c["v"]))
    return out


if __name__ == "__main__":
    R = lambda **k: dict(dict(established=True, fam="stair", ink=1.0), **k)
    C = lambda **k: dict(dict(v=3, crop=None, blank=False, aligned=True), **k)
    assert decide(R(), C(), "A")[0] == "Found"
    assert decide(R(ink=0.1), C(), "A")[0] == "Found"  # arm A leaves a 3 of 3 cell alone, the gap this experiment measures
    assert decide(R(ink=0.1), C(), "B")[0] == "Check"
    assert decide(R(), C(crop="no"), "B")[0] == "Found" and decide(R(), C(crop="no"), "C")[0] == "Check"
    assert decide(R(), C(v=2, crop="yes"), "A")[0] == "Found" and decide(R(), C(v=2, crop="no"), "A")[0] == "Check" and decide(R(), C(v=2, blank=True), "A")[0] == "Check"
    assert decide(R(ink=0.1), C(v=0, crop="no"), "A")[0] == "Missing" and decide(R(), C(v=0, crop="yes"), "C")[0] == "Check" and decide(R(ink=None), C(v=1, blank=True), "C")[0] == "Missing"
    assert decide(R(ink=0.95), C(v=0, crop="no"), "A")[0] == "Check"
    assert decide(R(), C(aligned=False), "C")[0] == "Check"
    assert decide(R(established=False), C(v=1, crop="yes"), "C")[0] == "Check" and decide(R(established=False), C(v=1, crop="no"), "C")[0] is None
    assert decide(R(fam="shaft", ink=0.1), C(), "C")[0] == "Found"
    print("rule self-check ok")
