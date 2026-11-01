"""Are pinned-track masks SAM evidence, or the prompt box echoed back? No API.
For each flag SAM cancels on Ppin, was that floor itself a prompt floor for the track? Also mask rate on prompt floors against other floors, by truth."""
import json
import sys

sys.path.insert(0, "lab/audit2")
import rescore_tightseed as S


def main():
    tot = dict(prompt_present=[0, 0], prompt_absent=[0, 0], other_present=[0, 0], other_absent=[0, 0])
    cancelled = []
    for st in S.HELD + ["royce_east"]:
        tr = S.load(st, "Ppin_960")
        own = S.owners(st, tr)
        seeds = {}
        for o, f, b in tr["seeds"]:
            seeds.setdefault(str(o), set()).add(f)
        say = S.one_call(st) if st in S.HELD else {}
        for r in S.rows_of(st, tr):
            if not r["seeded"]:
                continue
            for o in own[r["stair"]]:
                pf = r["floor"] in seeds[o]
                mk = tr["tracks"].get(o, {}).get(str(r["floor"]))
                has = bool(mk and mk["area"] > 0 and (mk["conf"] or 0) >= .5)
                key = ("prompt_" if pf else "other_") + ("present" if r["present"] else "absent")
                tot[key][0] += has
                tot[key][1] += 1
            if st in S.HELD:
                v = say.get((r["stair"], r["floor"]), [])
                spark_ok = bool(v) and all(x == r["present"] for x in v)
                spark_absent = spark_ok != r["present"]
                if spark_absent and r["claimed"]:
                    onprompt = any(r["floor"] in seeds[o] for o in own[r["stair"]])
                    cancelled.append((st, r["stair"], r["floor"], "truth present" if r["present"] else "TRUTH ABSENT", "prompt floor" if onprompt else "propagated"))
    print("mask at conf>=.5 / track-floors")
    for k, v in tot.items():
        print(f"  {k:16s} {v[0]}/{v[1]}")
    print("flags cancelled by a Ppin mask, held-out")
    for c in cancelled:
        print("  ", c)
    print("on a prompt floor:", sum(c[4] == "prompt floor" for c in cancelled), "of", len(cancelled))


main()
