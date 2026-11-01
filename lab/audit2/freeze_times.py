"""Compare artefact mtimes with declared freeze times. No API. Run with -I."""
import glob, json, os, datetime as dt
ts = lambda s: dt.datetime.fromisoformat("2026-10-31T" + s).timestamp()
iso = lambda t: dt.datetime.fromtimestamp(t).strftime("%H:%M:%S")
ROOT = ""
def earliest(pats):
    fs = [f for p in pats for f in glob.glob(ROOT + p)]
    fs = [(os.stat(f).st_mtime, f.replace(ROOT, "")) for f in fs]
    return sorted(fs)
def show(name, pats, cut):
    fs = earliest(pats)
    bad = [x for x in fs if x[0] < ts(cut)]
    print(f"{name}: {len(fs)} files, earliest {iso(fs[0][0]) if fs else '-'} ({fs[0][1] if fs else ''}), freeze {cut}, before-freeze {len(bad)}")
    for t, f in bad[:8]:
        print("    BEFORE", iso(t), f)
held_p = ["bme_k", "habs_va_bldg1", "wfu_huffman", "schenley", "habs_stelizabeths_atkins"]
show("presence clean runs", [f"data/lab/presence/spark/{b}_r*.json" for b in held_p], "16:18:50")
show("presence planted runs", [f"data/lab/presence/spark/{b}__*_r*.json" for b in held_p], "16:18:50")
show("presence crops any", ["data/lab/presence/crops/*.json"], "16:03:32")
show("presence cells", ["data/lab/presence/cells/*.json"], "16:18:50")
show("presence planted sheets", ["data/lab/presence/planted/*/*.jpg"], "16:01:57")
held_t = ["bme_k", "habs_va_bldg1", "wfu_huffman", "queens_royce1", "leonard4", "mml_abbot"]
show("tightseed per-floor grounding, held-out", [f"data/lab/tightseed/spark/{s}_f*.json" for s in held_t], "16:11:05")
show("tightseed clusters+seeds, held-out", [f"data/lab/tightseed/spark/{s}_clusters.json" for s in held_t] + [f"data/lab/tightseed/spark/{s}_seeds.json" for s in held_t], "16:11:05")
show("tightseed SAM tracks, held-out", [f"data/lab/tightseed/sam/{s}__*.json" for s in held_t], "16:11:05")
show("tightseed one-call comparator", [f"data/lab/tightseed/spark/{s}_floors.json" for s in held_t], "16:13:19")
show("tightseed P seeds", ["data/lab/tightseed/sam/*__P_960.json", "data/lab/tightseed/sam/*__Ppin_960.json"], "16:11:05")
show("tightseed royce_east any spark", ["data/lab/tightseed/spark/royce_east*"], "16:36:10")
show("tightseed royce_east any sam", ["data/lab/tightseed/sam/royce_east*"], "16:36:10")
show("tightseed queens_royce1 any spark", ["data/lab/tightseed/spark/queens_royce1*"], "15:52:58")
show("tightseed any spark (all)", ["data/lab/tightseed/spark/*"], "15:52:58")
show("agent runs", ["data/lab/agent/runs/*/*.json"], "16:01:26")
show("samtricks cache", ["data/lab/samtricks/cache/*"], "15:55:33")
