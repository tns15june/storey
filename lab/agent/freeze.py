"""Select the 32 checked cases, shuffle, rename, write truth and rule, print sha256. Run once, before any model call on these sheets. No API."""
import hashlib, json, random, shutil, sys, time
from pathlib import Path
ROOT = Path("."); D = ROOT / "data/lab/agent"; CAND = D / "cand"; OUT = D / "cases"
sys.path.insert(0, str(ROOT / "lab/agent"))
import prompts
PICK = """stair_bad_E stair_bad_B stair_bad_F stair_bad_D stair_good_bme__14_0 stair_good_bme__6_3 stair_good_habs_4_0 stair_good_wfu__2_1
route_bad_100 route_bad_72 route_plant_1_30 route_plant_2_44 route_good_1_8 route_good_1_29 route_good_2_8 route_good_2_55
door_bad_0 door_bad_1 door_bad_2 door_bad_3 door_good_4 door_good_5 door_good_6 door_good_7
value_wrong_2 value_wrong_4 value_misq_cm_0 value_neigh_bm_1 value_agree_0 value_ok_cm_1 value_true_cm_1 value_true_bm_1""".split()
RULE = {
    "caught": "verdict is defect on a bad case",
    "false_rejection": "verdict is defect on a good case",
    "unparsed": "a truncated or unparseable answer counts as not flagged on both sides and is counted apart",
    "located": "routes only, shown beside the score, never replacing it: a catch counts as located if its x,y is within 150 of the true crossing on the 0-1000 grid",
    "baselines": "always-accept catches 0/16 and rejects 0/16; always-reject catches 16/16 and rejects 16/16; blind control is Spark low on the claim text with no image",
    "config_choice": "highest catch on all 16 bad cases, ties broken by lower false rejection, then lower cost",
    "recommend_agent_drives": "needs catch >= 12/16 and false rejection <= 4/16 on the chosen config, and on the end-to-end run: every seeded finding filed, zero invented findings, at most one false finding, at most 1.00 USD and 10 minutes per run",
    "recommend_audit_after_pipeline": "needs catch >= 12/16, false rejection <= 4/16, catch on real mistakes >= 8/11, and at most 0.05 USD per overlay on the chosen config",
    "recommend_no_agent": "if neither bar is met",
    "wonder_medium_subset": "if projected total spend passes 4.5 USD, Wonder medium runs only the first 16 cases in shuffled order and is reported as a subset",
}
cases = []
all_c = {c["img"][:-4]: c for f in ("stairs", "routes", "doors", "values") for c in json.load(open(CAND / f"{f}.json"))}
for k in PICK:
    cases.append(dict(all_c[k], key=k))
assert len(cases) == 32 and sum(c["label"] == "bad" for c in cases) == 16
for cat in ("stair", "route", "door", "value"):
    assert sum(c["cat"] == cat and c["label"] == "bad" for c in cases) == 4 and sum(c["cat"] == cat for c in cases) == 8
random.Random(20261031 + 3).shuffle(cases)
OUT.mkdir(exist_ok=True)
pub, truth = [], {}
for i, c in enumerate(cases):
    cid = f"case_{i:02d}"
    shutil.copyfile(CAND / c["img"], OUT / f"{cid}.jpg")
    pub.append(dict(id=cid, claim=c["claim"], image=f"cases/{cid}.jpg"))
    truth[cid] = dict(cat=c["cat"], label=c["label"], source=c["source"], key=c["key"], note=c["note"], defect=c.get("defect"))
(D / "cases.json").write_text(json.dumps(pub, indent=1))
doc = dict(_seed=20261031, _written=time.strftime("%Y-%m-%d %H:%M:%S %z"), _labeller="one labeller by eye, not a licensed reviewer; routes also checked against the IFC mask", _rule=RULE,
           _system_prompt_sha256=hashlib.sha256(prompts.SYSTEM.encode()).hexdigest(), cases=truth)
(D / "truth.json").write_text(json.dumps(doc, indent=1))
print("sha256", hashlib.sha256((D / "truth.json").read_bytes()).hexdigest(), doc["_written"])
