"""The evaluation harness on hand-built reviews, truth, sheets and PDFs. Nothing here opens data/plans or data/lab, runs the pipeline's models or reaches the network."""
import base64
import json
import shutil

import cv2
import numpy as np
import pymupdf
import pytest

from eval import baseline, plant, report, run, score
from storey import meta, pipeline
from storey.floors import ink

NUMBERS = ["G0.1", "A1.1", "A1.2", "A1.3"]
STAIR = [400.0, 300.0, 600.0, 400.0]   # sheet points, 1 px per point on the review sheet here
CATALOG = {"swiss_dwellings": {"public_demo": "yes"}, "va_trb_richmond": {"public_demo": "yes"}, "queens_royce": {"public_demo": "no"}}
SECRET = "Comment text from the examiner's record"


def refuse(kind, **request):
    raise AssertionError("A test reached the network")


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    monkeypatch.setattr(meta, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(meta, "_client", lambda: pytest.fail("a test reached the real client"))
    monkeypatch.setattr(score, "MATCHES", tmp_path / "matches")
    monkeypatch.setattr(score, "SHEETS", tmp_path / "sheets")
    monkeypatch.setattr(baseline, "OUT", tmp_path / "baseline")
    monkeypatch.setattr(run, "RUNS", tmp_path / "runs.json")
    monkeypatch.setattr(run, "FROZEN", tmp_path / "frozen.json")
    meta.set_transport(refuse)
    yield
    meta.set_transport(None)


def generated():
    return {"sheets": [{"number": n, "title": f"SHEET {n}", "kind": "plan", "floor": None} for n in NUMBERS], "px_per_pt": {"review.json": 1.0},
            "defects": [{"kind": "stair_mid", "check": 9, "severity": ["fix", "check"], "sheet": "A1.2", "box": STAIR, "wording": "The stair is left off SECOND FLOOR.", "phrase": "Stair"},
                        {"kind": "index_extra", "check": 2, "severity": ["fix"], "sheet": "G0.1", "box": [0, 0, 10, 10], "wording": "The index lists A7.1, and the set holds no such sheet.", "phrase": "A7.1"}],
            "check_11": [{"jurisdiction": "houston", "exits_required": 2, "floors": [{"floor": "SECOND FLOOR", "story": 2, "outcome": "too few stairs"}]}]}


def finding(id, check, severity, sheet, title, measured="", vertical=None):
    return {"id": id, "comment_id": id, "key": id, "check": check, "severity": severity, "sheet": sheet, "box": None, "title": title, "measured": measured, "vertical": vertical}


def review(middle="missing", findings=None):
    def cell(state):
        return {"state": state, "point": [500.0, 350.0]}
    return {"project": {"jurisdiction": "houston"}, "sheets": [{"id": str(i), "number": n, "w": 2000, "h": 1333} for i, n in enumerate(NUMBERS)],
            "verticals": [{"id": "A", "established": True, "floors": {"1": cell("found"), "2": cell(middle), "3": cell("found")}}],
            "findings": [finding("A-01", 9, "fix", "2", "Stair A is missing on SECOND FLOOR", vertical="A"), finding("A-02", 11, "check", "2", "Too few stairs on SECOND FLOOR"),
                         finding("A-03", 1, "check", None, "No site plan in the set")] if findings is None else findings, "recheck": None}


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return path


def stored(tmp_path, name, rev):
    """A finished review folder, with the record eval.run leaves after a run that began in a new folder on a cold cache."""
    folder = tmp_path / "reviews" / name
    write(folder / "review.json", rev)
    write(folder / "eval_run.json", {"fresh": True, "usd": 0.3, "minutes": 1.5})
    return folder


def truth_of(tmp_path, obj, name="gen"):
    return score.read_truth(write(tmp_path / f"truth_{name}.json", obj), name)


def judge(truth, method, matches):
    """Writes the judge's mapping for the match sheet the scorer exported."""
    sheet = json.loads((score.SHEETS / f"{truth['name']}.{method}.json").read_text())
    write(score.MATCHES / f"{truth['name']}.{method}.json", {"set": truth["name"], "method": method, "judge": "test", "sheet_sha256": sheet["sha256"],
                                                             "matches": [{"id": k, "truth": v} for k, v in matches.items()]})


def entry(name, source, storey, base=None, status="development"):
    return {"set": name, "status": status, "source": source, "Storey": storey, "baseline": base}


# The scores, in the spec's order


def test_wrong_pass_counted_first(tmp_path):
    t = truth_of(tmp_path, generated())
    s = score.storey(t, stored(tmp_path, "gen", review(middle="found", findings=[])))
    assert s["wrong passes"] == 1 and s["wrong pass list"][0]["item"] == "stair_mid"
    full, _ = report.build([entry("gen", t["source"], s)], CATALOG)
    first = next(r for r in full["scores"] if r["set"] == "gen")
    assert (first["score"], first["Storey"]) == ("wrong passes", 1)
    assert list(full).index("wrong passes") < list(full).index("misses")


def test_baseline_beside_each_score(tmp_path):
    t = truth_of(tmp_path, generated())
    s = score.storey(t, stored(tmp_path, "gen", review()))
    record = {"problems": [{"id": "B01", "sheet": "A1.2", "statement": "The stair is not drawn on the second floor."}, {"id": "B02", "sheet": "A1.1", "statement": "The plan has no north arrow."}], "usd": 0.08, "minutes": 0.5}
    assert score.baseline(t, record)["waiting"]
    judge(t, "baseline", {"B01": ["stair_mid"], "B02": []})
    b = score.baseline(t, record)
    full, _ = report.build([entry("gen", t["source"], s, b)], CATALOG)
    rows = [r for r in full["scores"] if r["set"] == "gen"]
    assert [r["score"] for r in rows] == list(report.SCORES)
    assert all(list(r)[list(r).index("Storey") + 1] == "baseline" for r in rows)
    assert {r["score"]: r["baseline"] for r in rows} == {"wrong passes": None, "findings judged real": 1, "false findings": 1, "not judged": None, "cannot tell": 0, "truth items found": 1, "cost USD": 0.08, "minutes": 0.5}
    found = next(r for r in rows if r["score"] == "truth items found")
    assert (found["out of"], found["baseline out of"]) == (4, 4)


def test_miss_listed_by_name(tmp_path):
    t = truth_of(tmp_path, generated())
    full, _ = report.build([entry("gen", t["source"], score.storey(t, stored(tmp_path, "gen", review())))], CATALOG)
    assert full["misses"] == [{"set": "gen", "method": "Storey", "item": "index_extra", "statement": "The index lists A7.1, and the set holds no such sheet.", "why": "Not raised."}]


def test_check_11_and_the_missing_site_plan_are_true_on_a_generated_set(tmp_path):
    findings = [*review()["findings"], finding("A-04", 3, "check", "1", "Floor name differs"), finding("A-05", 14, "unknown", "1", "Corridor on FIRST FLOOR")]
    s = score.storey(truth_of(tmp_path, generated()), stored(tmp_path, "gen", review(findings=findings)))
    assert (s["wrong passes"], s["findings judged real"], s["false findings"], s["cannot tell"]) == (0, 3, 1, 1)
    assert (s["truth items found"], s["out of"]) == (3, 4)   # two planted defects, check 11 on one floor and the site plan
    assert [f["id"] for f in s["false list"]] == ["A-04"]
    assert (s["cost USD"], s["minutes"]) == (0.3, 1.5)


def test_a_cannot_tell_item_is_never_judged_real_and_a_floor_that_depends_on_conditions_is_no_truth_item(tmp_path):
    truth = generated()
    truth["defects"][1]["severity"] = ["fix", "unknown"]
    truth["check_11"][0]["floors"].append({"floor": "THIRD FLOOR", "story": 3, "outcome": "one stair depends on conditions"})
    t = truth_of(tmp_path, truth)
    findings = [*review()["findings"], finding("A-04", 11, "unknown", None, "One stair here depends on conditions", "One stair on THIRD FLOOR, against 2 exits for a load of 90."),
                finding("A-05", 2, "unknown", "0", "Sheet A7.1 is on the index and not in the set")]
    s = score.storey(t, stored(tmp_path, "gen", review(findings=findings)))
    assert (s["findings judged real"], s["false findings"], s["cannot tell"], s["truth items found"], s["out of"]) == (3, 0, 2, 3, 4)
    assert [m["item"] for m in s["misses"]] == ["index_extra"] and "THIRD FLOOR" not in json.dumps(score.items(t))


def test_a_finding_counts_on_its_page_when_the_read_got_the_sheet_number_wrong(tmp_path):
    misread = review()
    misread["sheets"][2]["number"] = "A1.Z"
    s = score.storey(truth_of(tmp_path, generated()), stored(tmp_path, "gen", misread))
    assert "stair_mid" not in [m["item"] for m in s["misses"]]


def test_a_review_of_another_set_is_refused(tmp_path):
    other = review()
    other["sheets"].append({"id": "4", "number": "A9.9"})
    with pytest.raises(ValueError):
        score.storey(truth_of(tmp_path, generated()), stored(tmp_path, "gen", other))


def test_cost_and_minutes_come_only_from_a_fresh_eval_run(tmp_path):
    t = truth_of(tmp_path, generated())
    folder = stored(tmp_path, "gen", review())

    def cost():
        s = score.storey(t, folder)
        return s["cost USD"], s["minutes"]

    assert cost() == (0.3, 1.5)
    write(folder / "eval_run.json", {"fresh": False, "usd": 0.3, "minutes": 1.5})
    assert cost() == (None, None)   # resumed, or begun on a warm cache
    (folder / "eval_run.json").unlink()
    write(folder / "cost.json", {"read": {"usd": 0.01, "seconds": 30.0}})
    assert cost() == (None, None)   # made outside eval.run, on whatever cache it found


def test_looks_resolved_on_a_comment_truth_holds_open_is_a_wrong_pass():
    parent = review(findings=[finding("A-01", 2, "fix", "0", "Sheet A7.1 is on the index and not in the set"), finding("A-02", 9, "fix", "2", "Stair A is missing on SECOND FLOOR", vertical="A")])
    second = {"comments": [{"kind": "index_extra", "check": 2, "sheet": "G0.1", "state": "Still open"}, {"kind": "stair_mid", "check": 9, "sheet": "A1.2", "state": "Looks resolved"},
                           {"kind": "partition", "check": 15, "sheet": "A1.3", "state": "New change"}]}
    now = {"recheck": {"comments": [{"comment_id": "A-01", "state": "resolved_looks"}, {"comment_id": "A-02", "state": "resolved_looks"}]}}
    rows, wrong = score.comments(second, now, parent, generated())
    assert [(r["item"], r["got"]) for r in rows] == [("index_extra", ["Looks resolved"]), ("stair_mid", ["Looks resolved"])]
    assert [w["item"] for w in wrong] == ["index_extra"]


def test_a_new_change_counts_when_a_listed_change_overlaps_it_and_is_a_miss_by_name_otherwise(tmp_path, monkeypatch):
    first = write(tmp_path / "first.json", generated())
    monkeypatch.setattr(run, "truth_path", lambda name: first)
    partition = {"kind": "partition", "check": 15, "sheet": "A1.3", "box": [100.0, 200.0, 300.0, 207.0], "state": "New change", "wording": "A partition now crosses room 518, and no cloud marks it."}
    t = truth_of(tmp_path, generated() | {"parent": "first", "comments": [partition]}, "second")
    stored(tmp_path, "first", review())
    rev = review() | {"recheck": {"comments": [], "changes": [{"sheet": "3", "box": [90.0, 205.0, 150.0, 260.0], "note": "Changed since the last set, and no comment covers it."}]}}
    s = score.storey(t, stored(tmp_path, "second", rev))
    assert (s["truth items found"], s["out of"]) == (4, 5) and [m["item"] for m in s["misses"]] == ["index_extra"]
    rev["recheck"]["changes"][0]["sheet"] = "2"   # the same place on another sheet
    s = score.storey(t, stored(tmp_path, "second", rev))
    assert (s["truth items found"], s["out of"]) == (3, 5) and [m["item"] for m in s["misses"]] == ["index_extra", "partition"]
    assert "partition" not in json.dumps(score.items(t))   # the baseline is shown one set, so only a comparison could find it


# Judgement


def test_an_audit_set_waits_for_the_judge_then_reads_the_mapping(tmp_path):
    audit = {"items": [{"id": "T01", "sheets": ["A1.0"], "statement": "The index gives P6.1 the wrong title.", "status": "real"},
                       {"id": "T02", "sheets": ["A2.4"], "statement": "Two titles differ by a plural.", "status": "not_a_defect"},
                       {"id": "T03", "sheets": ["A2.3"], "statement": "Room 205 has no tag.", "status": "likely"}]}
    t = truth_of(tmp_path, audit, "cap")
    rev = review(findings=[finding("A-01", 3, "check", "0", "Sheet title differs from the index"), finding("A-02", 3, "check", "1", "Floor name differs"),
                           finding("A-03", 4, "unknown", "0", "Code field area: two reads differ")])
    folder = stored(tmp_path, "cap", rev)
    assert score.storey(t, folder)["waiting"]
    judge(t, "Storey", {"A-01": ["T01"], "A-02": ["T02"], "A-03": []})
    s = score.storey(t, folder)
    assert (s["wrong passes"], s["findings judged real"], s["false findings"], s["cannot tell"], s["truth items found"], s["out of"]) == (None, 1, 1, 1, 1, 2)
    assert s["misses"] == [{"item": "T03", "sheets": ["A2.3"], "statement": "Room 205 has no tag."}]


def test_findings_from_rules_written_after_capitol_was_audited_are_post_hoc(tmp_path):
    audit = {"items": [{"id": "T41", "sheets": ["A2.1"], "statement": "A dimension string adds to more than its overall.", "status": "real"}]}
    t = truth_of(tmp_path, audit, "mml_capitol")
    folder = stored(tmp_path, "mml_capitol", review(findings=[finding("A-01", 17, "check", "1", "Dimension string does not add up to 36'-0\"")]))
    score.storey(t, folder)
    judge(t, "Storey", {"A-01": ["T41"]})
    s = score.storey(t, folder)
    assert (s["findings judged real"], s["truth items found"]) == (0, 0)
    assert s["post hoc"] == {"findings judged real": 1, "false findings": 0, "truth items found": 1}
    assert [(m["item"], m["why"]) for m in s["misses"]] == [("T41", score.LATE)]   # found plus misses stays out of
    full, _ = report.build([entry("mml_capitol", "mml_capitol", s)], CATALOG)
    assert {r["status"] for r in full["scores"] if r["score"] == "findings judged real"} == {"development", "post hoc"}


def test_the_richmond_record_scores_which_comments_were_raised_and_judges_no_finding_false(tmp_path):
    record = {"decision_rule": "Score each sheet_only comment as raised or not.", "sheet": {"id": "A1"},
              "comments": [{"id": "C01", "text": "Show the kitchen's occupant load.", "sheet_only": True}, {"id": "C02", "text": "A permit is required.", "sheet_only": False},
                           {"id": "C03", "text": "Label each exit.", "sheet_only": True}]}
    t = truth_of(tmp_path, record, "va_trb_richmond")
    folder = stored(tmp_path, "va_trb_richmond", review(findings=[finding("A-01", 6, "check", "0", "Occupant table differs"), finding("A-02", 3, "check", "0", "Floor name differs")]))
    score.storey(t, folder)
    judge(t, "Storey", {"A-01": ["C01"], "A-02": []})
    s = score.storey(t, folder)
    assert (s["findings judged real"], s["false findings"], s["not judged"], s["truth items found"], s["out of"]) == (1, None, 1, 1, 2)
    full, _ = report.build([entry("va_trb_richmond", t["source"], s)], CATALOG)
    assert full["false findings"] == [] and "examiner's letter" in report.ABOUT


def test_a_mapping_must_answer_this_match_sheet_once_per_claim_with_known_truth_ids(tmp_path):
    sheet = score.match_sheet("cap", "Storey", [{"id": "A-01", "sheet": "A1.0", "severity": "check", "statement": "x"}, {"id": "A-02", "sheet": None, "severity": "fix", "statement": "y"}],
                              [{"id": "T01", "sheets": ["A1.0"], "statement": "z", "status": "real"}])
    path = tmp_path / "m.json"

    def mapping(**changes):
        return write(path, {"set": "cap", "method": "Storey", "judge": "test", "sheet_sha256": sheet["sha256"], "matches": [{"id": "A-01", "truth": ["T01"]}, {"id": "A-02", "truth": []}]} | changes)

    assert score.read_matches(mapping(), sheet) == {"A-01": ["T01"], "A-02": []}
    for bad in ({"sheet_sha256": "0" * 64}, {"matches": [{"id": "A-01", "truth": ["T99"]}, {"id": "A-02", "truth": []}]}, {"matches": [{"id": "A-01", "truth": ["T01"]}]},
                {"matches": [{"id": "A-01", "truth": ["T01"]}, {"id": "A-02", "truth": []}, {"id": "A-02", "truth": []}]}, {"matches": [{"id": "A-01", "truth": "T01"}, {"id": "A-02", "truth": []}]}):
        with pytest.raises(ValueError):
            score.read_matches(mapping(**bad), sheet)


# Public filter and statuses


def test_eval_json_shows_only_counts_for_sets_that_are_not_public(tmp_path):
    def scored(statement):
        return {"wrong passes": 0, "findings judged real": 1, "false findings": 1, "truth items found": 1, "out of": 2, "cost USD": 0.4, "minutes": 2.0,
                "misses": [{"item": "R1", "sheets": ["A1"], "statement": statement}], "false list": [{"id": "A-02", "sheet": "A1", "severity": "check", "statement": statement}],
                "wrong pass list": [{"item": statement, "sheet": "2", "stair": "A", "state": "Found"}]}
    entries = [entry("gen", "swiss_dwellings", scored("Shown statement")), entry("va_trb_richmond", "va_trb_richmond", scored(SECRET)), entry("queens_royce", "queens_royce", scored(SECRET + " queens"))]
    full, public = report.build(entries, CATALOG)
    shown = json.dumps(public)
    assert SECRET not in shown and "Shown statement" in shown and SECRET in json.dumps(full)
    assert {r["set"] for r in public["scores"]} == {"gen", "va_trb_richmond", "queens_royce"}
    assert all(isinstance(r[k], (int, float, type(None))) for r in public["scores"] for k in ("Storey", "baseline"))
    assert {r["set"] for k in ("misses", "false findings", "wrong passes") for r in public[k]} == {"gen"}
    for text in (shown, json.dumps(full)):
        assert "one labeller" in text and "not a licensed examiner" in text and "93%" not in text and "35 of 137" in text


def test_the_public_copy_is_not_written_while_a_set_waits(tmp_path, monkeypatch):
    waiting = entry("gen", "swiss_dwellings", {"waiting": f"A judge's mapping at {tmp_path}/matches/gen.Storey.json"})
    assert report.build([waiting], CATALOG)[1] is None
    monkeypatch.setattr(report, "collect", lambda runs: [waiting])
    monkeypatch.setattr(report, "catalog", lambda: CATALOG)
    monkeypatch.setattr(report, "REPORT", tmp_path / "report.json")
    monkeypatch.setattr(report, "PUBLIC", tmp_path / "eval.json")
    with pytest.raises(SystemExit):
        report.main(["gen=folder"])
    assert (tmp_path / "report.json").exists() and not (tmp_path / "eval.json").exists()


def test_held_out_status_needs_the_one_run_on_the_frozen_hashes_and_a_recorded_truth(monkeypatch):
    frozen = {"storey": "s", "eval": run.eval_hash(), "truth": {"army_bt_bcof": "a", "mml_capitol": "a"}, "runs": {"army_bt_bcof": "/r/army_bt_bcof"}}
    record = {"review": "/r/army_bt_bcof", "storey_sha256": "s", "truth_sha256": "a", "frozen": run.pinned(frozen, "army_bt_bcof")}
    assert report.status("army_bt_bcof", record, "a", frozen) == "frozen, held-out"
    assert report.status("mml_capitol", record | {"review": "/r/mml_capitol", "frozen": run.pinned(frozen, "mml_capitol")}, "a", frozen) == "development"
    assert report.status("army_bt_bcof", record, "b", frozen) == "post hoc"   # truth edited after the run
    assert report.status("710_1_mixed", {}, "a", frozen) == "post hoc"   # no truth recorded with the run
    assert report.status("army_bt_bcof", record | {"review": "/r2/army_bt_bcof"}, "a", frozen) == "post hoc"   # not the set's one held-out run
    assert report.status("army_bt_bcof", record, "a", frozen | {"storey": "t"}) == "post hoc"   # frozen.json edited after the run
    monkeypatch.setattr(run, "eval_hash", lambda: {"score.py": "edited"})
    assert report.status("army_bt_bcof", record, "a", frozen) == "post hoc"   # scoring edited after the run


# Running a set


def pdf(path, pages):
    doc = pymupdf.open()
    for n in pages:
        doc.new_page().insert_text((72, 72), f"packet page {n}")
    doc.save(path)
    return path


@pytest.fixture
def packet(tmp_path, monkeypatch):
    path = pdf(tmp_path / run.PACKET.name, range(1, 181))
    monkeypatch.setattr(run, "PACKET", path)
    monkeypatch.setattr(run, "RICHMOND_TRUTH", tmp_path / "no_truth.json")
    return path


@pytest.fixture
def code(tmp_path, monkeypatch):
    """A code tree of its own, so another session editing storey/ cannot change the hash halfway through a test."""
    root = tmp_path / "storey_copy"
    root.mkdir()
    (root / "pipeline.py").write_text("STAGES = []")
    monkeypatch.setattr(run, "STOREY", root)
    return root


def fake_pipeline(path, **_):
    assert pipeline.Meter().cap_usd == 1.5 and meta.CACHE == path.parent / "cache" / path.name
    costs = json.loads((path / "cost.json").read_text()) | {"read": {"usd": 0.05, "seconds": 30.0}, "verticals": {"usd": 0.1, "seconds": 90.0}}
    (path / "cost.json").write_text(json.dumps(costs))


@pytest.mark.parametrize("page", run.REFUSED)
def test_a_refused_page_of_the_richmond_packet_is_never_sent(tmp_path, packet, page):
    with pytest.raises(ValueError, match="Richmond packet"):
        run.pages(packet, [page], tmp_path / "out.pdf")
    assert not (tmp_path / "out.pdf").exists()


@pytest.mark.parametrize("pages", [[106], [108], [107, 176], [], None])
def test_page_107_alone_may_leave_the_richmond_packet(tmp_path, packet, pages):
    with pytest.raises(ValueError, match="Richmond packet"):
        run.pages(packet, pages, tmp_path / "out.pdf")


def test_the_richmond_set_reaches_the_pipeline_as_page_107_alone(tmp_path, packet, monkeypatch):
    monkeypatch.setattr(pipeline, "run", fake_pipeline)
    path = run.run(run.RICHMOND, tmp_path / "reviews", cap=1.5)
    with pymupdf.open(path / "upload" / "0000.pdf") as sent:
        assert sent.page_count == 1 and "packet page 107" in sent[0].get_text()
    assert [p.name for p in (path / "upload").iterdir()] == ["0000.pdf"]
    assert len(json.loads((path / "sheets.json").read_text())) == 1


def test_the_whole_richmond_packet_cannot_be_uploaded(tmp_path, packet, monkeypatch):
    monkeypatch.setattr(run, "inputs", lambda name: [(packet, None)])
    monkeypatch.setattr(pipeline, "run", lambda path, **_: pytest.fail("the pipeline ran"))
    with pytest.raises(ValueError, match="Richmond packet"):
        run.run("mml_capitol", tmp_path / "reviews")


def test_a_run_records_usd_and_minutes_per_stage_and_the_code_hash(tmp_path, monkeypatch, code):
    monkeypatch.setattr(run, "inputs", lambda name: [(pdf(tmp_path / "set.pdf", [1, 2]), None)])
    monkeypatch.setattr(pipeline, "run", fake_pipeline)
    path = run.run("mml_capitol", tmp_path / "reviews", cap=1.5)
    record = json.loads((path / "eval_run.json").read_text())
    assert record["stages"]["verticals"] == {"usd": 0.1, "minutes": 1.5} and record["usd"] == 0.15 and 2.0 <= record["minutes"] < 2.1 and record["fresh"] is True
    assert record["storey_sha256"] == run.code_hash(code)[0] and list(record["files"]) == ["pipeline.py"] and record["eval"] == run.eval_hash()
    assert pipeline.Meter is meta.Meter and meta.CACHE == tmp_path / "cache"   # restored after the run
    assert json.loads(run.RUNS.read_text()) == {"mml_capitol": str(path)}


def test_each_set_runs_on_a_cache_of_its_own_and_only_a_new_folder_on_a_cold_cache_is_fresh(tmp_path, monkeypatch, code):
    source = pdf(tmp_path / "set.pdf", [1])
    monkeypatch.setattr(run, "inputs", lambda name: [(source, None)])
    parents = {"second": "first"}
    monkeypatch.setattr(run, "truth_path", lambda name: write(tmp_path / f"{name}_truth.json", {"items": [], "parent": parents.get(name)}))
    caches = []

    def pipeline_run(path, **_):
        caches.append(meta.CACHE)
        meta.CACHE.mkdir(parents=True, exist_ok=True)   # the stages' model calls fill it

    monkeypatch.setattr(pipeline, "run", pipeline_run)
    data = (tmp_path / "reviews").resolve()

    def fresh(path):
        return json.loads((path / "eval_run.json").read_text())["fresh"]

    first, second, other = (run.run(name, data) for name in ("first", "second", "other"))
    assert caches == [data / "cache/first", data / "cache/first", data / "cache/other"]   # a second review shares its first review's cache, as on the server
    assert fresh(first) and fresh(second) and fresh(other)
    with pytest.raises(ValueError):
        run.run("first", data)   # a finished folder
    shutil.rmtree(first)
    assert fresh(run.run("first", data)) is False   # a new folder on the cache an earlier run filled
    (other / "eval_run.json").unlink()
    assert fresh(run.run("other", data)) is False   # resumed, so cost.json holds only the retried stages' last attempt


def test_a_held_out_set_runs_only_on_frozen_code_scoring_and_truth(tmp_path, monkeypatch, code):
    truth = write(tmp_path / "army.json", {"items": []})
    monkeypatch.setattr(run, "truth_path", lambda name: truth)
    monkeypatch.setattr(run, "inputs", lambda name: [(pdf(tmp_path / "set.pdf", [1]), None)])
    monkeypatch.setattr(pipeline, "run", lambda path, **_: pytest.fail("the pipeline ran"))
    data = tmp_path / "reviews"
    with pytest.raises(ValueError):
        run.run("army_bt_bcof", data)   # nothing frozen yet
    write(run.FROZEN, {"storey": "0" * 64, "eval": run.eval_hash(), "truth": {"army_bt_bcof": run._sha(truth)}})
    with pytest.raises(ValueError):
        run.run("army_bt_bcof", data)   # other code
    run.FROZEN.unlink()
    run.freeze()
    for path, edited in ((truth, '{"items": [], "edited": true}'), (code / "pipeline.py", "STAGES = [1]")):
        kept = path.read_text()
        path.write_text(edited)
        with pytest.raises(ValueError):
            run.run("army_bt_bcof", data)   # truth or code edited after freezing
        path.write_text(kept)
    with monkeypatch.context() as m:
        m.setattr(run, "eval_hash", lambda: {"score.py": "edited"})
        with pytest.raises(ValueError):
            run.run("army_bt_bcof", data)   # scoring edited after freezing
    assert not data.exists()   # refused before anything was read or spent
    monkeypatch.setattr(pipeline, "run", lambda path, **_: None)
    path = run.run("army_bt_bcof", data)
    record = json.loads((path / "eval_run.json").read_text())
    assert report.status("army_bt_bcof", record, run._sha(truth), json.loads(run.FROZEN.read_text())) == "frozen, held-out"


def test_a_held_out_set_runs_once_and_a_frozen_hash_never_changes(tmp_path, monkeypatch, code):
    truths = {"army_bt_bcof": write(tmp_path / "army.json", {"items": []})}
    source = pdf(tmp_path / "set.pdf", [1])
    monkeypatch.setattr(run, "truth_path", truths.get)
    monkeypatch.setattr(run, "inputs", lambda name: [(source, None)])
    monkeypatch.setattr(pipeline, "run", lambda path, **_: None)
    run.freeze()
    first = run.run("army_bt_bcof", tmp_path / "data1")
    with pytest.raises(ValueError):
        run.run("army_bt_bcof", tmp_path / "data2")   # a second held-out run, which could be the best of several
    (first / "eval_run.json").unlink()   # as if the first run had stopped part way
    assert run.run("army_bt_bcof", tmp_path / "data1") == first   # its own folder may resume
    (code / "pipeline.py").write_text("STAGES = [2]")   # changed after the held-out output was seen
    with pytest.raises(ValueError):
        run.freeze()
    (code / "pipeline.py").write_text("STAGES = []")
    with monkeypatch.context() as m:
        m.setattr(run, "eval_hash", lambda: {"score.py": "edited"})
        with pytest.raises(ValueError):
            run.freeze()
    truths["queens_royce"] = write(tmp_path / "queens.json", {"items": []})
    run.freeze()   # a truth written later may join
    frozen = json.loads(run.FROZEN.read_text())
    assert {"army_bt_bcof", "queens_royce"} <= set(frozen["truth"]) and frozen["runs"] == {"army_bt_bcof": str(first)}


def test_the_code_hash_covers_every_file_under_storey_but_compiled_caches(tmp_path):
    (tmp_path / "rules").mkdir()
    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "rules" / "rows.json").write_text("[]")
    digest, files = run.code_hash(tmp_path)
    assert set(files) == {"a.py", "rules/rows.json"}
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"compiled")
    assert run.code_hash(tmp_path)[0] == digest
    (tmp_path / "rules" / "rows.json").write_text("[1]")
    assert run.code_hash(tmp_path)[0] != digest


# Plants


def stair_sheet():
    """Walls 8 px thick and treads 2 px thin on a 2000 px sheet, the sizes the removal rule was tested on."""
    im = np.full((1400, 2000, 3), 245, np.uint8)
    cv2.rectangle(im, (200, 200), (1800, 1200), (20, 20, 20), 8)
    cv2.rectangle(im, (800, 500), (1000, 700), (20, 20, 20), 8)
    for y in range(520, 690, 12):
        cv2.line(im, (815, y), (985, y), (20, 20, 20), 2)
    return im


@pytest.fixture
def presence(tmp_path):
    sheets = tmp_path / "sheets"
    (sheets / "tb").mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(sheets / "tb" / f"{i}.jpg"), stair_sheet())
    cv2.imwrite(str(sheets / "tb" / "contact.jpg"), stair_sheet())
    box = [800, 500, 1000, 700]
    truth = {"buildings": {"tb": {"split": "dev", "ref": 0, "discharge": 0, "objects": {"S": {"kind": "stair", "cls": "main", "boxes": {str(i): box for i in range(3)}},
                                                                                         "L": {"kind": "elevator", "cls": "main", "boxes": {"0": [300, 300, 400, 400]}}}}},
             "plant": {"tb": [["middle", "S", 1], ["top", "S", 2]]}}
    return write(tmp_path / "truth.json", truth), sheets


def test_plant_erases_the_listed_stairs_on_different_floors_and_keeps_their_walls(tmp_path, presence):
    truth, sheets = presence
    folder = plant.plant("tb", truth, sheets, tmp_path / "plants")
    written = json.loads((folder / "truth.json").read_text())
    assert [(c["object"], c["floor"], c["index"], c["expected"]) for c in written["cells"]] == [("S", 1, 1, 9), ("S", 2, 2, None)]
    assert written["source"] == "tb" and written["sheets"] == ["000.png", "001.png", "002.png"] and written["floors"] == [0, 1, 2]
    assert written["stairs"] == {"S": {str(i): [800.0, 500.0, 1000.0, 700.0] for i in range(3)}}   # every stair on every floor it is drawn, so a Found where none is drawn counts too
    original = cv2.imread(str(sheets / "tb" / "0.jpg"))
    assert np.array_equal(cv2.imread(str(folder / "000.png")), original)
    for name in ("001.png", "002.png"):
        after = ink(cv2.imread(str(folder / name)))
        assert after[530:680, 830:970].sum() < 0.1 * ink(original)[530:680, 830:970].sum()   # treads gone
        assert after[590:610, 796:804].mean() > 0.5 * ink(original)[590:610, 796:804].mean()   # the stair room's wall stays
    assert all(c["ink_after"] < c["ink_before"] for c in written["cells"])
    assert cv2.imread(str(folder / "check.jpg")).shape[0] > 2 * 2 * 200   # each erasure before and after, for a person to look at


def test_plant_refuses_two_erasures_on_one_floor(tmp_path, presence):
    truth, sheets = presence
    data = json.loads(truth.read_text())
    data["plant"]["tb"] = [["middle", "S", 1], ["top", "S", 1]]
    truth.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        plant.plant("tb", truth, sheets, tmp_path / "plants")


TOP = [1400.0, 300.0, 1600.0, 400.0]
NORTH = [1000.0, 900.0, 1100.0, 1000.0]
BESIDE = [400.0, 405.0, 600.0, 445.0]   # touches the stair at STAIR, as exterior stairs touch the main ones on habs_stelizabeths_center


def planted(tmp_path):
    """S erased on floor 1 and T on floor 2. N is drawn on floor 1 alone, and E on floor 1 beside S."""
    cells = [{"object": "S", "floor": 1, "index": 1, "kind": "middle", "expected": 9, "box": STAIR}, {"object": "T", "floor": 2, "index": 2, "kind": "top", "expected": None, "box": TOP}]
    stairs = {"S": {str(i): STAIR for i in range(3)}, "T": {str(i): TOP for i in range(3)}, "N": {"1": NORTH}, "E": {"1": BESIDE}}
    return truth_of(tmp_path, {"source": "tb", "sheets": ["000.png", "001.png", "002.png"], "floors": [0, 1, 2], "stairs": stairs, "cells": cells}, "tb_plant")


def plant_review(findings):
    def row(id, point, states):
        return {"id": id, "established": True, "floors": {str(i): {"state": state, "point": point} for i, state in enumerate(states)}}
    rows = [row("A", [500, 350], ["found", "check", "found"]), row("B", [1500, 350], ["found", "found", "found"]), row("C", [1050, 950], ["end", "found", "found"]),
            row("D", [500, 425], ["missing", "found", "missing"]), row("X", [1800, 1200], ["found", "found", "found"])]
    return {"sheets": [{"id": str(i), "number": ""} for i in range(3)], "verticals": rows, "findings": findings}


def test_a_stair_shown_found_where_truth_draws_none_or_erased_it_is_a_wrong_pass(tmp_path):
    t = planted(tmp_path)
    findings = [finding("A-01", 9, "check", "1", "Stair A is not confirmed on SECOND FLOOR", vertical="A"), finding("A-02", 10, "check", "0", "Stair A stops above FIRST FLOOR", vertical="A"),
                finding("A-03", 2, "fix", "0", "Sheet A7.1 is on the index and not in the set"), finding("A-04", 14, "unknown", "1", "Corridor on SECOND FLOOR")]
    s = score.storey(t, stored(tmp_path, "tb_plant", plant_review(findings)))
    assert [(w["item"], w["stair"], w["truth"]) for w in s["wrong pass list"]] == [("T on floor 2", "B", "erased"), ("N on floor 2", "C", "not drawn on this floor")]
    assert (s["wrong passes"], s["findings judged real"], s["truth items found"], s["out of"], s["cannot tell"]) == (2, 1, 1, 2, 1)
    assert s["states"] == {"Check": 1, "Found": 1}   # D, Found on the stair beside the erased one, is not counted for it
    assert t["source"] == "tb"


def test_a_plant_judges_storey_and_the_baseline_by_one_rule(tmp_path):
    t = planted(tmp_path)
    findings = [finding("A-01", 9, "check", "1", "Stair A is not confirmed on SECOND FLOOR", vertical="A"), finding("A-02", 3, "check", "0", "Floor name differs"),
                finding("A-03", 7, "fix", "0", "Door 12 has no schedule row")]
    s = score.storey(t, stored(tmp_path, "tb_plant", plant_review(findings)))
    record = {"problems": [{"id": "B01", "sheet": "001", "statement": "The stair in the left wing is not drawn on the second floor."},
                           {"id": "B02", "sheet": "000", "statement": "The floor name in the title block differs from the drawing's label."},
                           {"id": "B03", "sheet": "000", "statement": "Door 12 has no row in the door schedule."}], "usd": 0.05, "minutes": 0.4}
    score.baseline(t, record)
    judge(t, "baseline", {"B01": ["S on floor 1"], "B02": [], "B03": []})
    b = score.baseline(t, record)
    keys = ("findings judged real", "false findings", "not judged", "truth items found", "out of")
    assert [s[k] for k in keys] == [b[k] for k in keys] == [1, None, 2, 1, 2]


# Baseline


@pytest.fixture
def sheets_folder(tmp_path):
    folder = tmp_path / "reviews" / "gen"
    folder.mkdir(parents=True)
    rows = []
    for i in range(3):
        cv2.imwrite(str(folder / f"sheet_{i:03d}.png"), np.full((2000, 3000, 3), 200 + i, np.uint8))
        rows.append({"index": i, "image": f"sheet_{i:03d}.png", "width": 3000, "height": 2000, "kind": "scan"})
    write(folder / "sheets.json", rows)
    return folder


def test_the_baseline_asks_once_with_every_sheet_at_2000_px_and_a_plain_prompt(tmp_path, sheets_folder):
    calls = []

    def fake(kind, **request):
        calls.append(request)
        return {"text": json.dumps({"problems": [{"sheet": "A1.2", "statement": "The stair is approved as drawn."}]}), "input": 9000, "output": 400}

    meta.set_transport(fake)
    meter = meta.Meter(cap_usd=0.3, ledger=tmp_path / "ledger.jsonl")
    record = baseline.ask("gen", sheets_folder, meter)
    (request,) = calls
    content = request["messages"][-1]["content"]
    images = [c["image_url"]["url"] for c in content if c["type"] == "image_url"]
    first = cv2.imdecode(np.frombuffer(base64.b64decode(images[0].split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
    assert len(images) == 3 and max(first.shape[:2]) == 2000
    assert (request["model"], request["reasoning_effort"], request["max_completion_tokens"]) == ("muse-spark-1.3", "medium", 24000)
    item = request["response_format"]["json_schema"]["schema"]["properties"]["problems"]["items"]
    assert set(item["required"]) == {"sheet", "statement"}
    prompt = content[0]["text"].lower()
    assert not any(word in prompt for word in ("stair", "door", "window", "index", "schedule", "occupan", "exit", "title", "code", "dimension", "room", "corridor", "travel", "floor", "label"))
    assert record["problems"][0]["id"] == "B01" and "approved" not in record["problems"][0]["statement"]
    assert record["usd"] > 0 and record["sheets_sent"] == 3
    assert baseline.ask("gen", sheets_folder, meter) == record and len(calls) == 1   # the same sheets are not asked again


def test_a_baseline_with_no_record_is_asked_again_when_its_reply_is_cached(tmp_path, sheets_folder):
    calls = []

    def fake(kind, **request):
        calls.append(request)
        return {"text": json.dumps({"problems": []}), "input": 9000, "output": 400}

    meta.set_transport(fake)
    meter = meta.Meter(cap_usd=0.3, ledger=tmp_path / "ledger.jsonl")
    first = baseline.ask("gen", sheets_folder, meter)
    (baseline.OUT / "gen.json").unlink()   # the record is gone and the model cache still holds the reply
    again = baseline.ask("gen", sheets_folder, meter)
    assert len(calls) == 2 and again["usd"] == first["usd"] > 0


def test_a_baseline_asked_on_other_drawings_waits(tmp_path, monkeypatch, sheets_folder):
    monkeypatch.setattr(run, "truth_path", lambda name: write(tmp_path / "gen_truth.json", generated()))
    write(sheets_folder / "review.json", review())
    record = {"problems": [], "usd": 0.05, "minutes": 0.4}
    write(baseline.OUT / "gen.json", record | {"request": "0" * 64})   # asked before the set was drawn again
    assert report.collect({"gen": str(sheets_folder)})[0]["baseline"]["waiting"]
    write(baseline.OUT / "gen.json", record | {"request": baseline.request(sheets_folder)[2]})
    assert report.collect({"gen": str(sheets_folder)})[0]["baseline"]["out of"] == 4


def test_a_baseline_that_lists_nothing_needs_no_judge_and_misses_everything(tmp_path):
    b = score.baseline(truth_of(tmp_path, generated()), {"problems": [], "usd": 0.05, "minutes": 0.4})
    assert (b["findings judged real"], b["false findings"], b["truth items found"], b["out of"]) == (0, 0, 0, 4)
    assert [m["item"] for m in b["misses"]] == ["stair_mid", "index_extra", "check 11 on SECOND FLOOR", "no site plan"]


def test_an_unreadable_baseline_reply_is_a_visible_failure(tmp_path, sheets_folder):
    meta.set_transport(lambda kind, **request: {"text": json.dumps({"list": []}), "input": 10, "output": 5})
    record = baseline.ask("gen", sheets_folder, meta.Meter(cap_usd=0.3, ledger=tmp_path / "ledger.jsonl"))
    assert record["failed"] and record["problems"] == []
    t = truth_of(tmp_path, generated())
    assert score.baseline(t, record)["failed"] == record["failed"]
