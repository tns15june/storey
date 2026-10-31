import json

import cv2
import pymupdf
import numpy as np
import pytest

from storey.ingest import ingest


def _pdf(path):
    doc = pymupdf.open()
    page = doc.new_page(width=800, height=600)
    page.insert_text((50, 60), "A-101 LEVEL 1 FLOOR PLAN")
    shape = page.new_shape()
    for i in range(250):
        shape.draw_line((50, 80 + i), (700, 80 + i))
    shape.finish(color=(0, 0, 0), width=.2)
    shape.commit()
    doc.new_page(width=800, height=600)
    doc.save(path)


def test_pdf_pages_become_sheets_with_text_and_kind(tmp_path):
    src = tmp_path / "set.pdf"
    _pdf(src)
    sheets = ingest([src], tmp_path / "out")
    assert [s.kind for s in sheets] == ["vector", "scan"]
    assert "A-101" in sheets[0].text
    assert (tmp_path / "out" / sheets[0].image).exists()
    assert json.loads((tmp_path / "out" / "sheets.json").read_text())[0]["width"] == sheets[0].width
    word = next(w for w in sheets[0].words if w[4] == "A-101")
    assert 0 <= word[0] < word[2] <= sheets[0].width


def test_images_sort_by_name(tmp_path):
    for name, v in (("b.png", 200), ("a.png", 100)):
        cv2.imwrite(str(tmp_path / name), np.full((40, 60, 3), v, np.uint8))
    sheets = ingest([tmp_path / "b.png", tmp_path / "a.png"], tmp_path / "out")
    assert cv2.imread(str(tmp_path / "out" / sheets[0].image))[0, 0, 0] == 100


def test_rejects_unreadable_and_mixed_uploads(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_text("not an image")
    with pytest.raises(ValueError, match="bad.png"):
        ingest([bad], tmp_path / "out")
    pdf = tmp_path / "x.pdf"
    _pdf(pdf)
    with pytest.raises(ValueError, match="not both"):
        ingest([pdf, bad], tmp_path / "out")


def test_rejects_too_many_pages_and_locked_pdfs(tmp_path, monkeypatch):
    from storey import ingest as mod
    monkeypatch.setattr(mod, "MAX_SHEETS", 1)
    src = tmp_path / "set.pdf"
    _pdf(src)
    with pytest.raises(ValueError, match="limit"):
        ingest([src], tmp_path / "out")
    monkeypatch.setattr(mod, "MAX_SHEETS", 10)
    doc = pymupdf.open()
    doc.new_page()
    doc.save(tmp_path / "locked.pdf", encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="x", owner_pw="y")
    with pytest.raises(ValueError, match="password"):
        ingest([tmp_path / "locked.pdf"], tmp_path / "out")


def test_large_image_is_rejected_from_its_header(tmp_path, monkeypatch):
    from storey import ingest as mod
    monkeypatch.setattr(mod, "MAX_PIXELS", 100)
    cv2.imwrite(str(tmp_path / "big.png"), np.zeros((40, 60, 3), np.uint8))
    with pytest.raises(ValueError, match="too large"):
        ingest([tmp_path / "big.png"], tmp_path / "out")


def test_wide_image_is_scaled_to_the_edge_limit(tmp_path, monkeypatch):
    from storey import ingest as mod
    monkeypatch.setattr(mod, "MAX_EDGE", 30)
    cv2.imwrite(str(tmp_path / "w.png"), np.zeros((40, 60, 3), np.uint8))
    assert ingest([tmp_path / "w.png"], tmp_path / "out")[0].width == 30
