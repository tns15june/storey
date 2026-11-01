"""Static contract checks on the screens. A browser run covers what the pages do, and needs no place here because tests stay off the network."""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

WEB = Path(__file__).resolve().parent.parent / "app/web"
STATUS_LINE = "Draft for examiner. Not an approval."
PAGES = ["index.html", "start.html", "eval.html"]
CODE = [*PAGES, "common.js", "app.js", "start.js", "eval.js", "style.css"]
RECHECK_STATES = ("resolved_looks", "open", "unknown", "new_change")


def text(name):
    return (WEB / name).read_text()


def ids_in(page):
    return set(re.findall(r'\bid="([\w-]+)"', text(page)))


@pytest.mark.parametrize("page", PAGES)
def test_status_line_is_on_every_page(page):
    assert STATUS_LINE in text(page)


@pytest.mark.parametrize("name", CODE)
def test_no_page_uses_a_state_of_approval(name):
    assert not re.search(r"\b(approved|approve|compliant|compliance|pass|passed|green tick)\b", text(name), re.I)


@pytest.mark.parametrize("page", PAGES)
def test_pages_work_under_the_content_policy(page):
    html = text(page)
    assert not re.search(r"<script(?![^>]*\bsrc=)", html), "inline script"
    assert not re.search(r"\son\w+=", html), "inline handler"
    assert not re.findall(r'<script[^>]+src="https?:', html), "script from another site"


@pytest.mark.parametrize("page", PAGES)
def test_pages_load_their_files_by_relative_path(page):
    """The static demo is served from a sub-path, where a leading slash would miss the site."""
    assert not re.findall(r'(?:src|href)="/', text(page))
    for name in re.findall(r'(?:src|href)="([\w.-]+\.(?:js|css|html))"', text(page)):
        assert (WEB / name).is_file(), name


def test_every_page_loads_the_shared_helpers_before_its_own_script():
    for page, script in (("index.html", "app.js"), ("start.html", "start.js"), ("eval.html", "eval.js")):
        html = text(page)
        assert html.index("common.js") < html.index(script)


@pytest.mark.parametrize("page, script", [("index.html", "app.js"), ("start.html", "start.js")])
def test_every_id_a_script_reaches_for_is_on_its_page(page, script):
    # #tab-… and #panel-… are built from a name, which is a template literal this match skips.
    wanted = set(re.findall(r"\$\('#([\w-]+)'\)", text(script)))
    assert wanted <= ids_in(page), wanted - ids_in(page)


def test_colours_are_tokens_with_a_dark_mode():
    css = text("style.css")
    rules = re.sub(r"--[\w-]+:\s*#[0-9a-f]{3,6};?", "", css)
    assert not re.findall(r"#[0-9a-f]{3,6}\b", rules), "a colour outside the tokens"
    assert "prefers-color-scheme: dark" in css and ':root[data-theme="dark"]' in css
    assert not re.findall(r"#[0-9a-f]{6}", text("app.js")), "a colour in the script"


def test_matrix_reads_each_state_and_says_so_in_the_legend():
    app, index = text("app.js"), text("index.html")
    for word in ("Found", "Missing", "Check", "Not followed", "Not on this floor"):
        assert word in app and f"</i>{word}<" in index
    assert "Seen by one run" in app


def test_second_review_names_exactly_the_recheck_states():
    states = re.search(r"const RECHECK = \{(.*?)\}", text("app.js")).group(1)
    assert tuple(re.findall(r"(\w+):", states)) == RECHECK_STATES


def test_test_tool_is_labelled_as_a_test():
    assert "Test: remove this stair from the sheet" in text("app.js")


def test_upload_pages_accept_what_the_server_accepts():
    for page in ("start.html", "index.html"):
        assert 'accept=".pdf,.png,.jpg,.jpeg"' in text(page)


def test_images_wait_for_the_session():
    """Images sit under /api/, so nothing may load before the cookie is set."""
    for script, first_load in (("app.js", "follow()"), ("start.js", "fetch('/api/reviews')")):
        body = text(script).split("async function start()")[1]
        assert body.index("await openSession()") < body.index(first_load)


def test_a_review_is_read_past_the_browser_cache():
    """Every Keep and selector rewrites review.json, and the file route sends no Cache-Control."""
    assert "request(BASE + 'review.json', { cache: 'no-cache' })" in text("app.js")


def test_session_key_leaves_the_address_bar_before_it_is_sent():
    common = text("common.js")
    assert common.index("history.replaceState") < common.index("/api/session")


@pytest.mark.parametrize("path", ["/", "/start.html", "/eval.html", "/style.css", "/common.js", "/app.js", "/start.js", "/eval.js"])
def test_screens_are_served_with_no_key(path):
    from storey import server
    response = TestClient(server.app, base_url="http://127.0.0.1").get(path)
    assert response.status_code == 200
    assert "default-src 'self'" in response.headers["content-security-policy"]
