"""Browser test of the full stack: the real models, LM Studio, the API, and the Vite client in Chrome.

It is slow and needs LM Studio running, so it runs only with GRAPEVINE_E2E=1.
"""

import os
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from grapevine.config import ROOT

pytestmark = pytest.mark.skipif(os.environ.get("GRAPEVINE_E2E") != "1", reason="set GRAPEVINE_E2E=1")

API_PORT = 8791
WEB_PORT = 5174


def wait_for(url: str, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            httpx.get(url, timeout=1)
            return
        except httpx.HTTPError:
            time.sleep(0.5)
    raise TimeoutError(url)


def free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("localhost", port)) != 0


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    assert free(API_PORT) and free(WEB_PORT), "the e2e ports are taken"
    env = {**os.environ, "GRAPEVINE_DB": str(tmp_path_factory.mktemp("e2e") / "e2e.db"), "PORT": str(API_PORT)}
    api = subprocess.Popen(["uv", "run", "grapevine", "serve"], cwd=ROOT / "backend", env=env)
    web = subprocess.Popen(["npx", "vite", "--port", str(WEB_PORT), "--strictPort"], cwd=ROOT, env=env)
    try:
        wait_for(f"http://localhost:{API_PORT}/api/sessions", 60)
        wait_for(f"http://localhost:{WEB_PORT}/", 60)
        yield f"http://localhost:{WEB_PORT}"
    finally:
        api.terminate()
        web.terminate()


@pytest.fixture
def page(app):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1512, "height": 945})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        yield page
        browser.close()
        assert errors == []


def shoot(page, name: str) -> None:
    out = Path(os.environ.get("GRAPEVINE_SHOTS", "/tmp/grapevine-e2e"))
    out.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(out / f"{name}.png"))


def test_replay_of_the_sample_builds_the_graph_and_takes_answers(app, page):
    page.goto(app)
    page.get_by_role("button", name="New session").click()
    page.get_by_role("button", name="kickback-3p.mp3").click()
    page.get_by_role("button", name="Replay fast").click()
    page.get_by_test_id("run-status").filter(has_text="done").wait_for(timeout=300_000)
    shoot(page, "1-graph")

    assert page.locator("g.node").count() >= 7
    assert page.locator("g.edges path").count() >= 5

    which = page.get_by_test_id("prompt").filter(has_text="Which Andrew")
    which.wait_for()
    facts_before = int(page.get_by_test_id("stats").locator("span.font-semibold").nth(2).inner_text())
    which.get_by_role("button", name="Andrew Gao").click()
    which.wait_for(state="detached")
    page.wait_for_function(
        f"() => Number(document.querySelectorAll('[data-testid=stats] span.font-semibold')[2].textContent) > {facts_before}"
    )
    shoot(page, "2-answered")

    page.locator("g.node").first.click()
    page.get_by_test_id("person-details").wait_for()
    page.keyboard.press("Escape")

    page.get_by_role("radio", name="Timeline").click()
    page.get_by_test_id("timeline-event").first.wait_for()
    assert page.get_by_test_id("story").count() >= 3
    shoot(page, "3-timeline")

    run_id = page.get_by_test_id("run-status").inner_text().split()[1]
    page.goto(f"{app}/#/debug?run={run_id}")
    page.get_by_role("tab", name="Model calls").wait_for()
    page.get_by_test_id("transcript").locator("li").first.wait_for()
    assert page.get_by_test_id("transcript").locator("li").count() == 18
    page.get_by_role("tab", name="Model calls").click()
    assert page.get_by_test_id("model-calls").locator("li").count() >= 18
    shoot(page, "4-debug")
