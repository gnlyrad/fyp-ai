"""Browser smoke test for the React dashboard served by scripts/serve_dashboard.py.

Run the server first (Local source, built web/dist), then:
    python evaluation/tests/browser_web_dashboard.py [http://127.0.0.1:8000]
"""
import re
import sys

from playwright.sync_api import expect, sync_playwright


def main(base_url: str) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        page.goto(f"{base_url}/local", wait_until="networkidle")
        page.get_by_role("heading", name="Datasets").wait_for()
        page.get_by_role("link", name=re.compile("Pilot")).first.click()
        page.get_by_text("Run history", exact=True).wait_for()
        expect(page.get_by_text(re.compile("SGT")).first).to_be_visible()

        # Whole run row opens the run workspace.
        page.locator("tbody tr").first.click()
        page.get_by_role("heading", name="Compliance decisioning").wait_for()
        page.get_by_role("heading", name="Structured extraction").wait_for()

        # Confusion-matrix cell -> case list -> Individual Result -> Back restores the drawer.
        page.get_by_role("button", name=re.compile("Correct pass")).click()
        drawer = page.get_by_role("dialog")
        drawer.get_by_role("link").first.click()
        page.get_by_text("Expected (ground truth)").wait_for()
        page.get_by_text("Evidence match", exact=True).wait_for()
        page.get_by_role("link", name=re.compile("^Back to")).click()
        expect(page.get_by_role("dialog")).to_be_visible()
        page.keyboard.press("Escape")

        # Analysis: row = whole call, cell = one dimension.
        page.get_by_role("tab", name="Analysis").click()
        page.get_by_role("button", name=re.compile("caught")).first.click()
        expect(page.get_by_role("dialog").get_by_text(re.compile(r"^Actual breaches"))).to_be_visible()
        page.keyboard.press("Escape")
        page.get_by_role("tab", name="By question").click()
        page.get_by_text("Compliance questions", exact=True).wait_for()
        page.get_by_text("Structured extraction", exact=True).wait_for()

        page.get_by_role("tab", name=re.compile("Diagnostics")).click()
        page.get_by_text("Evidence selection (developer metrics)").wait_for()

        assert not errors, errors
        browser.close()
        print("Dashboard smoke test passed.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000")
