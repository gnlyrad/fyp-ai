"""Browser smoke test for the Local Streamlit dashboard."""
from playwright.sync_api import sync_playwright


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto("http://127.0.0.1:8501", wait_until="networkidle")
        page.get_by_text("Local", exact=True).click()
        page.get_by_role("heading", name="Evaluation datasets").wait_for()
        page.get_by_role("button", name="Open dataset").click()
        page.get_by_role("heading", name="Pilot", exact=True).wait_for()
        page.get_by_text("Run history", exact=True).wait_for()
        page.get_by_role("button", name="Open run").click()
        page.get_by_role("tab", name="Compliance").wait_for()
        page.get_by_role("tab", name="Structured Extraction").wait_for()
        page.get_by_role("tab", name="Compliance").click()
        page.get_by_role("heading", name="1. Decision Capability", exact=True).wait_for()
        page.get_by_role("heading", name="3. End-to-End Actual Breach Handling", exact=True).wait_for()
        page.get_by_role("tab", name="By Call").click()
        page.get_by_role("button", name="View call").click()
        page.get_by_role("heading", name="CALL_001", exact=True).wait_for()
        page.get_by_text("Results in this call", exact=True).wait_for()
        page.get_by_role("button", name="View individual result").click()
        page.get_by_role("heading", name="CALL_001 × C01", exact=True).wait_for()
        page.get_by_text("Ground truth", exact=True).wait_for()
        page.get_by_text("Pipeline result", exact=True).wait_for()
        page.get_by_text("Transcript evidence", exact=True).wait_for()
        page.get_by_role("button", name="20261002T095827Z-ebce4131").click()
        page.get_by_role("tab", name="By Question").click()
        page.get_by_role("button", name="View question").click()
        page.get_by_role("heading", name="C01 — The employee must confirm the client's investment objective.", exact=True).wait_for()
        page.get_by_text("Results by call", exact=True).wait_for()
        page.get_by_role("button", name="View individual result").click()
        page.get_by_role("heading", name="CALL_001 × C01", exact=True).wait_for()
        assert not page.locator("text=Uncaught app exception").count()
        browser.close()


if __name__ == "__main__":
    main()
