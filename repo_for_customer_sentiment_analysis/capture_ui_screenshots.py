from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:8040"
RESULTS = Path("results")


def main():
    RESULTS.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 2200})
        page.goto(BASE_URL, wait_until="networkidle")
        page.screenshot(path=str(RESULTS / "dashboard_ui.png"), full_page=True)
        page.locator("#input-form").screenshot(path=str(RESULTS / "input_form_page.png"))

        page.fill("#feedback_text", "I am disappointed with billing errors and delayed response.")
        page.fill("#customer_segment", "Enterprise")
        page.fill("#source_channel", "Chat")
        page.fill("#confidence_score", "0.91")
        page.click("button:has-text('Analyze Sentiment')")
        page.wait_for_selector("#result-box", state="visible")

        page.locator("#result-box").screenshot(path=str(RESULTS / "prediction_result.png"))
        page.locator(".plots").screenshot(path=str(RESULTS / "generated_visualizations_ui.png"))
        browser.close()


if __name__ == "__main__":
    main()
