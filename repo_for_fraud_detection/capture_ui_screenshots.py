from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:8030"
RESULTS = Path("results")


def main():
    RESULTS.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 2200})
        page.goto(BASE_URL, wait_until="networkidle")
        page.screenshot(path=str(RESULTS / "dashboard_ui.png"), full_page=True)
        page.locator("#input-form").screenshot(path=str(RESULTS / "input_form_page.png"))

        page.fill("#transaction_amount", "4200")
        page.fill("#transaction_type", "International")
        page.fill("#customer_age", "46")
        page.fill("#account_balance", "8000")
        page.fill("#device_type", "UnknownDevice")
        page.fill("#transaction_time", "85000")
        page.fill("#merchant_category", "Jewelry")
        page.fill("#location", "Remote")
        page.fill("#payment_method", "CardNotPresent")
        page.click("button:has-text('Run Fraud Detection')")
        page.wait_for_selector("#result-box", state="visible")
        page.locator("#result-box").screenshot(path=str(RESULTS / "prediction_result.png"))
        page.locator(".plots").screenshot(path=str(RESULTS / "generated_visualizations_ui.png"))
        browser.close()


if __name__ == "__main__":
    main()
