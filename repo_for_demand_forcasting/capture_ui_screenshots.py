from pathlib import Path

from playwright.sync_api import sync_playwright


BASE_URL = "http://127.0.0.1:8010"
RESULTS = Path("results")


def main():
    RESULTS.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 2200})
        page.goto(BASE_URL, wait_until="networkidle")

        page.screenshot(path=str(RESULTS / "dashboard_ui.png"), full_page=True)
        form = page.locator("#input-form")
        form.screenshot(path=str(RESULTS / "input_form_page.png"))

        page.fill("#product_category", "Seasonal Essentials")
        page.fill("#historical_sales", "165")
        page.fill("#inventory_level", "145")
        page.fill("#seasonal_index", "1.25")
        page.fill("#marketing_spend", "80")
        page.fill("#region", "West")
        page.fill("#month", "11")
        page.fill("#promotional_offers", "1")
        page.fill("#customer_demand_trends", "0.92")
        page.click("button:has-text('Forecast Product Demand')")
        page.wait_for_selector("#result-box", state="visible")

        page.locator("#result-box").screenshot(path=str(RESULTS / "forecasting_result.png"))
        page.locator(".plots").screenshot(path=str(RESULTS / "generated_visualizations_ui.png"))
        browser.close()


if __name__ == "__main__":
    main()
