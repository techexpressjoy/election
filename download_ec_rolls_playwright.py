import sys
# --- macOS asyncio event loop policy fix ---
if sys.platform == "darwin":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

import asyncio
from playwright.async_api import async_playwright
import pandas as pd
import os

EXCEL_FILE = "Bihar MLAs Details.xlsx"
DATA_DIR = "data/Bihar"
STATE_TO_SELECT = "Bihar"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

async def main():
    try:
        df = pd.read_excel(EXCEL_FILE)
    except FileNotFoundError:
        print(f"FATAL ERROR: The file '{EXCEL_FILE}' was not found.")
        return

    district_col, constituency_col = "District", "Constituency"
    if district_col not in df.columns or constituency_col not in df.columns:
        print(f"FATAL ERROR: Excel must have '{district_col}' and '{constituency_col}' columns.")
        return

    # Forward-fill district names to handle blank cells for constituencies under the same district
    df[district_col] = df[district_col].ffill()

    df = df[[district_col, constituency_col]].dropna().drop_duplicates()
    print(f"Found {len(df)} unique constituencies to process.")

    ensure_dir(DATA_DIR)

    # --- Ensure Playwright browsers are installed ---
    # If you see errors about missing browsers, run: playwright install

    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=False)
        except Exception as e:
            print("FATAL ERROR: Could not launch browser. Make sure you have run 'playwright install' to install browser binaries.")
            print("Error details:", e)
            return

        try:
            context = await browser.new_context(accept_downloads=True)
            page = await context.new_page()
            await page.goto("https://voters.eci.gov.in/download-eroll")
        except Exception as e:
            print("FATAL ERROR: Could not open the website. Check your internet connection and ensure the site is accessible.")
            print("Error details:", e)
            await browser.close()
            return

        print("\n--- ACTION REQUIRED ---")
        print("Solve the CAPTCHA in the browser, then press Enter here to continue...")
        input("Press Enter after solving CAPTCHA and seeing the state dropdown...")

        for index, row in df.iterrows():
            district = str(row[district_col]).strip()
            constituency = str(row[constituency_col]).strip()
            print(f"\n▶️ Processing item {index + 1}/{len(df)}: District='{district}', Constituency='{constituency}'")

            try:
                # Select State
                await page.wait_for_selector('select[name="stateCode"]', timeout=60000)
                await page.select_option('select[name="stateCode"]', label=STATE_TO_SELECT)
                await asyncio.sleep(1.5)

                # Select District
                await page.wait_for_selector('select[name="district"]', timeout=60000)
                # Wait for district dropdown to have more than one option
                for _ in range(60):
                    options = await page.eval_on_selector_all(
                        'select[name="district"] option',
                        'opts => opts.map(o => o.textContent.trim())'
                    )
                    if len(options) > 1:
                        break
                    await asyncio.sleep(1)
                else:
                    print("District dropdown did not populate in time.")
                    print("Available options:", options)
                    raise Exception("District dropdown not populated.")

                # Try to select the district using case-insensitive match
                options_map = {opt.upper(): opt for opt in options}
                district_upper = district.upper()
                if district_upper in options_map:
                    await page.select_option('select[name="district"]', label=options_map[district_upper])
                else:
                    print(f"Could not select district '{district}'. Available options: {options}")
                    raise Exception(f"District '{district}' not found in dropdown options.")
                await asyncio.sleep(1.5)

                # Select Constituency (React Select)
                await page.wait_for_selector('input[id^="react-select-2-input"]', timeout=60000)
                await page.click('input[id^="react-select-2-input"]')
                await page.fill('input[id^="react-select-2-input"]', constituency)
                await asyncio.sleep(1)
                await page.keyboard.press("Enter")
                await asyncio.sleep(1)

                # Select Language and Roll Type
                await page.wait_for_selector('select[name="langCd"]', timeout=60000)
                await page.select_option('select[name="langCd"]', label="HINDI")
                await asyncio.sleep(0.5)
                await page.wait_for_selector('select[name="roleType"]', timeout=60000)
                await page.select_option('select[name="roleType"]', label="Final Roll - 2025")
                await asyncio.sleep(0.5)

                # Prepare directory structure: data/Bihar/{District}/{Constituency}
                district_dir = os.path.join(DATA_DIR, district)
                constituency_dir = os.path.join(district_dir, constituency)
                os.makedirs(constituency_dir, exist_ok=True)

                # --- PAGINATION LOOP FOR BOOTHS ---
                page_num = 1
                while True:
                    print(f"   - Processing page {page_num} of booths...")

                    # Wait for polling stations table and select all booths
                    await page.wait_for_selector('#selectAll', timeout=60000)
                    await page.check('#selectAll')
                    await asyncio.sleep(1)

                    # Get booth names/part numbers from the table for filenames
                    booth_rows = await page.query_selector_all('table.contenttable-eroll tbody tr')
                    booth_names = []
                    for row in booth_rows:
                        cells = await row.query_selector_all('td')
                        if len(cells) >= 2:
                            booth_label = (await cells[1].inner_text()).strip()
                            booth_names.append(booth_label)
                        else:
                            booth_names.append(f"booth_{len(booth_names)+1}")

                    # --- Robust Event-Driven Download Handling ---
                    download_queue = asyncio.Queue()

                    def handle_download_event(download):
                        print(f"   - Download started: {download.suggested_filename}")
                        download_queue.put_nowait(download)

                    # Attach the listener BEFORE clicking the download button
                    page.on("download", handle_download_event)

                    # Click the button that starts all downloads
                    await page.click("button:has-text('Download Selected PDFs')")
                    print("   - Download button clicked. Waiting for files to be processed from the queue...")

                    # Process all expected downloads from the queue
                    for i in range(len(booth_names)):
                        try:
                            download = await asyncio.wait_for(download_queue.get(), timeout=60.0)
                            booth_name = booth_names[i] if i < len(booth_names) else f"booth_{i+1}"
                            safe_booth_name = "".join(c for c in booth_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                            file_path = os.path.join(constituency_dir, f"{safe_booth_name}.pdf")
                            await download.save_as(file_path)
                            print(f"   - 🚀 Download complete: {file_path}")
                        except asyncio.TimeoutError:
                            print("   - Waited 60 seconds, but no new download started. Assuming all are complete for this batch.")
                            break

                    # Remove the listener to avoid cross-batch interference
                    page.remove_listener("download", handle_download_event)

                    # --- ALWAYS PROMPT USER TO SOLVE CAPTCHA AFTER EACH PAGE ---
                    print("\n--- CAPTCHA REQUIRED ---")
                    print("Please solve the new CAPTCHA in the browser for this page.")
                    input("After solving the CAPTCHA and seeing the booths table, press Enter to continue...")

                    # --- PAGINATION: Check if ">" (next) button is enabled ---
                    control_btns = await page.query_selector_all('button.control-btn')
                    next_btn = None
                    for btn in control_btns:
                        btn_text = (await btn.inner_text()).strip()
                        if btn_text == ">":
                            next_btn = btn
                            break
                    if next_btn:
                        is_disabled = await next_btn.get_attribute("disabled")
                        if is_disabled is not None:
                            print("   - Last page reached for this constituency.")
                            break
                        else:
                            await next_btn.click()
                            await asyncio.sleep(2)
                            page_num += 1
                    else:
                        print("   - Next page (>) button not found, assuming last page.")
                        break

                # After all pages for this constituency, refresh for next item
                await page.reload()
                input("=> Page refreshed for the next item. Solve the new CAPTCHA, wait for states to load, then press Enter...")

            except Exception as e:
                print(f"   ❌ An unexpected error occurred: {e}")
                await page.reload()
                input("=> Page refreshed due to error. Solve the CAPTCHA, wait for states, then press Enter...")
                continue

        print("\n\n🎉 All items have been processed.")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
