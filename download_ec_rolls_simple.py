import sys
import os
import asyncio
from pathlib import Path
from zipfile import ZipFile
from playwright.async_api import async_playwright
import pandas as pd

TARGET_URL = "https://www.eci.gov.in/eci-backend/public/ER/s04/SIR/roll.html"
DOWNLOAD_DIR = "eci_downloads"
EXCEL_FILE = "Bihar MLAs Details.xlsx"

async def main():
    # macOS event loop fix
    if sys.platform == "darwin":
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Read required constituencies from Excel
    try:
        df = pd.read_excel(EXCEL_FILE)
        district_col = "District"
        constituency_col = "Constituency"
        for col in [district_col, constituency_col]:
            if col not in df.columns:
                print(f"FATAL ERROR: Excel must have a '{col}' column.")
                return
        # Forward-fill district names to handle blank cells
        df[district_col] = df[district_col].ffill()

        # Build a mapping: normalized constituency name -> (district, constituency)
        constituency_map = {}
        for _, row in df.iterrows():
            district_val = row[district_col]
            constituency_val = row[constituency_col]
            if pd.isna(district_val) or pd.isna(constituency_val):
                continue
            district = str(district_val).strip()
            constituency = str(constituency_val).strip()
            if not district or not constituency or district.lower() == "nan":
                continue
            constituency_map[constituency.upper()] = (district, constituency)
        print(f"Loaded {len(constituency_map)} constituencies from Excel.")
    except Exception as e:
        print(f"FATAL ERROR: Could not read Excel file '{EXCEL_FILE}': {e}")
        return

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()
        await page.goto(TARGET_URL)

        print("Page loaded. Locating all assembly rows...")

        # Wait for the table to load
        await page.wait_for_selector("table")
        rows = await page.query_selector_all("table tbody tr")
        print(f"Found {len(rows)} rows.")

        for idx, row in enumerate(rows):
            try:
                # Get constituency name
                cells = await row.query_selector_all("td")
                if len(cells) < 2:
                    print(f"Row {idx+1}: Skipped (not enough columns)")
                    continue
                constituency = (await cells[1].inner_text()).strip()
                constituency_norm = constituency.upper()
                if constituency_norm not in constituency_map:
                    print(f"[{idx+1}/{len(rows)}] Skipping: {constituency} (not in Excel list)")
                    continue
                district, constituency_actual = constituency_map[constituency_norm]
                print(f"\n[{idx+1}/{len(rows)}] Processing: {district} / {constituency_actual}")

                # Find the download button (assume it's the last cell)
                download_btn = await cells[-1].query_selector("a, button")
                if not download_btn:
                    print(f"  No download button found for {constituency_actual}. Skipping.")
                    continue

                # Start download
                print("  Clicking download button...")
                async with page.expect_download() as download_info:
                    await download_btn.click()
                download = await download_info.value

                # Save ZIP file in District/Constituency
                constituency_dir = Path(DOWNLOAD_DIR) / district / constituency_actual
                constituency_dir.mkdir(parents=True, exist_ok=True)
                zip_path = constituency_dir / f"{constituency_actual}.zip"
                await download.save_as(str(zip_path))
                print(f"  Downloaded ZIP to {zip_path}")

                # Extract ZIP (flatten structure: extract only PDFs to constituency_dir)
                try:
                    with ZipFile(zip_path, 'r') as zip_ref:
                        for member in zip_ref.namelist():
                            if member.lower().endswith('.pdf') and not member.endswith('/'):
                                # Extract PDF directly to constituency_dir
                                source = zip_ref.open(member)
                                target_path = constituency_dir / Path(member).name
                                with open(target_path, "wb") as target:
                                    target.write(source.read())
                    print(f"  Extracted PDFs to {constituency_dir}")
                except Exception as e:
                    print(f"  Failed to extract ZIP for {constituency_actual}: {e}")
                    continue

                # Optionally, remove the ZIP after extraction
                os.remove(zip_path)

            except Exception as e:
                print(f"  Error processing row {idx+1}: {e}")

        print("\nAll downloads and extractions complete.")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
