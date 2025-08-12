import os
import pandas as pd
import tempfile
from pdf2image import convert_from_path
import google.generativeai as genai
import concurrent.futures
import time
import PyPDF2
import re

# Import consolidated modules
from election_modules import EnhancedParser, Config, processing_logger, encode_image_to_base64

# Initialize the enhanced parser
parser = EnhancedParser('english')


def process_page_parallel(args):
    """Process a single page for parallel execution"""
    temp_dir, page_image, page_num = args

    # Save the image temporarily
    image_path = os.path.join(temp_dir, f"page_{page_num}.jpg")
    page_image.save(image_path, 'JPEG', quality=Config.IMAGE_QUALITY)

    # Process the page with enhanced parser
    voter_entries = parser.parse_page(image_path, page_num, expected_entries=30)

    return voter_entries

def process_electoral_roll(pdf_path, output_csv, start_page=Config.DEFAULT_START_PAGE, end_page=None, check_total=True, max_parallel_workers=Config.DEFAULT_MAX_PARALLEL_WORKERS):
    """Process electoral roll PDF using enhanced parser and convert to CSV"""
    processing_logger.info(f"Processing {pdf_path}")

    # Initialize list to store voter data
    all_voter_entries = []

    # Get total number of pages in the PDF
    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total_pdf_pages = len(pdf_reader.pages)

    # If end_page is not specified, use the second-to-last page
    # This ensures we don't process the last page for voter data, only for verification
    if end_page is None:
        end_page = total_pdf_pages - 1

    # Ensure end_page doesn't exceed the second-to-last page
    end_page = min(end_page, total_pdf_pages - 1)

    # Calculate the number of pages to process
    num_pages = end_page - start_page + 1

    # Always process the last page separately for verification
    process_last_page = check_total

    # Create a temporary directory to store the images
    with tempfile.TemporaryDirectory() as temp_dir:
        processing_logger.info(f"Converting PDF pages {start_page} to {end_page} to images...")

        # Convert pages to images (excluding the last page of the PDF)
        pages = convert_from_path(pdf_path, dpi=300, first_page=start_page, last_page=end_page)
        processing_logger.info(f"Converted {len(pages)} pages to images for voter data extraction")

        # Always convert the last page separately for verification of total voters
        last_page_image = None
        if process_last_page:
            processing_logger.info(f"Also converting last page (page {total_pdf_pages}) to check total voters...")
            last_page_images = convert_from_path(pdf_path, dpi=300, first_page=total_pdf_pages, last_page=total_pdf_pages)
            if last_page_images:
                last_page_image = last_page_images[0]

        # Process pages in parallel
        processing_logger.info(f"Processing {len(pages)} pages in parallel...")
        start_time = time.time()

        # Prepare arguments for parallel processing
        process_args = [
            (temp_dir, page_image, start_page + i)
            for i, page_image in enumerate(pages)
        ]

        # Use ThreadPoolExecutor for parallel processing
        # Adjust max_workers based on parameter and number of pages
        max_workers = min(max_parallel_workers, len(pages))  # Use up to max_parallel_workers, or fewer if there are fewer pages
        processing_logger.info(f"Using {max_workers} parallel workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and collect results as they complete
            future_to_page = {executor.submit(process_page_parallel, args): args[2] for args in process_args}

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    voter_entries = future.result()
                    all_voter_entries.extend(voter_entries)
                    processing_logger.info(f"Completed processing page {page_num}/{start_page+len(pages)-1}")
                except Exception as e:
                    processing_logger.error(f"Error processing page {page_num}: {e}")

        end_time = time.time()
        processing_time = end_time - start_time
        processing_logger.info(f"Parallel processing completed in {processing_time:.2f} seconds")
        processing_logger.info(f"Average time per page: {processing_time/len(pages):.2f} seconds")

        # Process the last page to check total number of voters if needed
        total_voters_from_pdf = None
        if process_last_page and last_page_image:
            processing_logger.info("Processing last page to check total number of voters...")
            # Save the last page image temporarily
            last_page_path = os.path.join(temp_dir, "last_page.jpg")
            last_page_image.save(last_page_path, 'JPEG', quality=95)

            # Create a Gemini model for the last page
            model = genai.GenerativeModel(Config.GEMINI_MODEL)

            # Prepare a specific prompt for extracting the total number of voters
            prompt = """
            Look at this last page of an electoral roll document and find the total number of voters/electors.
            The total number is usually mentioned in a section like 'SUMMARY OF ELECTORS' or 'NUMBER OF ELECTORS'.
            Return ONLY the number, nothing else.
            """

            # Encode the image to base64
            base64_image = encode_image_to_base64(last_page_path)

            try:
                # Generate content with Gemini
                response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": base64_image}])
                response_text = response.text.strip()

                # Try to extract a number from the response
                number_match = re.search(r'\d+', response_text)
                if number_match:
                    total_voters_from_pdf = int(number_match.group(0))
                    processing_logger.info(f"Found total voters in PDF: {total_voters_from_pdf}")
            except Exception as e:
                processing_logger.error(f"Error processing last page: {e}")

    # Convert to DataFrame
    if all_voter_entries:
        # Create DataFrame from the voter entries
        df = pd.DataFrame(all_voter_entries)

        # Ensure all required columns exist
        required_columns = ['serial_no', 'name', 'father_husband', 'age', 'gender', 'eid']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ""

        # Reorder columns
        df = df[required_columns]

        # Rename columns to match requirements
        df = df.rename(columns={
            'serial_no': 'Serial No',
            'name': 'Name',
            'father_husband': 'Father/Husband Name',
            'age': 'Age',
            'gender': 'Gender',
            'eid': 'EID'
        })

        # Clean the data
        df = df.replace('', None)
        df = df.dropna(how='all')  # Remove rows that are all NaN

        # Save to CSV
        df.to_csv(output_csv, index=False)
        processing_logger.info(f"Data saved to {output_csv} with {len(df)} records")

        # Check if the number of records matches the total from the PDF
        if total_voters_from_pdf:
            if len(df) == total_voters_from_pdf:
                processing_logger.info(f"✅ SUCCESS: Number of extracted records ({len(df)}) matches the total in the PDF ({total_voters_from_pdf})")
            else:
                processing_logger.warning(f"⚠️ WARNING: Number of extracted records ({len(df)}) does not match the total in the PDF ({total_voters_from_pdf})")
                if len(df) < total_voters_from_pdf:
                    missing = total_voters_from_pdf - len(df)
                    processing_logger.warning(f"Missing {missing} records. Consider processing more pages.")

        # Check if we processed all pages
        if start_page + num_pages - 1 < total_pdf_pages:
            processing_logger.info("Note: Only processed a subset of pages. To process all pages, increase the num_pages parameter.")
    else:
        processing_logger.warning("No voter data extracted. Creating empty CSV with headers.")
        pd.DataFrame(columns=['Serial No', 'Name', 'Father/Husband Name', 'Age', 'Gender', 'EID']).to_csv(output_csv, index=False)
        processing_logger.info(f"Empty CSV file created at {output_csv}")

if __name__ == "__main__":
    import sys

    # Input and output paths
    pdf_path = "/home/lt-340/Desktop/Election/2025-EROLLGEN-U05-2-DraftRoll-Revision1-ENG-1-WI.pdf"
    output_csv = "/home/lt-340/Desktop/Election/voter_data_gemini_U05.csv"

    # Get total number of pages in the PDF
    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total_pdf_pages = len(pdf_reader.pages)

    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        start_page = int(sys.argv[1]) if len(sys.argv) > 1 else Config.DEFAULT_START_PAGE
        end_page = int(sys.argv[2]) if len(sys.argv) > 2 else total_pdf_pages - 1
        max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else Config.DEFAULT_MAX_PARALLEL_WORKERS
    else:
        start_page = Config.DEFAULT_START_PAGE
        end_page = total_pdf_pages - 1
        max_workers = Config.DEFAULT_MAX_PARALLEL_WORKERS

    print(f"PDF has {total_pdf_pages} total pages")
    print(f"Processing pages {start_page} to {end_page} with {max_workers} parallel workers")
    print(f"The last page (page {total_pdf_pages}) will be used only for verification of total voters")
    print(f"Note: Using 4 workers as requested. This may occasionally hit API rate limits, but the script will retry automatically.")

    # Process the electoral roll
    process_electoral_roll(pdf_path, output_csv, start_page=start_page, end_page=end_page,
                          check_total=True, max_parallel_workers=max_workers)

    print("\nUsage:")
    print("  python election_english.py [start_page] [end_page] [max_workers]")
    print("\nExamples:")
    print("  python election_english.py                # Process all pages with 4 workers (default)")
    print("  python election_english.py 3 7           # Process pages 3-7 with 4 workers")
    print("  python election_english.py 3 7 5         # Process pages 3-7 with 5 workers")
    print("  python election_english.py 8 17 2        # Process pages 8-17 with 2 workers")
    print(f"  python election_english.py 3 {total_pdf_pages-1} 4  # Process all pages from 3 to {total_pdf_pages-1} with 4 workers")
    print("\nNote: Using more than 4-5 workers may cause API rate limit errors.")
    print("The last page is used only for verification of total voters.")
