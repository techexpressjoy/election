"""
Consolidated Election Roll Processing Modules
All shared functionality in one file for easier management
"""

import base64
import json
import os
import re
import time
import logging
import sys
from typing import List, Dict, Optional
from datetime import datetime
import google.generativeai as genai
from PIL import Image, ImageEnhance, ImageFilter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # API Configuration
    GEMINI_MODEL = 'gemini-1.5-flash'
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2

    # Processing Configuration
    DEFAULT_START_PAGE = 3
    DEFAULT_MAX_PARALLEL_WORKERS = 8
    IMAGE_QUALITY = 85
    OCR_IMAGE_QUALITY = 95

    # Validation Configuration
    MIN_AGE = 18
    MAX_AGE = 120
    VALID_GENDERS = ['Male', 'Female']

    # EID Patterns
    EID_PATTERNS = [
        r'^[A-Z0-9]{10}$',
        r'^BR/\d{2}/\d{3}/\d{6}$'
    ]

    # Required fields for validation
    REQUIRED_FIELDS = ['serial_no', 'name', 'eid']
    REQUIRED_FIELDS_HINDI = ['serial_no', 'name_hindi', 'eid']

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logger(name, level=logging.INFO):
    """Set up logger with console handler"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Create logger
processing_logger = setup_logger('processing')

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def encode_image_to_base64(image_path):
    """Convert an image to base64 encoding for API transmission"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def retry_with_backoff(func, max_retries=3, base_delay=2, *args, **kwargs):
    """Generic retry mechanism with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            processing_logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {error_msg}")
            
            if "429" in error_msg and "quota" in error_msg and attempt < max_retries - 1:
                wait_time = base_delay * (2 ** attempt)
                processing_logger.info(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                time.sleep(base_delay)
            else:
                raise e
    
    raise Exception(f"All {max_retries} attempts failed")

def validate_voter_entry(entry, page_num, language='english'):
    """Validate a voter entry for required fields and format"""
    required_fields = Config.REQUIRED_FIELDS_HINDI if language == 'hindi' else Config.REQUIRED_FIELDS
    missing_fields = [field for field in required_fields if field not in entry or not entry[field]]

    if missing_fields:
        processing_logger.warning(f"Page {page_num}: Entry missing required fields: {', '.join(missing_fields)}")
        return False

    # EID validation
    if 'eid' in entry and entry['eid']:
        eid = entry['eid']
        if any(re.match(pattern, eid) for pattern in Config.EID_PATTERNS):
            entry['eid_validated'] = True
        else:
            processing_logger.warning(f"Page {page_num}: Invalid EID format: {eid}")
            entry['eid_validated'] = False

    # Age validation
    if 'age' in entry and entry['age']:
        try:
            age = int(entry['age'])
            if not (Config.MIN_AGE <= age <= Config.MAX_AGE):
                processing_logger.warning(f"Page {page_num}: Invalid age: {age}")
                entry['age_validated'] = False
            else:
                entry['age_validated'] = True
        except ValueError:
            processing_logger.warning(f"Page {page_num}: Non-numeric age: {entry['age']}")
            entry['age_validated'] = False

    # Gender validation
    if 'gender' in entry and entry['gender']:
        if entry['gender'] not in Config.VALID_GENDERS:
            processing_logger.warning(f"Page {page_num}: Invalid gender: {entry['gender']}")
            entry['gender_validated'] = False
        else:
            entry['gender_validated'] = True

    return True

def split_name(name):
    """Split full name into components"""
    parts = name.strip().split()
    result = {}

    if len(parts) == 1:
        result["first_name"] = parts[0]
        result["middle_name"] = ""
        result["last_name"] = ""
    elif len(parts) == 2:
        result["first_name"] = parts[0]
        result["middle_name"] = ""
        result["last_name"] = parts[1]
    else:
        result["first_name"] = parts[0]
        result["middle_name"] = " ".join(parts[1:-1])
        result["last_name"] = parts[-1]

    return result

def transliterate_hindi_to_english(hindi_text, max_retries=3, retry_delay=2):
    """Transliterate Hindi text to English using Gemini API"""
    if not hindi_text or hindi_text.strip() == "":
        return ""

    try:
        model = genai.GenerativeModel(Config.GEMINI_MODEL)
        prompt = f"""
        Transliterate the following Hindi name to English, preserving the pronunciation as closely as possible.
        Only return the transliterated name, nothing else.

        Hindi name: {hindi_text}
        """

        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                transliterated_text = response.text.strip()
                transliterated_text = transliterated_text.strip('"\'\'\n ')
                processing_logger.info(f"Transliterated: '{hindi_text}' → '{transliterated_text}'")
                return transliterated_text

            except Exception as e:
                error_msg = str(e)
                processing_logger.warning(f"Attempt {attempt+1}/{max_retries} failed for transliteration: {error_msg}")

                if "429" in error_msg and "quota" in error_msg and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    processing_logger.info(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    break

        processing_logger.warning(f"All transliteration attempts failed for: {hindi_text}")
        return hindi_text

    except Exception as e:
        processing_logger.error(f"Error in transliteration: {e}")
        return hindi_text

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

class PromptTemplates:
    
    @staticmethod
    def get_base_extraction_prompt():
        return """
You are a highly precise electoral data extraction system. Your task is to extract voter information from this electoral roll page with maximum accuracy.

CRITICAL EXTRACTION RULES:
1. Extract EVERY voter entry on the page - count them to ensure none are missed
2. SKIP entries marked as "DELETED", "निरस्त", or with strikethrough
3. Look carefully at edges and corners for partially visible entries
4. If text is unclear, use "[unclear]" rather than guessing
5. Verify each entry has minimum required fields: serial_no, name, eid

QUALITY CHECKS:
- Double-check each entry after extraction
- Count total entries to match expected page count
- Validate EID format consistency
"""

    @staticmethod
    def get_english_prompt():
        base = PromptTemplates.get_base_extraction_prompt()
        return f"""{base}

For each voter entry, extract:
1. Serial Number - Left side of each entry
2. Name - Full name as written
3. Father's/Husband's/Mother's name
4. Age - Numeric value only
5. Gender - "Male" or "Female" only
6. Electoral ID (EID) - Alphanumeric code (e.g., AGZ1234567)
7. House Number - If available

JSON Format:
[
    {{
        "serial_no": "1",
        "name": "John Smith",
        "father_husband": "Michael Smith",
        "age": "35",
        "gender": "Male",
        "eid": "AGZ1234567",
        "house_no": "123"
    }}
]
"""

    @staticmethod
    def get_hindi_prompt():
        base = PromptTemplates.get_base_extraction_prompt()
        return f"""{base}

For each voter entry, extract:
1. Serial Number - Left side of each entry
2. Name in Hindi - Exactly as written with all diacritical marks
3. Name in English - Phonetically accurate transliteration
4. Father's/Husband's/Mother's name in Hindi
5. Age - Numeric value only
6. Gender - "Male" or "Female" only
7. Electoral ID (EID) - Alphanumeric code (e.g., AZA1234567)
8. House Number - If available

TRANSLITERATION RULES:
- Use standard Hindi-to-English conventions
- Preserve pronunciation accuracy
- Handle compound names properly

JSON Format:
[
    {{
        "serial_no": "61",
        "name_hindi": "राम कुमार शर्मा",
        "name_english": "Ram Kumar Sharma",
        "father_husband_hindi": "श्याम लाल शर्मा",
        "age": "35",
        "gender": "Male",
        "eid": "AZA1234567",
        "house_no": "123"
    }}
]
"""

# =============================================================================
# IMAGE PROCESSING
# =============================================================================

class ImageProcessor:
    
    @staticmethod
    def preprocess_for_ocr(image_path, output_path=None):
        """Apply preprocessing techniques for better OCR"""
        # Read image
        pil_img = Image.open(image_path)
        
        # Apply preprocessing pipeline
        processed_img = ImageProcessor._apply_preprocessing_pipeline(pil_img)
        
        # Save if output path provided
        if output_path:
            processed_img.save(output_path, 'JPEG', quality=95, optimize=True)
        
        return processed_img
    
    @staticmethod
    def _apply_preprocessing_pipeline(pil_img):
        """Apply series of image enhancements"""
        # 1. Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_img)
        img = enhancer.enhance(1.2)
        
        # 2. Enhance sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.1)
        
        # 3. Convert to grayscale for better text recognition
        img = img.convert('L')
        
        # 4. Apply slight gaussian blur to reduce noise
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # 5. Convert back to RGB for API compatibility
        img = img.convert('RGB')
        
        return img

# =============================================================================
# ENHANCED PARSER
# =============================================================================

class EnhancedParser:
    
    def __init__(self, language='english'):
        self.language = language
        self.model = genai.GenerativeModel(
            Config.GEMINI_MODEL,
            generation_config={
                'temperature': 0.1,  # Lower for more consistent extraction
                'top_p': 0.95,
                'top_k': 40,
                'max_output_tokens': 8192,
            }
        )
    
    def parse_page(self, image_path: str, page_num: int, expected_entries: int = 30) -> List[Dict]:
        """Enhanced page parsing with multiple strategies"""
        processing_logger.info(f"Starting enhanced parsing for page {page_num}")
        
        # Strategy 1: Try with original image
        result = self._parse_with_image(image_path, page_num, expected_entries, "original")
        
        if self._is_extraction_complete(result, expected_entries):
            return result
        
        # Strategy 2: Try with preprocessed image
        processing_logger.info(f"Original extraction incomplete for page {page_num}, trying preprocessed image")
        processed_image = ImageProcessor.preprocess_for_ocr(image_path)
        processed_path = image_path.replace('.jpg', '_processed.jpg')
        processed_image.save(processed_path, 'JPEG', quality=95)
        
        result = self._parse_with_image(processed_path, page_num, expected_entries, "preprocessed")
        
        if self._is_extraction_complete(result, expected_entries):
            return result
        
        # Strategy 3: Try with focused re-prompting
        processing_logger.info(f"Trying focused re-extraction for page {page_num}")
        result = self._parse_with_focused_prompt(image_path, page_num, expected_entries, len(result))
        
        return result
    
    def _parse_with_image(self, image_path: str, page_num: int, expected_entries: int, strategy: str) -> List[Dict]:
        """Parse image with specific strategy"""
        try:
            base64_image = encode_image_to_base64(image_path)
            
            if self.language == 'hindi':
                prompt = PromptTemplates.get_hindi_prompt()
            else:
                prompt = PromptTemplates.get_english_prompt()
            
            def api_call():
                return self.model.generate_content([prompt, {"mime_type": "image/jpeg", "data": base64_image}])
            
            response = retry_with_backoff(api_call, Config.MAX_RETRIES, Config.RETRY_BASE_DELAY)
            
            entries = self._extract_json_from_response(response.text, page_num)
            processing_logger.info(f"Strategy '{strategy}' extracted {len(entries)} entries from page {page_num}")
            
            return self._post_process_entries(entries, page_num)
            
        except Exception as e:
            processing_logger.error(f"Error in {strategy} parsing for page {page_num}: {e}")
            return []
    
    def _parse_with_focused_prompt(self, image_path: str, page_num: int, expected_entries: int, current_count: int) -> List[Dict]:
        """Use focused prompt to find missing entries"""
        missing_count = expected_entries - current_count
        
        focused_prompt = f"""
        FOCUSED RE-EXTRACTION: You previously found {current_count} entries, but there should be approximately {expected_entries} entries on this page.
        
        Please carefully re-examine the image and look for {missing_count} additional entries that may have been missed:
        - Check the very top and bottom edges of the page
        - Look for entries with faded or light text
        - Check for entries that might be partially cut off
        - Look for entries in unusual positions or formatting
        
        Extract ALL entries you can find, including the ones you found before.
        """
        
        if self.language == 'hindi':
            base_prompt = PromptTemplates.get_hindi_prompt()
        else:
            base_prompt = PromptTemplates.get_english_prompt()
        
        combined_prompt = focused_prompt + "\n\n" + base_prompt
        
        try:
            base64_image = encode_image_to_base64(image_path)
            
            def api_call():
                return self.model.generate_content([combined_prompt, {"mime_type": "image/jpeg", "data": base64_image}])
            
            response = retry_with_backoff(api_call, Config.MAX_RETRIES, Config.RETRY_BASE_DELAY)
            entries = self._extract_json_from_response(response.text, page_num)
            
            processing_logger.info(f"Focused re-extraction found {len(entries)} entries for page {page_num}")
            return self._post_process_entries(entries, page_num)
            
        except Exception as e:
            processing_logger.error(f"Error in focused parsing for page {page_num}: {e}")
            return []
    
    def _extract_json_from_response(self, response_text: str, page_num: int) -> List[Dict]:
        """Extract and parse JSON from API response"""
        # Try multiple JSON extraction patterns
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',  # Markdown code blocks
            r'```\s*([\s\S]*?)\s*```',      # Generic code blocks
            r'\[\s*\{[\s\S]*\}\s*\]',       # Direct JSON array
        ]
        
        for pattern in patterns:
            json_match = re.search(pattern, response_text)
            if json_match:
                json_str = json_match.group(1) if 'json' in pattern else json_match.group(0)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        # If no pattern works, try the entire response
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            processing_logger.error(f"Failed to parse JSON from page {page_num}: {e}")
            processing_logger.debug(f"Response text: {response_text[:500]}...")
            return []
    
    def _post_process_entries(self, entries: List[Dict], page_num: int) -> List[Dict]:
        """Post-process extracted entries"""
        valid_entries = []
        
        for entry in entries:
            # Clean and validate entry
            cleaned_entry = self._clean_entry(entry)
            
            if validate_voter_entry(cleaned_entry, page_num, self.language):
                valid_entries.append(cleaned_entry)
        
        processing_logger.info(f"Post-processing: {len(valid_entries)}/{len(entries)} entries valid for page {page_num}")
        return valid_entries
    
    def _clean_entry(self, entry: Dict) -> Dict:
        """Clean individual entry data"""
        cleaned = {}
        
        for key, value in entry.items():
            if isinstance(value, str):
                # Remove extra whitespace and clean up
                cleaned_value = re.sub(r'\s+', ' ', str(value).strip())
                # Remove common OCR artifacts for EID
                if key == 'eid':
                    cleaned_value = re.sub(r'[^\w\s\-\.\/]', '', cleaned_value)
                cleaned[key] = cleaned_value
            else:
                cleaned[key] = value
        
        return cleaned
    
    def _is_extraction_complete(self, entries: List[Dict], expected_entries: int) -> bool:
        """Check if extraction is reasonably complete"""
        if not entries:
            return False
        
        coverage = len(entries) / expected_entries
        return coverage >= 0.85  # 85% coverage threshold