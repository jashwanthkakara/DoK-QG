"""
PDF Loader - Generalized Clean Text Extraction
==============================================
Extracts raw text from PDFs and cleans general visual margin noise.
"""

import logging
import re
from pathlib import Path
from typing import Tuple, List

import fitz  # PyMUPDF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_page_text(text: str) -> str:
    """Removes general PDF visual noise (headers, footers, page numbers)."""
    # Remove isolated page numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Remove common dotted lines from exercises
    text = re.sub(r'\.{10,}', '', text)
    
    # Remove common textbook watermarks/footers
    text = re.sub(r'(?i)^\s*(Reprint\s*\d{4}-\d{2}|not to be republished)\s*$', '', text, flags=re.MULTILINE)
    
    # IMPROVED: Remove ALL CAPS and Title Case headers (like "Education and Health") 
    # that appear isolated on a single line at the top/bottom of pages.
    text = re.sub(r'^\s*([A-Z][a-zA-Z\s&]+){3,}\s*$', '', text, flags=re.MULTILINE)
    
    return text.strip()

def load_pdf(pdf_path: str) -> Tuple[List[str], str]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    logger.info(f"Loading PDF: {pdf_path.name}")
    doc = fitz.open(str(pdf_path))
    pages = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        cleaned_text = clean_page_text(text)
        if cleaned_text:
            pages.append(cleaned_text)
        
    doc.close()
    
    full_text = "\n\n".join(pages)
    logger.info(f"✓ Loaded {len(pages)} pages, {len(full_text):,.} characters")
    
    return pages, full_text
