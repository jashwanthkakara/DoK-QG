"""
Patram Text Processor via Groq API
===================================
Uses Groq to call Patram for intelligent text understanding.
"""

import json
import logging
import re
from typing import List, Dict, Optional

from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatramProcessor:
    """Wrapper for Groq/Patram API calls."""
    
    def __init__(self, groq_client: OpenAI, model: str = "mixtral-8x7b-32768"):
        self.client = groq_client
        self.model = model
        logger.info(f"Initialized Patram processor with model: {model}")
    
    def _call_groq(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call Groq API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    
    def _parse_json_response(self, raw: str) -> dict:
        """Parse JSON response, removing markdown fences."""
        raw = re.sub(r"^```json?\n?", "", raw.strip())
        raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    
    def extract_toc(self, full_text: str) -> List[Dict]:
        """Extract table of contents from document text."""
        logger.info("Extracting table of contents via Patram...")
        
        toc_section = "\n".join(full_text.split("\n")[:3000])
        
        prompt = f'''Analyze the following document text and extract the Table of Contents.

DOCUMENT TEXT:
{toc_section}

Your task:
1. Identify all chapters and sections in the document
2. Determine which pages each chapter/section spans
3. Return ONLY a JSON array with NO explanation

Output format:
[
  {{
    "title": "Chapter 1: Introduction",
    "level": 1,
    "page_start": 1,
    "page_end": 15
  }},
  {{
    "title": "1.1 Background",
    "level": 2,
    "page_start": 3,
    "page_end": 8
  }}
]

Rules:
- level 1 = chapter/main section
- level 2 = subsection
- page_start and page_end must be integers
- If no clear TOC structure exists, identify logical chapters
- Return ONLY the JSON array, no markdown fences
'''
        
        try:
            raw = self._call_groq(prompt, max_tokens=3000)
            data = self._parse_json_response(raw)
            
            logger.info(f"✓ Extracted {len(data)} chapters")
            return data
            
        except Exception as e:
            logger.warning(f"TOC extraction failed: {e}")
            return []
    
    def semantic_split(self, text: str, max_length: int = 1500) -> List[str]:
        """Intelligently split text into semantic chunks."""
        word_count = len(text.split())
        
        if word_count <= max_length:
            return [text]
        
        logger.debug(f"Semantic splitting {word_count} words via Patram...")
        
        num_chunks = max(2, (word_count + max_length - 1) // max_length)
        
        prompt = f'''Split the following text into {num_chunks} semantically coherent sections.
Each section should cover one distinct topic.

TEXT:
{text[:4000]}

Rules:
1. Keep ALL original text (no summarization)
2. Split at logical topic boundaries
3. Each section should be roughly similar length
4. Preserve complete sentences/paragraphs

Return JSON array of text strings:
[
  "section 1 text...",
  "section 2 text...",
  "section 3 text..."
]

Return ONLY JSON array, no markdown fences.'''        
        try:
            raw = self._call_groq(prompt, max_tokens=5000)
            data = self._parse_json_response(raw)
            
            combined = "".join(data)
            if len(combined) < len(text) * 0.9:
                logger.warning("Semantic split lost too much text, returning original")
                return [text]
            
            return [s.strip() for s in data if s.strip()]    
            
        except Exception as e:
            logger.warning(f"Semantic split failed: {e}")
            return [text]
    
    def extract_section_title(self, text: str) -> str:
        """Extract the title/heading from a text block."""
        lines = text.split("\n")
        
        for line in lines[:5]:
            line = line.strip()
            if line and not line.startswith("["):
                return line[:100]
        
        return "Untitled Section"