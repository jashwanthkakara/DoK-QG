import logging
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dataclass
class HierarchicalChunk:
    chunk_id: str
    document_id: str
    text: str
    title: str
    level: int  
    
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data["word_count"] = len(self.text.split())
        return data


class HierarchicalChunker:
    def __init__(self):
        pass
    
    def infer_level(self, marker: str) -> int:
        """Dynamically infer the hierarchy level based on the header type."""
        marker = marker.strip().lower()
        
        # Level 1: Major structural blocks (Unit 24, Chapter 1)
        if any(marker.startswith(x) for x in ['chapter', 'unit', 'part', 'module', 'section']):
            return 1
            
        # Level 2-4: Decimal depth (24.8 -> L2, 24.8.1 -> L3, 24.8.1.1 -> L4)
        decimals = re.findall(r'\d+', marker)
        if '.' in marker and len(decimals) >= 2:
            return min(len(decimals), 4)
            
        # Level 2: Single digit with a dot (1., 2.) OR uppercase letters (A., B.)
        if re.match(r'^\d+\.$', marker) or re.match(r'^[a-z]\.$', marker):
            return 2
            
        # Level 4: Roman Numerals with parenthesis i), ii), iii)
        if re.match(r'^[ivxlcdm]+\)$', marker):
            return 4
            
        # Level 3: Tables and Figures
        if any(marker.startswith(x) for x in ['example', 'figure', 'table']):
            return 3
            
        return 2 # Fallback

    def chunk_document(self, pages: List[str], full_text: str, document_id: str, max_chunk_words: int = 800) -> List[HierarchicalChunk]:
        logger.info(f"Starting Universal Chunking for {document_id}")
        
        # =====================================================================
        # STRICT UNIVERSAL HEADER PATTERN
        # =====================================================================
        pattern = re.compile(
            r'^[ \t]*(?P<marker>'
            r'(?:Chapter|Unit|Part|Module|Section)\s+[A-Z0-9]+'  # Unit 24
            r'|\d+\.\d+(?:\.\d+)*'                                # Decimals: 24.8, 24.8.1 (no trailing dot)
            r'|\d+\.'                                             # Single numbers: 1.
            r'|[A-Z]\.'                                           # Alphabetic: A., B.
            r'|[ivxlcdm]+\)'                                      # Roman numerals with parenthesis: i), ii)
            r'|(?:Example|Figure|Table)\s+\d+(?:\.\d+)?'          # Table 24.4
            r')'
            # Look for the title text on the SAME LINE.
            # Constraints: max 150 chars, doesn't end with sentence punctuation.
            r'[ \t]+(?P<title>(?![a-z])[^
]{2,150}?)(?:\r?\n|$)', 
            re.MULTILINE
        )
        
        matches = list(pattern.finditer(full_text))
        chunks = []
        
        if not matches:
            logger.warning("No structured headers found. Falling back to simple page chunking.")
            chunks.append(HierarchicalChunk(
                chunk_id=f"{document_id}_root", document_id=document_id,
                text=full_text, title="Full Document", level=1
            ))
            return chunks
            
        # Root Node
        root_chunk = HierarchicalChunk(
            chunk_id=f"{document_id}_root",
            document_id=document_id,
            text=full_text[:matches[0].start()].strip() if matches else full_text,
            title="Document Root / Introduction",
            level=1
        )
        chunks.append(root_chunk)
        
        # Keep track of the last seen IDs for hierarchy linking
        last_seen_at_level = {1: root_chunk.chunk_id, 2: None, 3: None, 4: None, 5: None}
        
        for i, match in enumerate(matches):
            marker = match.group('marker').strip()
            title_text = match.group('title').strip()
            
            # CRITICAL FILTER: Skip false positives
            # 1. Skip if the "title" contains verbs that indicate a normal sentence
            lower_title = title_text.lower()
            if any(verb in lower_title for verb in [' provides ', ' shows ', ' is a ', ' are ']):
                continue
                
            # 2. Skip if the "title" is just floating numbers (like table data '55.5 60.3')
            if re.match(r'^[\d\s\.]+$', title_text):
                continue
                
            full_title = f"{marker} {title_text}"
            
            start_idx = match.end()
            # To find the end, we need to know where the NEXT *valid* match starts.
            # For simplicity, we assume the next match in the loop is the boundary.
            end_idx = matches[i+1].start() if i + 1 < len(matches) else len(full_text)
            section_text = full_text[start_idx:end_idx].strip()
            
            level = self.infer_level(marker)
            chunk_id = f"{document_id}_sec_{i}"
            
            # Determine Parent
            parent_level = level - 1
            while parent_level > 0 and last_seen_at_level.get(parent_level) is None:
                parent_level -= 1
            
            parent_id = last_seen_at_level.get(parent_level, root_chunk.chunk_id)
            
            # Update tracker
            last_seen_at_level[level] = chunk_id
            for deeper_level in range(level + 1, 6):
                last_seen_at_level[deeper_level] = None
                
            chunk = HierarchicalChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=f"{full_title}\n\n{section_text}",
                title=full_title,
                level=level,
                parent_id=parent_id
            )
            chunks.append(chunk)

        # Connect Parent/Child Arrays
        chunk_dict = {c.chunk_id: c for c in chunks}
        for c in chunks:
            if c.parent_id and c.parent_id in chunk_dict:
                chunk_dict[c.parent_id].children_ids.append(c.chunk_id)

        logger.info(f"✓ Generated {len(chunks)} strictly structured chunks.")
        return chunks
