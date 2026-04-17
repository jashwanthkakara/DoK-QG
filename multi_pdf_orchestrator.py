"""
Multi-PDF Orchestrator (Recursive Directory Support)
====================================================
Batch processes PDFs with per-document indices, auto-generates KG visualizations,
and mirrors deep folder structures using 100% local Python (No API keys needed).
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
import argparse
import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.pdf_loader import load_pdf
from utils.hierarchical_chunker import HierarchicalChunker
from utils.knowledge_graph_builder import build_hierarchical_kg
from utils.visualize_kg import visualize_hierarchy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler('rag_orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load the pre-trained embedding model once globally
# 'all-MiniLM-L6-v2' is fast, lightweight, and runs 100% locally on CPU or GPU
logger.info("Loading pre-trained embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def process_single_pdf(
    pdf_path: Path,
    output_dir: Path
) -> dict:
    """Process a single PDF and save it in the designated output directory."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing: {pdf_path.name}")
    logger.info(f"{'='*70}")
    
    try:
        pages, full_text = load_pdf(str(pdf_path))
        
        doc_hash = hashlib.md5(str(pdf_path).encode()).hexdigest()[:8]
        document_id = f"{pdf_path.stem}_{doc_hash}"
        
        # 100% Local Universal Regex Chunker (No API needed)
        chunker = HierarchicalChunker() 
        chunks = chunker.chunk_document(pages, full_text, document_id)
        
        if not chunks:
            logger.error("No chunks generated")
            return None
        
        logger.info("\nBuilding knowledge graph...")
        graph = build_hierarchical_kg(chunks)
        
        logger.info("Computing embeddings...")
        texts = [f"{c.title}: {c.text[:1000]}" for c in chunks]
        
        # Compute embeddings in a shared global vector space
        # normalize_embeddings=True uses cosine similarity mapping
        embeddings = embed_model.encode(texts, normalize_embeddings=True).astype(np.float32)
        
        # Create the specific document folder inside the target directory
        doc_dir = output_dir / document_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving to {doc_dir}/...")
        
        with open(doc_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in chunks], f, indent=2, ensure_ascii=False)
        
        graph_path = doc_dir / "graph.pkl"
        with open(graph_path, "wb") as f:
            pickle.dump(graph, f)
            
        # Generate Visualization using local math (No Graphviz needed)
        try:
            image_path = doc_dir / "graph_visualization.png"
            visualize_hierarchy(str(graph_path), str(image_path))
        except Exception as vis_err:
            logger.warning(f"Could not generate graph image: {vis_err}")
        
        # Save the embeddings array
        np.save(doc_dir / "embeddings.npy", embeddings)
        
        with open(doc_dir / "id_index.json", "w") as f:
            json.dump({str(i): c.chunk_id for i, c in enumerate(chunks)}, f)
        
        metadata = {
            "document_id": document_id,
            "file_name": pdf_path.name,
            "processed_at": datetime.now().isoformat(),
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "total_words": sum(len(c.text.split()) for c in chunks),
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
            "store_path": str(doc_dir), # Save exact path so Retriever finds it easily
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dim": embeddings.shape[1]
        }
        
        with open(doc_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Saved {len(chunks)} chunks to {doc_dir}/")
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path.name}: {e}", exc_info=True)
        return None


def batch_process(input_dir: str, output_dir: str = None):
    """Batch process all PDFs, preserving nested folder structures."""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"Input path not found: {input_dir}")
        sys.exit(1)
        
    # Auto-name output directory if none is provided
    if not output_dir:
        output_path = Path(f"rag_store_{input_path.name}" if input_path.is_dir() else "rag_store")
    else:
        output_path = Path(output_dir)
        
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Handle both single file and recursive directory scanning
    if input_path.is_file():
        pdfs = [input_path]
        base_input_path = input_path.parent
    else:
        pdfs = sorted(input_path.rglob("*.pdf"))  
        base_input_path = input_path
    
    logger.info(f"\n{'='*70}")
    logger.info(f"  Hierarchical Multi-PDF RAG Orchestrator (100% Local)")
    logger.info(f"{'='*70}")
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"PDFs:   {len(pdfs)}\n")
    
    all_metadata = {}
    
    for i, pdf_path in enumerate(pdfs, 1):
        logger.info(f"\n[{i}/{len(pdfs)}] Processing: {pdf_path.relative_to(base_input_path)}")
        
        rel_path = pdf_path.relative_to(base_input_path)
        
        target_dir = output_path / rel_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Process without API client
        metadata = process_single_pdf(pdf_path, target_dir)
        if metadata:
            all_metadata[metadata["document_id"]] = metadata
    
    master = {
        "timestamp": datetime.now().isoformat(),
        "input_source": str(input_path),
        "embedding_model": "all-MiniLM-L6-v2",
        "total_documents": len(all_metadata),
        "documents": all_metadata,
    }
    
    with open(output_path / "master_index.json", "w") as f:
        json.dump(master, f, indent=2)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"✓ Successfully processed {len(all_metadata)} PDFs")
    logger.info(f"✓ Master index: {output_path}/master_index.json")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input PDF or nested directory of PDFs")
    parser.add_argument("--output", "-o", default=None, help="Output directory (defaults to rag_store_<input_name>)")
    args = parser.parse_args()
    
    batch_process(args.input, args.output)
