"""
Context Retriever (Graph-RAG Enabled)
=====================================
Takes a topic/heading/query, retrieves highly relevant chunks via semantic search,
filters out irrelevant chunks using a dynamic similarity drop-off, and uses the 
locally indexed Knowledge Graph to expand the context with connected hierarchical 
nodes (parents, children, siblings).
"""

import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextRetriever:
    """Retrieves raw text context from the Graph-RAG store."""
    
    def __init__(self, store_dir: str = "rag_store/"):
        """Load RAG store and embedding model."""
        self.store_dir = Path(store_dir)
        
        if not self.store_dir.exists():
            raise FileNotFoundError(f"Store not found: {store_dir}")
        
        logger.info(f"Loading master index from {store_dir}...")
        with open(self.store_dir / "master_index.json", "r") as f:
            self.master = json.load(f)
            
        # Load the same model used in the orchestrator
        logger.info("Loading pre-trained embedding model...")
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Pre-load all chunks, embeddings, and graphs into memory for fast global search
        self.all_chunks = []
        self.all_embeddings = []
        self.graphs = {}
        self.chunk_lookup = {}
        
        self._load_all_documents()

    def _load_all_documents(self):
        """Loads all chunks, embeddings, and knowledge graphs into memory."""
        logger.info("Loading document embeddings and graphs into memory...")
        for doc_id, meta in self.master["documents"].items():
            
            # 1. Try to use the exact path saved by the orchestrator
            doc_dir = Path(meta["store_path"])
            
            # 2. Fallback: If the store folder was renamed/moved, dynamically search for the doc_id folder
            if not doc_dir.exists():
                logger.info(f"Exact path not found, searching inside {self.store_dir}...")
                found_dirs = list(self.store_dir.rglob(doc_id))
                if found_dirs:
                    doc_dir = found_dirs[0]
                else:
                    logger.error(f"Could not find chunks for document {doc_id}. Skipping.")
                    continue
            
            # Load chunks
            with open(doc_dir / "chunks.json", "r") as f:
                chunks = json.load(f)
                
            # Load embeddings
            embeddings = np.load(doc_dir / "embeddings.npy")
            
            # Load Knowledge Graph
            graph_path = doc_dir / "graph.pkl"
            if graph_path.exists():
                with open(graph_path, "rb") as f:
                    self.graphs[doc_id] = pickle.load(f)
            
            # Initialize fast lookup for this document
            self.chunk_lookup[doc_id] = {}
            
            # Tag chunks with their source document ID and build lookup dictionary
            for chunk in chunks:
                chunk["doc_id"] = doc_id
                chunk["file_name"] = meta["file_name"]
                
                # Save to fast lookup dictionary by chunk_id for Graph expansion
                chunk_id_val = chunk.get("chunk_id")
                if chunk_id_val:
                    self.chunk_lookup[doc_id][chunk_id_val] = chunk
                
            self.all_chunks.extend(chunks)
            self.all_embeddings.append(embeddings)
            
        # Stack all embeddings into a single giant matrix
        if self.all_embeddings:
            self.all_embeddings = np.vstack(self.all_embeddings)
            logger.info(f"Loaded {len(self.all_chunks)} total chunks and {len(self.graphs)} Knowledge Graphs.")

    def search_chunks(self, topic: str, top_k: int = 5) -> List[Dict]:
        """Finds the top_k most relevant chunks across ALL documents."""
        if not self.all_chunks:
            return []
            
        # Encode the topic/heading into the same vector space
        query_vec = self.embed_model.encode([topic], normalize_embeddings=True).astype(np.float32)
        
        # Calculate cosine similarity against ALL chunks at once (dot product of normalized vectors)
        sims = np.dot(self.all_embeddings, query_vec.T).flatten()
        
        # Get indices of top_k highest scores
        top_idx = np.argsort(sims)[::-1][:top_k]
        
        results = []
        for idx in top_idx:
            chunk = self.all_chunks[idx].copy()
            chunk["similarity"] = float(sims[idx])
            results.append(chunk)
            
        return results

    def filter_by_dynamic_dropoff(self, results: List[Dict], max_drop_pct: float = 0.20) -> List[Dict]:
        """
        Takes a list of sorted chunks and filters out the ones where the similarity 
        score drops by more than `max_drop_pct` compared to the best match.
        """
        if not results:
            return []
            
        filtered_results = []
        best_score = results[0].get("similarity", 0.0)
        
        for idx, chunk in enumerate(results):
            score = chunk.get("similarity", 0.0)
            
            # Calculate how much the score dropped compared to the #1 result
            if best_score > 0:
                drop_pct = (best_score - score) / best_score
            else:
                drop_pct = 0.0
                
            # If the drop is too large (and we already have at least 1 result), stop taking more
            if drop_pct > max_drop_pct and len(filtered_results) >= 1:
                logger.info(f"Dynamic Cutoff: Dropping remaining chunks. Chunk {idx+1} dropped by {drop_pct:.1%} (Limit: {max_drop_pct:.1%})")
                break
                
            filtered_results.append(chunk)
            
        if len(filtered_results) < len(results):
            logger.info(f"Filtered {len(results)} initial chunks down to top {len(filtered_results)}.")
            
        return filtered_results

    def expand_context_with_graph(self, doc_id: str, chunk_id: str) -> str:
        """Finds connected nodes in the knowledge graph to add surrounding context."""
        if doc_id not in self.graphs or not chunk_id:
            return ""
            
        graph = self.graphs[doc_id]
        if chunk_id not in graph.nodes:
            return ""
            
        expanded_text = []
        
        try:
            # If the graph is directed (DiGraph), separate parents (predecessors) and children (successors)
            if isinstance(graph, nx.DiGraph):
                # Get Parent contexts (broader concepts)
                parents = list(graph.predecessors(chunk_id))
                for p_id in parents:
                    p_chunk = self.chunk_lookup.get(doc_id, {}).get(p_id)
                    if p_chunk:
                        expanded_text.append(f"[Broader Context / Parent: {p_chunk.get('title', 'Untitled')}]:\n{p_chunk.get('text', '').strip()}")
                
                # Get Children contexts (deeper details/bullet points)
                children = list(graph.successors(chunk_id))
                for c_id in children:
                    c_chunk = self.chunk_lookup.get(doc_id, {}).get(c_id)
                    if c_chunk:
                        expanded_text.append(f"[Deeper Detail / Child: {c_chunk.get('title', 'Untitled')}]:\n{c_chunk.get('text', '').strip()}")
            
            # If it's an undirected graph, just get neighbors
            else:
                neighbors = list(graph.neighbors(chunk_id)) 
                for neighbor_id in neighbors:
                    neighbor_chunk = self.chunk_lookup.get(doc_id, {}).get(neighbor_id)
                    if neighbor_chunk:
                        expanded_text.append(f"[Related Context: {neighbor_chunk.get('title', 'Untitled')}]:\n{neighbor_chunk.get('text', '').strip()}")
                    
        except Exception as e:
            logger.warning(f"Error during graph traversal for {chunk_id}: {e}")
                
        return "\n\n".join(expanded_text)

    def get_relevant_context(self, topic: str, top_k: int = 5, max_drop_pct: float = 0.20) -> str:
        """
        Takes a topic/heading, searches the index, dynamically filters irrelevant hits,
        expands context using the Knowledge Graph, and returns a formatted string.
        """
        logger.info(f"Retrieving context for topic: '{topic}'")
        
        # 1. Get the initial top K results
        results = self.search_chunks(topic, top_k)
        
        if not results:
            return "No relevant context found in the indexed documents."
            
        # 2. Filter them dynamically to only keep the highly relevant ones
        results = self.filter_by_dynamic_dropoff(results, max_drop_pct=max_drop_pct)
            
        # Format the retrieved chunks into a single readable context block
        context_blocks = []
        for idx, r in enumerate(results, 1):
            # Safely get properties
            p_start = r.get('page_start', r.get('page', '?'))
            p_end = r.get('page_end', p_start)
            title = r.get('title', 'Untitled')
            file_name = r.get('file_name', 'Unknown File')
            similarity = r.get('similarity', 0.0)
            text = r.get('text', '').strip()
            
            # Graph properties
            chunk_id = r.get('chunk_id')
            doc_id = r.get('doc_id')
            
            # 1. Fetch normal text from the semantic match
            source = f"Source {idx}: {file_name} (Pages {p_start}-{p_end}) - Section: {title}"
            relevance = f"Relevance Score: {similarity:.3f}"
            
            # 2. Fetch Graph Connections (Parents/Children)
            if chunk_id and doc_id:
                graph_context = self.expand_context_with_graph(doc_id, chunk_id)
                if graph_context:
                    text = f"{text}\n\n--- Graph Context Expansion (Connected Sections) ---\n{graph_context}"
            
            block = f"--- {source} ---\n{relevance}\n\n{text}\n"
            context_blocks.append(block)
            
        return "\n\n".join(context_blocks)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", "-s", default="rag_store/")
    parser.add_argument("--topic", "-t", help="Heading, topic, or context to search for")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="Maximum number of primary context blocks to return")
    parser.add_argument("--drop_pct", "-d", type=float, default=0.20, help="Maximum percentage drop in similarity score before cutting off results (default: 0.20)")
    
    args = parser.parse_args()
    
    retriever = ContextRetriever(args.store)
    
    if args.topic:
        context = retriever.get_relevant_context(args.topic, top_k=args.top_k, max_drop_pct=args.drop_pct)
        print("\n" + "="*80)
        print(f"RETRIEVED GRAPH-RAG CONTEXT FOR: {args.topic}")
        print("="*80 + "\n")
        print(context)
    else:
        print("Please provide a topic to search for using --topic or -t")
