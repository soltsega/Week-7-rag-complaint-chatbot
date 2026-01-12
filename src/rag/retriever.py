import faiss
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

class ComplaintRetriever:
    def __init__(self, vector_store_dir: Path, model_name: str = 'all-MiniLM-L6-v2'):
        self.vector_store_dir = Path(vector_store_dir)
        
        # Check for full-scale index first, then medium, then fall back
        full_index_path = self.vector_store_dir / 'full_faiss_index.index'
        full_meta_path = self.vector_store_dir / 'full_metadata.json'
        
        if full_index_path.exists() and full_meta_path.exists():
            self.index_path = full_index_path
            self.metadata_path = full_meta_path
            self.is_full_scale = True
            print("🏢 [Retriever] Using FULL-SCALE index (1.3M+ Chunks)")
        else:
            self.index_path = self.vector_store_dir / 'medium_faiss_index.index'
            self.metadata_path = self.vector_store_dir / 'medium_metadata.json'
            self.is_full_scale = False
            print("📦 [Retriever] Using MEDIUM/Standard index")
        
        print(f"🔧 [Retriever] Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        print(f"📂 [Retriever] Loading index from {self.index_path}...")
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found at {self.index_path}. Please run indexing first.")
            
        self.index = faiss.read_index(str(self.index_path))
        
        print(f"📄 [Retriever] Loading metadata...")
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        print(f"✅ [Retriever] Ready! ({len(self.metadata):,} chunks loaded)")

    def search(self, query: str, top_k: int = 5, product_filter: str = None) -> list:
        """
        Search for relevant complaints.
        """
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # FAISS search
        # We fetch more than top_k initially to allow for filtering
        fetch_k = top_k * 5 if product_filter else top_k
        distances, indices = self.index.search(query_embedding, fetch_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1: continue # invalid index
            
            meta = self.metadata[idx]
            
            # Apply Filter
            if product_filter and product_filter.lower() not in meta.get('product', '').lower():
                continue
            
            # Format result based on metadata structure (full vs sample)
            if self.is_full_scale:
                # Full scale metadata structure: {'id': ..., 'text': ..., 'meta': {...}}
                results.append({
                    'text': meta['text'],
                    'product': meta['meta'].get('product', 'N/A'),
                    'chunk_id': meta['id'],
                    'score': float(1 / (1 + dist))
                })
            else:
                # Sample/Medium metadata structure: {'text': ..., 'product': ..., 'chunk_id': ...}
                results.append({
                    'text': meta['text'],
                    'product': meta.get('product', 'N/A'),
                    'chunk_id': meta['chunk_id'],
                    'score': float(1 / (1 + dist))
                })
            
            if len(results) >= top_k:
                break
                
        return results
