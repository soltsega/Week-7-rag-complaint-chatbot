import faiss
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

class ComplaintRetriever:
    def __init__(self, vector_store_dir: Path, model_name: str = 'all-MiniLM-L6-v2'):
        self.vector_store_dir = Path(vector_store_dir)
        self.index_path = self.vector_store_dir / 'medium_faiss_index.index'
        self.metadata_path = self.vector_store_dir / 'medium_metadata.json'
        
        print(f"🔧 [Retriever] Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        print(f"📂 [Retriever] Loading index from {self.index_path}...")
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found at {self.index_path}. Please run vectorization first.")
            
        self.index = faiss.read_index(str(self.index_path))
        
        print(f"📄 [Retriever] Loading metadata...")
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        print("✅ [Retriever] Ready!")

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
            
            results.append({
                'text': meta['text'],
                'product': meta.get('product', 'N/A'),
                'chunk_id': meta['chunk_id'],
                'score': float(1 / (1 + dist)) # Convert L2 distance to similarity score
            })
            
            if len(results) >= top_k:
                break
                
        return results
