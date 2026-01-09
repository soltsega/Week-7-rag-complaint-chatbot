import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pandas as pd

# Config
INPUT_FILE = Path(r"C:\Users\My Device\Desktop\week-7-rag-complaint-chatbot\data\processed\sampled_150k.jsonl")
OUTPUT_DIR = Path(r"C:\Users\My Device\Desktop\week-7-rag-complaint-chatbot\vector_store")
MODEL_NAME = 'all-MiniLM-L6-v2'
BATCH_SIZE = 256

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Loading chunks from {INPUT_FILE}...")
    chunks = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"✅ Loaded {len(chunks)} chunks.")
    
    # Initialize model
    print(f"🚀 Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    # Generate embeddings
    print("⚡ Generating embeddings (this may take 10-15 mins)...")
    texts = [c['text'] for c in chunks]
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)
    
    # Create Index
    print("🗄️ Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save Index
    index_path = OUTPUT_DIR / 'medium_faiss_index.index'
    faiss.write_index(index, str(index_path))
    print(f"💾 Index saved to {index_path}")
    
    # Save Metadata
    metadata_path = OUTPUT_DIR / 'medium_metadata.json'
    # Keep only necessary fields
    meta_export = [{
        'chunk_id': c['chunk_id'],
        'text': c['text'],
        'product': c.get('product', 'Unknown'),
        'original_id': c.get('original_id')
    } for c in chunks]
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(meta_export, f)
    print(f"📄 Metadata saved to {metadata_path}")
    print("🎉 Done! Vector store is ready.")

if __name__ == "__main__":
    main()
