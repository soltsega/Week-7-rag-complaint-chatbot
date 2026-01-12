import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pandas as pd

# Config
INPUT_FILE = Path(r"C:\Users\My Device\Desktop\week-7-rag-complaint-chatbot\data\processed\sampled_150k.jsonl")
OUTPUT_DIR = Path(r"C:\Users\My Device\Desktop\week-7-rag-complaint-chatbot\vector_store")
INDEX_PATH = OUTPUT_DIR / 'medium_faiss_index.index'
METADATA_PATH = OUTPUT_DIR / 'medium_metadata.json'
MODEL_NAME = 'all-MiniLM-L6-v2'
BATCH_SIZE = 256

def load_existing_progress():
    if INDEX_PATH.exists() and METADATA_PATH.exists():
        print(f"🔄 Found existing data. Loading...")
        try:
            index = faiss.read_index(str(INDEX_PATH))
            with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"✅ Loaded {index.ntotal} vectors and {len(metadata)} metadata items.")
            return index, metadata
        except Exception as e:
            print(f"⚠️ Error loading existing data: {e}. Starting fresh.")
            return None, []
    return None, []

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Input Data
    print(f"📂 Loading chunks from {INPUT_FILE}...")
    all_chunks = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            all_chunks.append(json.loads(line))
    print(f"✅ Total input chunks: {len(all_chunks)}")
    
    # 2. Check for existing progress
    index, metadata = load_existing_progress()
    
    if index is None:
        print("🆕 Starting fresh embedding process...")
        dimension = 384 # all-MiniLM-L6-v2 dimension
        index = faiss.IndexFlatL2(dimension)
        metadata = []
        processed_ids = set()
    else:
        # Create set of processed IDs for fast lookup
        processed_ids = {item['chunk_id'] for item in metadata}
        print(f"♻️ Resuming: {len(processed_ids)} chunks already processed.")
        
    # 3. Filter chunks to process
    chunks_to_process = [c for c in all_chunks if c['chunk_id'] not in processed_ids]
    
    if not chunks_to_process:
        print("🎉 All chunks already processed! Nothing to do.")
        return

    print(f"📝 {len(chunks_to_process)} chunks remaining to embed.")
    
    # 4. Initialize model
    print(f"🚀 Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 5. Process in batches
    batch_size = BATCH_SIZE
    total_processed = 0
    checkpoint_interval = 5000 # Save every 5000 chunks
    
    for i in range(0, len(chunks_to_process), batch_size):
        batch = chunks_to_process[i:i + batch_size]
        texts = [c['text'] for c in batch]
        
        # Embed
        embeddings = model.encode(texts, batch_size=len(batch), show_progress_bar=False, convert_to_numpy=True)
        
        # Add to index
        index.add(embeddings)
        
        # Update metadata
        for c in batch:
            metadata.append({
                'chunk_id': c['chunk_id'],
                'text': c['text'],
                'product': c.get('product', 'Unknown'),
                'original_id': c.get('original_id')
            })
            
        total_processed += len(batch)
        print(f"✅ Processed batch {i//batch_size + 1} ({len(batch)} items). Total new: {total_processed}")
        
        # Checkpoint
        if total_processed % checkpoint_interval < batch_size: # Approximate check
            print("💾 Saving checkpoint...")
            faiss.write_index(index, str(INDEX_PATH))
            with open(METADATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(metadata, f)
            print(f"Checkpoint saved at {len(metadata)} items.")

    # 6. Final Save
    print("💾 Saving final output...")
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f)
    print(f"🎉 Done! Final count: {len(metadata)} items.")

if __name__ == "__main__":
    main()
