import pandas as pd
import faiss
import numpy as np
import json
from pathlib import Path
import os
import pyarrow.parquet as pq

# Paths
RAW_DATA = Path("data/raw/complaint_embeddings.parquet")
OUTPUT_DIR = Path("vector_store")
INDEX_PATH = OUTPUT_DIR / "full_faiss_index.index"
METADATA_PATH = OUTPUT_DIR / "full_metadata.json"

def main():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    print(f"📂 Loading embeddings from {RAW_DATA}...")
    
    # We use pyarrow to read specifically the 'embedding' column for speed and RAM
    # And 'document' + 'metadata' for our retrieval context
    
    pf = pq.ParquetFile(RAW_DATA)
    total_rows = pf.metadata.num_rows
    print(f"📊 Total records found: {total_rows:,}")

    embeddings_list = []
    metadata_list = []
    
    # Process in batches to monitor memory
    batch_size = 100000 
    
    print(f"⚡ Processing {total_rows:,} records in batches of {batch_size:,}...")
    
    # We'll use a loop to read row groups or batches
    # For simplicity and given 16GB RAM, we might be able to read columns directly 
    # but let's be safe and use a generator/batch approach if possible.
    
    # Actually, Reading the whole 'embedding' column might take ~2.5GB.
    # Total RAM 16GB. This should be fine.
    
    print("🧠 Extracting embeddings...")
    df = pd.read_parquet(RAW_DATA, columns=['id', 'document', 'embedding', 'metadata'])
    
    print("🏗️ Converting embeddings to numpy array...")
    # 'embedding' column usually contains lists of floats
    all_embeddings = np.vstack(df['embedding'].values).astype('float32')
    
    print(f"🗄️ Building FAISS index (Size: {all_embeddings.shape})...")
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(all_embeddings)
    
    print(f"💾 Saving index to {INDEX_PATH}...")
    faiss.write_index(index, str(INDEX_PATH))
    
    print("📝 Saving metadata...")
    # Combining metadata column (which is likely a dict) with id and document
    # We convert to a list of dicts for the retriever
    metadata_entries = []
    for i, row in df.iterrows():
        entry = {
            'id': row['id'],
            'text': row['document'],
            'meta': row['metadata'] # This contains company, product, etc.
        }
        metadata_entries.append(entry)
        
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata_entries, f)

    print(f"✅ Full Indexing Complete! index: {INDEX_PATH.stat().st_size / 1024**2:.2f} MB")

if __name__ == "__main__":
    main()
