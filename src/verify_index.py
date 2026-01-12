import faiss
import json
import numpy as np
from pathlib import Path

# Config
OUTPUT_DIR = Path(r"C:\Users\My Device\Desktop\week-7-rag-complaint-chatbot\vector_store")
INDEX_PATH = OUTPUT_DIR / 'medium_faiss_index.index'
METADATA_PATH = OUTPUT_DIR / 'medium_metadata.json'

def verify():
    print("🔍 Verifying Vector Store Integrity...")
    
    # 1. Check files existence
    if not INDEX_PATH.exists():
        print(f"❌ Index file missing: {INDEX_PATH}")
        return
    if not METADATA_PATH.exists():
        print(f"❌ Metadata file missing: {METADATA_PATH}")
        return

    # 2. Load Index
    print(f"📂 Loading Index from {INDEX_PATH}...")
    try:
        index = faiss.read_index(str(INDEX_PATH))
    except Exception as e:
        print(f"❌ Failed to load index: {e}")
        return
    
    ntotal = index.ntotal
    print(f"✅ Index loaded. Total vectors: {ntotal}")

    # 3. Load Metadata
    print(f"📂 Loading Metadata from {METADATA_PATH}...")
    try:
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load metadata: {e}")
        return
    
    nmeta = len(metadata)
    print(f"✅ Metadata loaded. Total items: {nmeta}")

    # 4. Compare
    if ntotal == nmeta:
        print("🎉 SUCCESS: Index and Metadata counts match!")
    else:
        print(f"⚠️ WARNING: Count mismatch! Index: {ntotal}, Metadata: {nmeta}")

    # 5. Dummy Search Test
    print("\n🧪 Performing dummy search test...")
    if ntotal > 0:
        d = index.d
        query_vector = np.random.rand(1, d).astype('float32')
        k = 5
        D, I = index.search(query_vector, k)
        print(f"✅ Search successful. Returned indices: {I}")
    else:
        print("⚠️ Index is empty, skipping search test.")

if __name__ == "__main__":
    verify()
