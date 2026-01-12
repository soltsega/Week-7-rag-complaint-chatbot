import sys
from pathlib import Path
import time

root_dir = Path(".").resolve()
sys.path.append(str(root_dir))

from src.rag.pipeline import RAGPipeline

print("🔍 Starting Pipeline Diagnostic...")
start_time = time.time()

try:
    rag = RAGPipeline(vector_store_dir="vector_store")
    print(f"✅ Pipeline Loaded in {time.time() - start_time:.2f} seconds")
    
    # Test a small query
    print("Testing a small query...")
    res = rag.query("Test")
    print("✅ Query Successful!")
    print(f"Answer: {res['answer']}")
except Exception as e:
    print(f"❌ Diagnostic Failed: {e}")
