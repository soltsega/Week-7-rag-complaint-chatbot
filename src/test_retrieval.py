import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.retriever import ComplaintRetriever

def test_retrieval():
    vector_store_dir = Path(r"C:\Users\My Device\Desktop\week-7-rag-complaint-chatbot\vector_store")
    
    try:
        retriever = ComplaintRetriever(vector_store_dir)
    except Exception as e:
        print(f"❌ Error initializing retriever: {e}")
        return

    test_queries = [
        "What are common issues with credit reports?",
        "Difficulty with mortgage payments and foreclosure",
        "Identity theft and unauthorized accounts",
        "Debt collection harassment and verification"
    ]

    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        results = retriever.search(query, top_k=3)
        
        for i, res in enumerate(results):
            print(f"  [{i+1}] Score: {res['score']:.4f} | Product: {res['product']}")
            print(f"      Text: {res['text'][:150]}...")

if __name__ == "__main__":
    test_retrieval()
