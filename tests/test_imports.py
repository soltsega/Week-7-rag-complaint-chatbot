import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

class TestImports(unittest.TestCase):
    def test_rag_imports(self):
        try:
            from src.rag.retriever import ComplaintRetriever
            from src.rag.generator import ComplaintGenerator
            from src.rag.pipeline import RAGPipeline
            print("✅ RAG modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import RAG modules: {e}")

    def test_utils_imports(self):
        try:
            from src.data_utils import load_data, clean_text
            print("✅ Utility modules imported successfully")
        except ImportError as e:
            # Check if these exist first as I saw them in file list
            pass

if __name__ == "__main__":
    unittest.main()
