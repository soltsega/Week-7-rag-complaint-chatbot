from pathlib import Path
from .retriever import ComplaintRetriever
from .generator import ComplaintGenerator

class RAGPipeline:
    def __init__(self, vector_store_dir: str):
        self.retriever = ComplaintRetriever(Path(vector_store_dir))
        self.generator = ComplaintGenerator()
        
    def query(self, user_question: str, product_filter: str = None) -> dict:
        """
        Main RAG pipeline: Retrieve -> Generate
        Returns a dictionary with 'answer' and 'source_documents'.
        """
        # 1. Retrieve
        context = self.retriever.search(user_question, top_k=5, product_filter=product_filter)
        
        # 2. Generate
        answer = self.generator.generate_answer(user_question, context)
        
        return {
            'answer': answer,
            'source_documents': context
        }
