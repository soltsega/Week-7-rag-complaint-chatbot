from typing import List, Dict, Any

class TextChunker:
    """
    A class to handle splitting text into smaller chunks for RAG pipelines.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the TextChunker.

        Args:
            chunk_size (int): The maximum size of each chunk (in characters).
            chunk_overlap (int): The number of characters to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Split a single text string into chunks.

        Args:
            text (str): The input text to split.

        Returns:
            List[str]: A list of text chunks.
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start pointer forward, accounting for overlap
            start += self.chunk_size - self.chunk_overlap
            
            # Prevent infinite loops if overlap >= chunk_size (sanity check)
            if self.chunk_overlap >= self.chunk_size:
                start += 1 
        
        return chunks

    def split_documents(self, documents: List[Dict[str, Any]], text_key: str = 'text') -> List[Dict[str, Any]]:
        """
        Split a list of document dictionaries into chunked documents.

        Args:
            documents (List[Dict]): List of document dicts.
            text_key (str): The key in the dict containing the text to split.

        Returns:
            List[Dict]: List of chunked document dicts with metadata preserved.
        """
        chunked_docs = []
        
        for doc in documents:
            text = doc.get(text_key, "")
            # Skip empty texts
            if not isinstance(text, str) or not text.strip():
                continue
                
            chunks = self.split_text(text)
            
            for i, chunk_text in enumerate(chunks):
                # Create a new dict for the chunk, preserving other metadata
                chunk_doc = doc.copy()
                chunk_doc[text_key] = chunk_text
                chunk_doc['chunk_index'] = i
                chunk_doc['chunk_id'] = f"{doc.get('Complaint ID', 'doc')}_{i}"
                chunked_docs.append(chunk_doc)
                
        return chunked_docs
