import torch
from transformers import pipeline
import os

class LocalComplaintGenerator:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct"):
        print(f"🚀 [Generator] Loading Local LLM: {model_id}...")
        # device=-1 forces CPU. If you have a GPU, change to 0.
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype="auto",
            device_map="auto"
        )
        print("✅ [Generator] Local Model Ready!")

    def generate_answer(self, query: str, context_chunks: list) -> str:
        """
        Generate an answer based on the provided context.
        """
        context_text = "\n\n".join([f"- {c['text']}" for c in context_chunks])
        
        prompt = f"""You are a knowledgeable financial analyst assistant for CrediTrust. 
Your task is to answer questions about customer complaints using ONLY the provided context.
If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context_text}

Question: {query}

Answer:"""

        print("🧠 [Generator] Generating response...")
        outputs = self.pipe(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        full_text = outputs[0]["generated_text"]
        # Extract the answer part after the prompt
        answer = full_text.split("Answer:")[-1].strip()
        
        return answer
