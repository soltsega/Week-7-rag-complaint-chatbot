import json
from pathlib import Path

nb_path = Path(r"C:\Users\My Device\Desktop\week-7-rag-complaint-chatbot\notebooks\task-3-ragPipeline.ipynb")

cells = [
 {
  "cell_type": "markdown",
  "id": "md1",
  "metadata": {},
  "source": [
   "# Task 3: RAG Core Logic & Evaluation\n",
   "\n",
   "This notebook implements the core RAG components (Retriever, Generator, Pipeline) and performs an evaluation using test questions."
  ]
 },
 {
  "cell_type": "code",
  "execution_count": None,
  "id": "code1",
  "metadata": {},
  "outputs": [],
  "source": [
   "import json\n",
   "import faiss\n",
   "import numpy as np\n",
   "import pandas as pd\n",
   "from pathlib import Path\n",
   "from sentence_transformers import SentenceTransformer\n",
   "\n",
   "# Configuration\n",
   "VECTOR_STORE_DIR = Path('../vector_store')\n",
   "INDEX_PATH = VECTOR_STORE_DIR / 'medium_faiss_index.index'\n",
   "METADATA_PATH = VECTOR_STORE_DIR / 'medium_metadata.json'\n",
   "MODEL_NAME = 'all-MiniLM-L6-v2'"
  ]
 },
 {
  "cell_type": "markdown",
  "id": "md2",
  "metadata": {},
  "source": [
   "## 1. Implement Retriever\n",
   "The retriever loads the FAISS index and searches for semantically similar chunks."
  ]
 },
 {
  "cell_type": "code",
  "execution_count": None,
  "id": "code2",
  "metadata": {},
  "outputs": [],
  "source": [
   "class ComplaintRetriever:\n",
   "    def __init__(self, index_path, metadata_path, model_name='all-MiniLM-L6-v2'):\n",
   "        print(f\"🔧 Loading model: {model_name}...\")\n",
   "        self.model = SentenceTransformer(model_name)\n",
   "        \n",
   "        print(f\"📂 Loading index from {index_path}...\")\n",
   "        if not index_path.exists():\n",
   "            raise FileNotFoundError(f\"Index not found at {index_path}\")\n",
   "        self.index = faiss.read_index(str(index_path))\n",
   "        \n",
   "        print(f\"📄 Loading metadata...\")\n",
   "        with open(metadata_path, 'r', encoding='utf-8') as f:\n",
   "            self.metadata = json.load(f)\n",
   "\n",
   "    def search(self, query, top_k=5):\n",
   "        query_embedding = self.model.encode([query], convert_to_numpy=True)\n",
   "        distances, indices = self.index.search(query_embedding, top_k)\n",
   "        \n",
   "        results = []\n",
   "        for idx, dist in zip(indices[0], distances[0]):\n",
   "            if idx == -1: continue\n",
   "            meta = self.metadata[idx]\n",
   "            results.append({\n",
   "                'text': meta['text'],\n",
   "                'product': meta.get('product', 'N/A'),\n",
   "                'score': float(1 / (1 + dist))\n",
   "            })\n",
   "        return results"
  ]
 },
 {
  "cell_type": "markdown",
  "id": "md3",
  "metadata": {},
  "source": [
   "## 2. Implement Generator (Mock)\n",
   "Using a mock generator as per the project plan (Option 1)."
  ]
 },
 {
  "cell_type": "code",
  "execution_count": None,
  "id": "code3",
  "metadata": {},
  "outputs": [],
  "source": [
   "class ComplaintGenerator:\n",
   "    def generate_answer(self, query, context):\n",
   "        if not context:\n",
   "            return \"No relevant complaints found.\"\n",
   "        \n",
   "        # Heuristic/Mock generation\n",
   "        products = list(set(c['product'] for c in context))\n",
   "        response = f\"**Analysis for: {query}**\\n\\n\"\n",
   "        response += f\"Based on complaints regarding {', '.join(products)}, here are key insights:\\n\"\n",
   "        for i, c in enumerate(context[:3]):\n",
   "            response += f\"- Complaint {i+1}: {c['text'][:150]}...\\n\"\n",
   "        return response"
  ]
 },
 {
  "cell_type": "markdown",
  "id": "md4",
  "metadata": {},
  "source": [
   "## 3. RAG Pipeline\n",
   "Combining steps."
  ]
 },
 {
  "cell_type": "code",
  "execution_count": None,
  "id": "code4",
  "metadata": {},
  "outputs": [],
  "source": [
   "class RAGPipeline:\n",
   "    def __init__(self):\n",
   "        self.retriever = ComplaintRetriever(INDEX_PATH, METADATA_PATH)\n",
   "        self.generator = ComplaintGenerator()\n",
   "        \n",
   "    def run(self, query):\n",
   "        context = self.retriever.search(query)\n",
   "        answer = self.generator.generate_answer(query, context)\n",
   "        return {'answer': answer, 'context': context}\n",
   "\n",
   "# Initialize (Only run this AFTER index is ready)\n",
   "# pipeline = RAGPipeline()"
  ]
 },
 {
  "cell_type": "markdown",
  "id": "md5",
  "metadata": {},
  "source": [
   "## 4. Evaluation\n",
   "Running test questions."
  ]
 },
 {
  "cell_type": "code",
  "execution_count": None,
  "id": "code5",
  "metadata": {},
  "outputs": [],
  "source": [
   "test_questions = [\n",
   "    \"What are the most common complaints about credit cards?\",\n",
   "    \"How do students feel about their loans?\",\n",
   "    \"Are there issues with mortgage payments?\",\n",
   "    \"Complaints about unauthorized transactions?\",\n",
   "    \"Issues with checking account fees?\"\n",
   "]\n",
   "\n",
   "# Run Evaluation (Uncomment when ready)\n",
   "# results = []\n",
   "# pipeline = RAGPipeline()\n",
   "# for q in test_questions:\n",
   "#     out = pipeline.run(q)\n",
   "#     results.append({'Question': q, 'Answer': out['answer'], 'Sources': len(out['context'])})\n",
   "# pd.DataFrame(results)"
  ]
 }
]

notebook = {
 "cells": cells,
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
print(f"Created {nb_path}")
