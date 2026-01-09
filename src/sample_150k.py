import json
import random
from pathlib import Path
from tqdm.auto import tqdm

# Configuration
INPUT_FILE = Path(r"C:\Users\My Device\Desktop\week-7-rag-complaint-chatbot\data\processed\chunked_complaints_full.jsonl")
OUTPUT_FILE = Path(r"C:\Users\My Device\Desktop\week-7-rag-complaint-chatbot\data\processed\sampled_150k.jsonl")
SAMPLE_SIZE = 150000

def reservoir_sample(filename, k, seed=42):
    """
    Select k items from a stream of unknown length with uniform probability.
    """
    random.seed(seed)
    sample = []
    
    print(f"🎯 Sampling {k} chunks from {filename.name}...")
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Sampling", unit="lines")):
            if i < k:
                sample.append(line)
            else:
                j = random.randint(0, i)
                if j < k:
                    sample[j] = line
    return sample

def main():
    if not INPUT_FILE.exists():
        print(f"❌ Error: Input file not found at {INPUT_FILE}")
        return

    # 1. Sample the data
    sampled_lines = reservoir_sample(INPUT_FILE, SAMPLE_SIZE)
    
    # 2. Save the sample
    print(f"💾 Saving {len(sampled_lines)} chunks to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.writelines(sampled_lines)
    
    # 3. Output sample chunks (first 3)
    print("\n🔍 Sample Chunks:")
    for i, line in enumerate(sampled_lines[:3]):
        chunk = json.loads(line)
        print(f"\n--- Chunk {i+1} ---")
        print(f"ID: {chunk['chunk_id']}")
        print(f"Product: {chunk.get('product', 'N/A')}")
        print(f"Text: {chunk['text'][:200]}...")

if __name__ == "__main__":
    main()
