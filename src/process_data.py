import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from data_utils import load_complaints_data, filter_products, clean_narratives
from chunking import TextChunker

# Configuration
RAW_DATA_PATH = Path(r"C:\Users\My Device\Desktop\week-7-rag-complaint-chatbot\data\raw\complaints.csv")
FILTERED_OUTPUT_PATH = Path(r"C:\Users\My Device\Desktop\week-7-rag-complaint-chatbot\data\filtered_complaints.csv")
CHUNKED_OUTPUT_PATH = Path(r"C:\Users\My Device\Desktop\week-7-rag-complaint-chatbot\data\processed\chunks_preview.jsonl")

TARGET_PRODUCTS = [
    "Credit card",
    "Credit card or prepaid card", # Often grouped
    "Mortgage",
    "Checking or savings account",
    "Student loan",
    "Vehicle loan or lease"
]

def process_data():
    print("🚀 Starting Data Processing Pipeline...")

    # 1. Load Data
    # Use chunksize to handle large file if necessary, but here we'll try standard load 
    # as implemented in data_utils, but simplified to avoid memory issues if possible.
    print(f"📂 Loading raw data from {RAW_DATA_PATH}...")
    
    # Check if we should use the utility or direct read for efficiency
    if not RAW_DATA_PATH.exists():
        print(f"❌ Error: File not found at {RAW_DATA_PATH}")
        return

    # Reading only necessary columns to save memory
    cols = ['Complaint ID', 'Product', 'Consumer complaint narrative']
    df = pd.read_csv(RAW_DATA_PATH, usecols=cols)
    print(f"✅ Loaded raw data: {len(df):,} rows")

    # 2. Filter Data
    print("🔍 Filtering by products...")
    # Normalize product names if needed or just filter
    df_filtered = df[df['Product'].isin(TARGET_PRODUCTS)].copy()
    print(f"✅ Filtered to {len(df_filtered):,} rows matching target products.")

    # 3. Clean Data
    print("🧹 Cleaning narratives...")
    # Drop rows with missing narratives
    df_filtered = df_filtered.dropna(subset=['Consumer complaint narrative'])
    
    # Simple cleaning (lowercase, strip) - data_utils has this too
    df_filtered['text'] = df_filtered['Consumer complaint narrative'].str.lower().str.strip()
    
    # Drop short complaints
    df_filtered = df_filtered[df_filtered['text'].str.len() > 50]
    print(f"✅ Cleaned data: {len(df_filtered):,} rows with valid narratives.")

    # 4. Save Filtered Dataset (Requirement 1)
    print(f"💾 Saving filtered dataset to {FILTERED_OUTPUT_PATH}...")
    # Save just the essential columns
    df_filtered[['Complaint ID', 'Product', 'text']].to_csv(FILTERED_OUTPUT_PATH, index=False)
    print(f"✅ Saved cleaned dataset.")

    # 5. Chunking Implementation (Requirement 2)
    print("✂️ Demonstrating Chunking Implementation...")
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    
    # Chunk a sample to demonstrate functionality (processing all might take too long for this script)
    sample_size = 1000
    df_sample = df_filtered.head(sample_size)
    print(f"Processing sample of {sample_size} documents...")
    
    documents = df_sample.to_dict('records')
    chunks = chunker.split_documents(documents, text_key='text')
    
    print(f"✅ generated {len(chunks)} chunks from {sample_size} documents.")
    print("Example Chunk:")
    print(chunks[0])

if __name__ == "__main__":
    process_data()
