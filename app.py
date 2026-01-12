import streamlit as st
import time
from src.rag.pipeline import RAGPipeline

# Page Config
st.set_page_config(
    page_title="CrediTrust Complaint Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize Pipeline (Cached to avoid reloading index on every run)
@st.cache_resource
def get_pipeline():
    # Only load if index exists, otherwise return None
    try:
        return RAGPipeline(vector_store_dir="vector_store")
    except Exception as e:
        return None

pipeline = get_pipeline()

# Header
st.title("🤖 CrediTrust Complaint Assistant")
st.markdown("""
This chatbot helps you analyze customer complaints from the CFPB database. 
Ask questions about trends, specific issues, or product-related problems.
""")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    
    # Product Filter
    product = st.selectbox(
        "Filter by Product",
        ["All Products", "Credit card", "Mortgage", "Checking or savings account", "Student loan", "Vehicle loan or lease"]
    )
    product_filter = None if product == "All Products" else product
    
    st.divider()
    st.info("💡 **Tip:** Be specific in your questions for better results.")
    
    # Status Indicator
    if pipeline:
        st.success("✅ System Ready")
    else:
        st.warning("⚠️ Index not found. Please complete Task 2 (Embedding).")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
             with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1} ({source['product']}):** {source['text']}")

# User Input
if prompt := st.chat_input("Ask a question about complaints..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        if pipeline:
            with st.spinner("Analyzing complaints..."):
                response = pipeline.query(prompt, product_filter=product_filter)
                
                st.markdown(response['answer'])
                
                # Show sources
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(response['source_documents']):
                        st.markdown(f"**Source {i+1} ({source['product']}) | Score: {source['score']:.2f}**")
                        st.text(source['text'])
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response['answer'],
                    "sources": response['source_documents']
                })
        else:
            st.error("RAG Pipeline is not ready. Please verify that the vector store has been created.")
