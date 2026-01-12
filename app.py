import gradio as gr
import sys
from pathlib import Path
import time
import os

# Ensure src is in path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.rag.pipeline import RAGPipeline

# Initialize Pipeline
# This will load the local LLM and the 1.3M FAISS index
print("🚀 Initializing CrediTrust RAG Pipeline...")
try:
    rag = RAGPipeline(vector_store_dir=str(root_dir / "vector_store"))
    print("✅ Pipeline Ready!")
except Exception as e:
    print(f"❌ Failed to load pipeline: {e}")
    rag = None

# Product options for filtering
PRODUCT_OPTIONS = [
    "All Products",
    "Credit card",
    "Mortgage",
    "Debt collection",
    "Student loan",
    "Vehicle loan",
    "Checking account",
    "Savings account",
    "Money transfer"
]

# Premium Theme Definition
THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

def respond(message, chat_history, product_filter):
    if not rag:
        return "", chat_history + [[message, "⚠️ System Error: RAG Pipeline failed to initialize."]], ""

    filter_val = None if product_filter == "All Products" else product_filter
    
    # 1. Run Query
    # Note: We aren't doing true streaming from the model yet as it requires 
    # specific yield logic in the generator. We'll simulate word-by-word for UX.
    response_data = rag.query(message, product_filter=filter_val)
    answer = response_data['answer']
    sources = response_data['source_documents']

    # 2. Format Sources for the accordion
    source_display = ""
    for i, doc in enumerate(sources):
        source_display += f"### Source {i+1}\n"
        source_display += f"**Product:** {doc['product']} | **Score:** {doc['score']:.4f}\n\n"
        source_display += f"{doc['text']}\n\n---\n\n"

    # 3. Simulated Streaming for UX
    chat_history.append((message, ""))
    temp_answer = ""
    for word in answer.split():
        temp_answer += word + " "
        chat_history[-1] = (message, temp_answer.strip())
        time.sleep(0.05) # Adjust speed for "thinking" effect
        yield "", chat_history, source_display

with gr.Blocks(theme=THEME, title="CrediTrust AI - Complaint Analyst") as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("""
            # 🛡️ CrediTrust AI
            ### Financial Complaint Analyst
            
            Welcome to the Internal Stakeholder Dashboard. Use this tool to analyze customer pain points across 1.3 million complaints.
            
            **Mission**: Transform customer feedback into actionable insights for CrediTrust.
            """)
            
            product_dropdown = gr.Dropdown(
                choices=PRODUCT_OPTIONS, 
                value="All Products", 
                label="Product Filter",
                info="Narrow down results to a specific business unit."
            )
            
            gr.Examples(
                examples=[
                    ["Why are people unhappy with Credit Cards?"],
                    ["Common issues with mortgage foreclosures?"],
                    ["What are the complaints about debt collection?"],
                    ["Issues with opening savings accounts?"]
                ],
                inputs=gr.Textbox(visible=False), # Hidden inputs for examples
                label="Common Questions",
                fn=lambda x: x,
                outputs=gr.Textbox(visible=False), # Placeholder
                run_on_click=True # Should fill the msg box
            )
            
            gr.Markdown("---")
            gr.Markdown("💡 **Tip**: Use the **Evidence** section below the chat to verify the AI's answer against real consumer narratives.")

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="CrediTrust Analyst", bubble_full_width=False, height=500)
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask Asha's question here... (e.g. why are people unhappy with personal loans?)",
                    label="Question",
                    scale=8
                )
                submit = gr.Button("🚀 Ask AI", scale=1, variant="primary")
            
            with gr.Row():
                clear = gr.ClearButton([msg, chatbot], scale=1)
                
            with gr.Accordion("🔍 Evidence (Source Complaints)", open=False):
                sources_output = gr.Markdown("Source excerpts will appear here after your first question.")

    # Event Handlers
    submit_event = submit.click(respond, [msg, chatbot, product_dropdown], [msg, chatbot, sources_output])
    msg.submit(respond, [msg, chatbot, product_dropdown], [msg, chatbot, sources_output])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False,
        show_api=False
    )
