import gradio as gr
from utils.qa_chain import search_code

# Create the Gradio interface
with gr.Blocks(title="Smart Code Search") as demo:
    gr.Markdown("# 🔍 Smart Code Search")
    gr.Markdown("Search your codebase and get AI-powered answers about your code")
    
    with gr.Row():
        with gr.Column(scale=3):
            # Search query input
            query_input = gr.Textbox(
                label="Search Query",
                placeholder="Ask a question about your code or search for specific functionality",
                lines=3
            )
            
            # Search options
            with gr.Row():
                search_type = gr.Radio(
                    ["Question Answering", "Code Retrieval"],
                    label="Search Type",
                    value="Question Answering"
                )
                num_results = gr.Slider(
                    minimum=1,
                    maximum=15,
                    value=5,
                    step=1,
                    label="Number of Results (for Code Retrieval)"
                )
            
            # Add re-ranking option
            with gr.Row():
                use_reranking = gr.Checkbox(
                    label="Use Re-ranking",
                    value=False,
                    info="Improves context relevance by using Maximal Marginal Relevance"
                )
            
            # Search button
            search_button = gr.Button("Search", variant="primary")
        
        with gr.Column(scale=4):
            # Results output
            results_output = gr.Markdown(label="Search Results")
    
    # Connect the search button to the search function
    search_button.click(
        fn=lambda query, search_type, k, rerank: search_code(
            query,
            "qa" if search_type == "Question Answering" else "similarity",
            k,
            rerank=rerank
        ),
        inputs=[query_input, search_type, num_results, use_reranking],
        outputs=results_output
    )
    
    gr.Markdown("## How to use this tool")
    gr.Markdown("""
    - **Question Answering**: Ask natural language questions about your codebase like "How does the document indexing work?"
    - **Code Retrieval**: Find specific code snippets with queries like "function to load documents" or "error handling in vectorstore"
    - **Re-ranking**: Enable this option to improve context relevance for complex queries
    """)

if __name__ == "__main__":
    demo.launch()