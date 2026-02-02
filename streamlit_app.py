#!/usr/bin/env python3
"""
Streamlit frontend for PageIndex - Document analysis and chat interface
"""
import os
import io
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, List
from dotenv import load_dotenv

import streamlit as st
from streamlit import runtime

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="PageIndex",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1f77b4;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Import PageIndex modules
from pageindex.utils import ChatGPT_API_async, resolve_model_name
from pageindex.page_index import page_index_main
from pageindex.page_index_md import page_index_main_md
from pageindex.utils import ConfigLoader
from pageindex.cost_tracker import get_global_tracker, reset_global_tracker

# Initialize session state
if "document_structure" not in st.session_state:
    st.session_state.document_structure = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "cost_summary" not in st.session_state:
    st.session_state.cost_summary = None


def get_available_models() -> Dict[str, str]:
    """Get available models from environment variables"""
    models = {}
    if os.getenv("MODEL_FREE"):
        models["MODEL_FREE"] = os.getenv("MODEL_FREE")
    if os.getenv("MODEL_FAST"):
        models["MODEL_FAST"] = os.getenv("MODEL_FAST")
    if os.getenv("MODEL_REASONING"):
        models["MODEL_REASONING"] = os.getenv("MODEL_REASONING")

    # Add default models as fallback
    if not models:
        models = {
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
        }
    return models


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary location"""
    temp_dir = Path("data/uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)

    file_path = temp_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)


def format_structure_for_chat(structure: dict) -> str:
    """Format document structure for chat context"""
    def format_node(node, depth=0):
        indent = "  " * depth
        result = f"{indent}- **{node.get('title', 'Untitled')}** (pages {node.get('start_index', '?')}-{node.get('end_index', '?')})\n"

        if 'summary' in node:
            result += f"{indent}  {node['summary'][:200]}...\n"

        if 'nodes' in node:
            for child in node['nodes']:
                result += format_node(child, depth + 1)
        return result

    formatted = "# Document Structure\n\n"
    for node in structure.get('structure', []):
        formatted += format_node(node)
    return formatted


async def analyze_document(file_path: str, model_var: str, doc_type: str) -> dict:
    """Analyze document using PageIndex"""
    try:
        config_loader = ConfigLoader()
        config = config_loader.load({
            "env_model_var": model_var,
            "if_add_node_id": "yes",
            "if_add_node_summary": "yes",
            "if_add_doc_description": "yes"
        })

        if doc_type == "pdf":
            result = await page_index_main(
                pdf_path=file_path,
                opt=config
            )
        else:  # markdown
            result = await page_index_main_md(
                md_path=file_path,
                model=config.model,
                if_thinning="no",
                summary_token_threshold=200,
                if_add_node_id="yes",
                if_add_node_summary="yes",
                if_add_doc_description="yes"
            )

        # Load the generated structure
        result_file = Path("results") / f"{Path(file_path).stem}_structure.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        st.error(f"Error analyzing document: {e}")
        return None


async def chat_with_document(query: str, model_name: str, structure: dict) -> str:
    """Chat with the analyzed document"""
    # Format structure for context
    doc_context = format_structure_for_chat(structure)

    prompt = f"""You are a helpful assistant that can answer questions about a document that has been analyzed.

Document Information:
{doc_context}

Document Description:
{structure.get('doc_description', 'No description available')}

User Question: {query}

Please provide a comprehensive answer based on the document structure and content above."""

    try:
        response = await ChatGPT_API_async(model=model_name, prompt=prompt)
        return response
    except Exception as e:
        return f"Error: {e}"


# Sidebar - Model Selection
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # Model selection
    available_models = get_available_models()
    model_options = list(available_models.keys())

    if model_options:
        selected_model_key = st.selectbox(
            "Select Model",
            options=model_options,
            index=0,
            help="Choose the AI model to use for analysis and chat"
        )
        st.session_state.selected_model = available_models[selected_model_key]
        st.caption(f"Using: {st.session_state.selected_model}")
    else:
        st.warning("No models found in environment. Please set MODEL_FREE, MODEL_FAST, or MODEL_REASONING in your .env file.")
        st.session_state.selected_model = "gpt-4o"

    st.markdown("---")

    # Document info
    if st.session_state.document_name:
        st.markdown("### 📄 Current Document")
        st.info(st.session_state.document_name)

        if st.session_state.document_structure:
            description = st.session_state.document_structure.get('doc_description', 'No description')
            st.caption(f"*{description[:150]}...*")

    st.markdown("---")

    # API status
    st.markdown("### 🔌 API Status")
    api_key = os.getenv("CHATGPT_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if api_key:
        st.success("✅ API key configured")
    else:
        st.error("❌ No API key found")

    st.markdown("---")

    # Cost tracking
    st.markdown("### 💰 API Costs")
    tracker = get_global_tracker()
    summary = tracker.get_summary()

    if summary['total_calls'] > 0:
        st.metric("Total Cost", f"${summary['total_cost']:.6f}")
        st.caption(f"{summary['total_calls']} calls • {summary['total_tokens']:,} tokens")

        # Show cost by model
        if summary['costs_by_model']:
            with st.expander("Costs by Model"):
                for model, cost in sorted(summary['costs_by_model'].items(), key=lambda x: -x[1]):
                    calls = summary['calls_by_model'][model]
                    st.write(f"**{model[:30]}...**: ${cost:.6f} ({calls} calls)")

        # Reset button
        if st.button("Reset Cost Tracker", key="reset_costs"):
            tracker.reset()
            st.rerun()
    else:
        st.caption("No API calls yet")

    st.markdown("---")
    st.markdown("### 📖 Usage")
    st.markdown("""
1. **Upload** a PDF or Markdown file
2. **Analyze** to extract structure
3. **Chat** with your document
    """)


# Main content
st.markdown('<h1 class="main-header">📄 PageIndex</h1>', unsafe_allow_html=True)
st.markdown("*Vectorless, reasoning-based document analysis and chat*")

# Upload section
st.markdown('<h2 class="section-header">📤 Upload Document</h2>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a PDF or Markdown file",
        type=['pdf', 'md'],
        help="Upload a document to analyze"
    )

with col2:
    doc_type = st.radio(
        "Document Type",
        ["PDF", "Markdown"],
        horizontal=True
    )

if uploaded_file:
    # Check if it's a different file
    if st.session_state.document_name != uploaded_file.name:
        with st.spinner("Saving uploaded file..."):
            file_path = save_uploaded_file(uploaded_file)
            st.session_state.document_name = uploaded_file.name
            st.session_state.document_structure = None
            st.session_state.chat_history = []
            st.success(f"File saved: {uploaded_file.name}")

        # Auto-analyze button
        st.markdown("---")
        st.markdown('<h2 class="section-header">🔍 Analyze Document</h2>', unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 3])

        with col_a:
            analyze_button = st.button("⚡ Analyze", type="primary", use_container_width=True)

        with col_b:
            st.caption("Click to extract document structure and generate summaries")

        if analyze_button:
            if not st.session_state.selected_model:
                st.error("Please select a model first")
            else:
                with st.spinner(f"Analyzing document with {st.session_state.selected_model}..."):
                    doc_type_param = "pdf" if doc_type == "PDF" else "md"
                    model_var = list(available_models.keys())[model_options.index(selected_model_key)] if model_options else None

                    # Run analysis
                    loop = runtime.scriptrunner.add_script_run_ctx.get_script_run_ctx()._asyncio_loop
                    if loop is None:
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    structure = loop.run_until_complete(
                        analyze_document(file_path, model_var, doc_type_param)
                    )

                    if structure:
                        st.session_state.document_structure = structure
                        st.success("✅ Document analyzed successfully!")

                        # Show cost summary for this analysis
                        tracker = get_global_tracker()
                        st.info(f"💰 Analysis Cost: ${tracker.get_total_cost():.6f} ({tracker.get_total_tokens()[2]:,} tokens)")

                        st.rerun()
                    else:
                        st.error("Failed to analyze document")

# Display document structure if available
if st.session_state.document_structure:
    st.markdown("---")
    st.markdown('<h2 class="section-header">📊 Document Structure</h2>', unsafe_allow_html=True)

    structure = st.session_state.document_structure

    # Document description
    if 'doc_description' in structure:
        st.info(f"**Description:** {structure['doc_description']}")

    # Structure tree
    def display_structure(nodes, depth=0):
        for node in nodes:
            indent = "📁 " + "&nbsp;&nbsp;&nbsp;" * depth
            title = node.get('title', 'Untitled')
            page_range = f"({node.get('start_index', '?')}-{node.get('end_index', '?')})"

            if depth == 0:
                st.markdown(f"### {indent}**{title}** {page_range}")
            else:
                st.markdown(f"{indent}**{title}** {page_range}")

            # Show summary if available
            if 'summary' in node and st.checkbox(f"Show summary: {title}", key=f"summary_{node.get('node_id')}"):
                st.caption(node['summary'])

            # Recursively display child nodes
            if 'nodes' in node:
                display_structure(node['nodes'], depth + 1)

    display_structure(structure.get('structure', []))

    # Chat section
    st.markdown("---")
    st.markdown('<h2 class="section-header">💬 Chat with Document</h2>', unsafe_allow_html=True)

    # Display chat history
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>',
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {message["content"]}</div>',
                          unsafe_allow_html=True)

    # Chat input
    st.markdown("---")
    user_input = st.text_input(
        "Ask a question about your document:",
        placeholder="e.g., What are the main topics discussed in this document?",
        key="chat_input"
    )

    col_send, col_clear = st.columns([1, 5])

    with col_send:
        send_button = st.button("Send 📨", type="primary", use_container_width=True)

    with col_clear:
        if st.button("Clear Chat History 🗑️"):
            st.session_state.chat_history = []
            st.rerun()

    if send_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        # Get assistant response
        with st.spinner("Thinking..."):
            loop = runtime.scriptrunner.add_script_run_ctx.get_script_run_ctx()._asyncio_loop
            if loop is None:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            response = loop.run_until_complete(
                chat_with_document(user_input, st.session_state.selected_model, structure)
            )

        # Add assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })

        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<center style="color: gray; font-size: 0.8rem;">
    <b>PageIndex</b> - Vectorless, reasoning-based RAG system<br>
    Powered by OpenAI/OpenRouter API
</center>
""", unsafe_allow_html=True)
