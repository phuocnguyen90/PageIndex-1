#!/usr/bin/env python3
"""
Streamlit frontend for PageIndex - Document analysis and chat interface
"""
import os
import io
import json
import asyncio
import logging
import threading
import concurrent.futures
from pathlib import Path
from typing import Optional, Dict, List
from dotenv import load_dotenv

import streamlit as st
from streamlit import runtime

# Load environment variables
load_dotenv()

# Setup logging
def setup_logging():
    """Setup logging for Streamlit app and PageIndex modules"""
    # Create root logger to capture all module logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if they already exist
    if not root_logger.handlers:
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler (stderr)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Stdout handler (for some environments where stderr isn't shown)
        import sys
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        root_logger.addHandler(stdout_handler)

        # File handler
        file_handler = logging.FileHandler('streamlit_app.log')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Suppress verbose logs from third-party libraries
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Return the app-specific logger for convenience
    return logging.getLogger("streamlit_app")

# Streamlit log handler for real-time display
class StreamlitLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_messages = []

    def emit(self, record):
        log_entry = self.format(record)
        self.log_messages.append(log_entry)
        # Keep only last 50 messages
        if len(self.log_messages) > 50:
            self.log_messages = self.log_messages[-50:]

# Initialize logging
if 'streamlit_handler' not in st.session_state:
    st.session_state.streamlit_handler = StreamlitLogHandler()
    st.session_state.streamlit_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger = setup_logging()
streamlit_handler = st.session_state.streamlit_handler

# Ensure streamlit_handler is added to root logger if not already present
root_logger = logging.getLogger()
if streamlit_handler not in root_logger.handlers:
    root_logger.addHandler(streamlit_handler)

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
from pageindex.page_index_md import md_to_tree
from pageindex.utils import ConfigLoader
from pageindex.cost_tracker import get_global_tracker, reset_global_tracker

# Initialize session state
if "document_structure" not in st.session_state:
    st.session_state.document_structure = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "document_path" not in st.session_state:
    st.session_state.document_path = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "selected_model_key" not in st.session_state:
    st.session_state.selected_model_key = None
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
    logger.info(f"Saving file: {uploaded_file.name}")
    temp_dir = Path("data/uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using temp directory: {temp_dir}")

    file_path = temp_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logger.info(f"File saved to: {file_path}")
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


def run_async(coro):
    """Helper function to run async code in Streamlit safely"""
    import threading
    import concurrent.futures

    try:
        # If we are already in an event loop, we need to run in a separate thread
        try:
            loop = asyncio.get_running_loop()
            
            result_container = []
            def run_in_new_loop():
                # Force a new event loop for this thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result = new_loop.run_until_complete(coro)
                    result_container.append(result)
                except Exception as e:
                    logger.error(f"Error in async thread: {e}", exc_info=True)
                    result_container.append(e)
                finally:
                    new_loop.close()

            thread = threading.Thread(target=run_in_new_loop)
            thread.start()
            thread.join()
            
            if result_container and isinstance(result_container[0], Exception):
                raise result_container[0]
            return result_container[0] if result_container else None
            
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(coro)

    except Exception as e:
        logger.error(f"Async execution failed: {e}", exc_info=True)
        raise e

def analyze_document(file_path: str, model_var: str, doc_type: str) -> dict:
    """Analyze document using PageIndex"""
    try:
        logger.info(f"Starting analysis for document: {file_path}")
        logger.info(f"Document type: {doc_type}, Model variable: {model_var}")

        st.info(f"🔍 Starting analysis for: {os.path.basename(file_path)}")

        config_loader = ConfigLoader()
        config = config_loader.load({
            "env_model_var": model_var,
            "if_add_node_id": "yes",
            "if_add_node_summary": "yes",
            "if_add_doc_description": "yes"
        })

        # Resolve the actual model name (e.g., from MODEL_FREE)
        config.model = resolve_model_name(
            model=config.model,
            api_provider=getattr(config, 'api_provider', None),
            model_name=getattr(config, 'model_name', None),
            env_model_var=getattr(config, 'env_model_var', None)
        )

        logger.info(f"Configuration loaded: Model={config.model}, Provider={getattr(config, 'api_provider', 'openai')}")

        if doc_type == "pdf":
            logger.info("Processing PDF document...")
            st.info("📄 Processing PDF document...")
            
            # Use run_async to wrap the synchronous page_index_main if it uses asyncio.run
            # to avoid event loop conflicts in Streamlit
            def sync_wrapper():
                return page_index_main(file_path, opt=config)
            
            # Since page_index_main calls asyncio.run, we use a thread pool to avoid loop conflicts
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(sync_wrapper)
                result = future.result()
                
            logger.info("PDF processing completed")
            st.success("✅ PDF processing completed!")
        else:  # markdown
            logger.info("Processing Markdown document...")
            st.info("📝 Processing Markdown document...")
            # md_to_tree is async, need to run it properly
            result = run_async(md_to_tree(
                md_path=file_path,
                if_thinning=False,
                min_token_threshold=5000,
                if_add_node_summary=getattr(config, 'if_add_node_summary', 'yes') == 'yes',
                summary_token_threshold=200,
                model=config.model,
                if_add_doc_description=getattr(config, 'if_add_doc_description', 'no') == 'yes',
                if_add_node_text=getattr(config, 'if_add_node_text', 'no') == 'no',
                if_add_node_id=getattr(config, 'if_add_node_id', 'yes') == 'yes'
            ))
            logger.info("Markdown processing completed")
            st.success("✅ Markdown processing completed!")

        if result:
            # Save the results to the results folder
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            result_file = results_dir / f"{Path(file_path).stem}_structure.json"
            
            logger.info(f"Saving results to: {result_file}")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            st.success(f"✅ Analysis complete! Found {len(result.get('structure', []))} main sections.")
            return result
        else:
            logger.error("Analysis failed to produce a result")
            st.error("❌ Analysis failed to produce a result.")
            return None

    except Exception as e:
        logger.error(f"Error analyzing document: {e}", exc_info=True)
        st.error(f"❌ Error analyzing document: {e}")
        return None


async def chat_with_document(query: str, model_name: str, structure: dict) -> str:
    """Chat with the analyzed document"""
    logger.info(f"Chat query: {query[:100]}... using model: {model_name}")

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
        logger.info("Sending query to AI model...")
        response = await ChatGPT_API_async(model=model_name, prompt=prompt)
        logger.info("AI response received successfully")
        return response
    except Exception as e:
        logger.error(f"Error in chat: {e}", exc_info=True)
        return f"Error: {e}"


# Sidebar - Model Selection
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # Model selection
    logger.info("Getting available models...")
    available_models = get_available_models()
    model_options = list(available_models.keys())
    logger.info(f"Available models: {model_options}")

    if model_options:
        selected_model_key = st.selectbox(
            "Select Model",
            options=model_options,
            index=0,
            help="Choose the AI model to use for analysis and chat"
        )
        st.session_state.selected_model_key = selected_model_key
        st.session_state.selected_model = available_models[selected_model_key]
        logger.info(f"Selected model: {st.session_state.selected_model}")
        st.caption(f"Using: {st.session_state.selected_model}")
    else:
        logger.warning("No models found in environment")
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
    logger.info(f"File uploaded: {uploaded_file.name}")
    # Check if it's a different file
    if st.session_state.document_name != uploaded_file.name:
        logger.info("New file detected, saving...")
        with st.spinner("Saving uploaded file..."):
            file_path = save_uploaded_file(uploaded_file)
            logger.info(f"File saved to: {file_path}")
            st.session_state.document_name = uploaded_file.name
            st.session_state.document_path = file_path  # Store path for analysis
            st.session_state.document_structure = None
            st.session_state.chat_history = []
            st.success(f"File saved: {uploaded_file.name}")

    # Auto-analyze button
    st.markdown("---")
    st.markdown('<h2 class="section-header">🔍 Analyze Document</h2>', unsafe_allow_html=True)

    with st.form("analyze_form"):
        col_a, col_b = st.columns([1, 3])

        with col_a:
            analyze_button = st.form_submit_button("⚡ Analyze", type="primary", use_container_width=True)

        with col_b:
            st.caption("Click to extract document structure and generate summaries")

        if analyze_button:
            logger.info("Analyze button clicked!")
            if not st.session_state.selected_model:
                st.error("Please select a model first")
            else:
                logger.info("Analyze button clicked by user")
                with st.spinner(f"🚀 Analyzing document with {st.session_state.selected_model}..."):
                    doc_type_param = "pdf" if doc_type == "PDF" else "md"
                    model_var = st.session_state.get('selected_model_key', 'MODEL_FREE')
                    logger.info(f"Starting analysis with model_var: {model_var}")

                    # Run analysis (analyze_document is now synchronous)
                    logger.info("Running analyze_document function...")
                    structure = analyze_document(st.session_state.document_path, model_var, doc_type_param)
                    logger.info("Analysis function completed")

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
    with st.form("chat_form"):
        col_send, col_clear = st.columns([1, 5])

        with col_send:
            send_button = st.form_submit_button("Send 📨", type="primary", use_container_width=True)

        with col_clear:
            clear_button = st.form_submit_button("Clear Chat History 🗑️")

        # Text input
        user_input = st.text_input(
            "Ask a question about your document:",
            placeholder="e.g., What are the main topics discussed in this document?",
            key="chat_input"
        )

        if clear_button:
            st.session_state.chat_history = []
            st.rerun()

        if send_button and user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            logger.info(f"User sent chat message: {user_input[:100]}...")

            # Get assistant response
            with st.spinner("🤔 Thinking..."):
                logger.info("Calling chat_with_document...")
                response = run_async(
                    chat_with_document(user_input, st.session_state.selected_model, structure)
                )
                logger.info("Chat response received")

            # Add assistant response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })

            st.rerun()

# Real-time logging section
with st.expander("📊 View Application Logs", expanded=False):
    st.markdown("### Recent Log Messages")

    # Display recent logs
    if hasattr(streamlit_handler, 'log_messages'):
        for log_msg in streamlit_handler.log_messages[-20:]:  # Show last 20 messages
            if 'ERROR' in log_msg:
                st.error(log_msg)
            elif 'WARNING' in log_msg:
                st.warning(log_msg)
            elif 'INFO' in log_msg:
                st.info(f"📝 {log_msg}")
            else:
                st.caption(f"⚪ {log_msg}")
    else:
        st.info("✅ Logging is active. Logs will appear here as you interact with the app.")

# Footer
st.markdown("---")
st.markdown("""
<center style="color: gray; font-size: 0.8rem;">
    <b>PageIndex</b> - Vectorless, reasoning-based RAG system<br>
    Powered by OpenAI/OpenRouter API
</center>
""", unsafe_allow_html=True)
