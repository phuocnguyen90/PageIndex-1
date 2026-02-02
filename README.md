<div align="center">

**Phuoc's PageIndex**

*An enhanced reasoning-based RAG system for document analysis*

<a href="https://github.com/phuocnguyen90/pageindex" target="_blank">
  <img src="https://img.shields.io/badge/GitHub-phuocnguyen90%2Fpageindex-blue?style=for-the-badge&logo=github" alt="GitHub" />
</a>

<br/>
<br/>


</div>

# PageIndex: Vectorless, Reasoning-based RAG


<p align="left"><b>Reasoning-based RAG&nbsp; ◦ &nbsp;No Vector DB&nbsp; ◦ &nbsp;No Chunking&nbsp; ◦&nbsp;Human-like Retrieval</b></p>

---

## 📝 Introduction

Welcome to my enhanced version of PageIndex - a vectorless, reasoning-based RAG system designed for superior document retrieval. As a developer and AI enthusiast, I've built this tool to address the fundamental limitation of traditional vector-based RAG systems: **similarity ≠ relevance**.

Traditional RAG systems rely on semantic similarity, but for complex professional documents, what we really need is true relevance - which requires reasoning. That's why I've enhanced the original PageIndex concept to create a system that simulates how human experts navigate through complex documents.

This system builds a hierarchical tree index from documents and uses LLMs to reason over that index, enabling agentic, context-aware retrieval. Instead of mindless vector similarity search, PageIndex enables LLMs to think and reason their way to the most relevant content.

The retrieval process works in two elegant steps:

1. **Tree Generation**: Create a hierarchical "Table-of-Contents" index of the entire document
2. **Reasoning-based Search**: Navigate the tree using LLM reasoning to find the most relevant sections

### 🚀 My Key Enhancements

Building upon the original concept, I've added several features to improve usability and developer experience:

- **💰 Cost Tracking**: Monitor API usage with automatic token counting and cost calculation
- **🖥️ Streamlit Web Interface**: User-friendly web interface for document analysis and querying
- **🔧 Modular Architecture**: Clean, maintainable codebase with focused modules
- **🧪 Comprehensive Testing**: Full pytest suite with shared fixtures and API test markers
- **🌐 Multi-Provider Support**: Flexible API provider switching (OpenAI, OpenRouter, and more)
- **📊 Environment Configuration**: Easy model selection via environment variables

### 🎯 About the Original Project

This project is based on the groundbreaking PageIndex concept originally created by [Vectify AI](https://vectify.ai). The original project introduced a revolutionary approach to RAG by replacing vector similarity with reasoning-based document navigation.

For more about the original concept:
- [Vectify AI Homepage](https://vectify.ai)
- [Original PageIndex Repository](https://github.com/VectifyAI/PageIndex)
- [Documentation](https://docs.pageindex.ai)

### 🎯 Core Features

Compared to traditional vector-based RAG, **PageIndex** features:
- **No Vector DB**: Uses document structure and LLM reasoning for retrieval, instead of vector similarity search.
- **No Chunking**: Documents are organized into natural sections, not artificial chunks.
- **Human-like Retrieval**: Simulates how human experts navigate and extract knowledge from complex documents.
- **Better Explainability and Traceability**: Retrieval is based on reasoning — traceable and interpretable, with page and section references.

---

## 🚀 Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/phuocnguyen90/pageindex.git
cd pageindex

# Install dependencies
pip3 install --upgrade -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the root directory:

```bash
# OpenAI API
CHATGPT_API_KEY=your_openai_key_here

# OR OpenRouter API (recommended for cost savings)
OPENROUTER_API_KEY=your_openrouter_key_here
```

### 3. Usage

**Basic PDF processing:**
```bash
python3 run_pageindex.py --pdf_path /path/to/document.pdf
```

**With environment variable model selection:**
```bash
python3 run_pageindex.py --pdf_path document.pdf \
  --env-model-var MODEL_FREE
```

**Launch Streamlit web interface:**
```bash
streamlit run streamlit_app.py
```

**Run tests:**
```bash
# Run all tests
pytest

# Run without API calls (faster)
pytest -m "not api"

# Run specific test
pytest tests/test_api.py::TestAPI::test_resolve_model_name
```

---

## 🌟 Advanced Features

### 💰 Cost Tracking
Monitor your API usage with comprehensive tracking:
- Input/output token counting for all API calls
- Real-time cost calculation based on model pricing
- Detailed summaries by model type
- Persistent tracking data that can be saved and loaded

```python
from pageindex.cost_tracker import get_global_tracker

tracker = get_global_tracker()
summary = tracker.get_summary()
print(f"Total API cost: ${summary['total_cost']:.6f}")
```

### 🖥️ Streamlit Web Interface
A modern web interface for interactive document analysis:
- Drag-and-drop document upload
- Visual display of hierarchical document structures
- Chat interface for intelligent document querying
- Real-time cost monitoring
- Easy model selection through environment variables

### 🔧 Clean Architecture
I've refactored the codebase into focused, maintainable modules:
- `api_client.py` - Multi-provider API client
- `cost_tracker.py` - Comprehensive usage tracking
- `token_utils.py` - Token counting and management
- `document_utils.py` - PDF/Markdown processing
- `config.py` - Flexible configuration management

---

## 📁 Project Structure

```
PageIndex/
├── pageindex/               # Core package
│   ├── api_client.py       # Multi-provider API client
│   ├── cost_tracker.py      # API usage and cost tracking
│   ├── token_utils.py       # Token counting utilities
│   ├── document_utils.py    # PDF and Markdown processing
│   ├── config.py            # Configuration management
│   ├── page_index.py       # PDF document processing
│   ├── page_index_md.py    # Markdown document processing
│   ├── utils.py             # Legacy compatibility module
│   └── config.yaml           # Default configuration
├── run_pageindex.py       # Command-line interface
├── streamlit_app.py          # Web interface application
├── tests/                   # Comprehensive test suite
│   ├── conftest.py          # Pytest fixtures and configuration
│   ├── test_api.py          # API integration tests
│   └── ...                  # Other test files
├── cookbook/                 # Example notebooks and tutorials
├── data/                    # Document processing cache
└── results/                 # Generated document structures
```

---

## ⚙️ Configuration

### API Configuration

Configure your preferred AI models and providers:

```bash
--model                 # Default model to use (e.g., gpt-4o)
--api-provider          # API provider (openai, openrouter)
--api-base-url          # Custom API endpoint URL
--model-name            # Provider-specific model name
--env-model-var          # Environment variable reference
```

### Document Processing

Fine-tune how documents are processed:

```bash
--toc-check-pages       # Pages to check for table of contents
--max-pages-per-node    # Maximum pages per node in tree
--max-tokens-per-node   # Maximum tokens per node
```

### Output Control

Customize the generated output:

```bash
--if-add-node-id        # Include unique IDs for each node
--if-add-node-summary   # Generate content summaries
--if-add-doc-description # Add overall document description
--if-add-node-text       # Include full text content
```

---

## 📖 Developer Resources

### Test Suite

I've implemented a comprehensive test suite using pytest:

| Test Category | Purpose |
|--------------|---------|
| `tests/test_api.py` | API integration and model resolution |
| `tests/test_env_model_resolution.py` | Environment variable configuration |
| `tests/test_openrouter.py` | OpenRouter provider integration |
| `tests/test_pageindex_integration.py` | End-to-end processing tests |
| `tests/test_real_api.py` | Real API endpoint validation |

### Running Tests

```bash
# Run all tests
pytest

# Skip API calls (faster for local development)
pytest -m "not api"

# Run specific test file
pytest tests/test_api.py

# Verbose output with coverage
pytest -v --cov=pageindex
```

---

## 🙏 Acknowledgments

This project builds upon the brilliant work of [Vectify AI](https://vectify.ai) and their original [PageIndex](https://github.com/VectifyAI/PageIndex) project. The core concept of using reasoning instead of vector similarity for document retrieval is truly innovative.

Special thanks to the Vectify AI team for open-sourcing their groundbreaking work. Without their original insights into the limitations of vector-based RAG, this project wouldn't exist.

### Inspiration and Credits

The original PageIndex was created to address a fundamental truth in information retrieval: **similarity ≠ relevance**. By replacing mindless vector similarity search with thoughtful LLM reasoning over hierarchical document structures, it enables a more human-like approach to document understanding.

My enhancements to this foundation include:
- Cost-aware API usage monitoring
- Accessible web interface
- Clean, maintainable architecture
- Comprehensive testing framework
- Flexible multi-provider support

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The original PageIndex project by Vectify AI is also licensed under the MIT License.

## 🤝 Contributing

I welcome contributions! Whether it's bug fixes, new features, or documentation improvements, feel free to open issues or submit pull requests.

## 📧 Contact

**Project Maintainer:** Phuoc Nguyen

For questions, suggestions, or collaboration:
- Open an issue on this repository
- Check out my other projects on [GitHub](https://github.com/phuocnguyen90)

For the original PageIndex concept, visit [Vectify AI](https://vectify.ai).

