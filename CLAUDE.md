# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PageIndex is a vectorless, reasoning-based RAG system that transforms long documents (PDFs and Markdown) into hierarchical tree structures for human-like document retrieval. Instead of using traditional vector databases and chunking, PageIndex uses LLM reasoning to navigate documents through tree search, simulating how human experts navigate complex documents.

## Core Architecture

### Main Components

1. **Document Processing Pipeline**
   - PDF processing: `pageindex/page_index.py` - Extracts text and identifies document structure
   - Markdown processing: `pageindex/page_index_md.py` - Processes markdown files using heading hierarchy
   - Both output JSON tree structures representing document hierarchy

2. **Tree Generation**
   - Uses OpenAI's GPT models to analyze document structure
   - Creates hierarchical nodes with page ranges, summaries, and optional text
   - Supports async processing for better performance

3. **Configuration Management**
   - Default config in `pageindex/config.yaml`
   - Override via command-line arguments
   - Environment variable: `CHATGPT_API_KEY` required in `.env`

### Key Features

- **No Vector Database**: Uses reasoning-based retrieval instead of embeddings
- **No Chunking**: Preserves natural document structure
- **Vision Support**: Can work directly with PDF page images (no OCR)
- **Human-like Navigation**: Tree-based search simulating expert document navigation
- **Multi-Provider Support**: Works with OpenAI, OpenRouter, and other OpenAI-compatible APIs
- **Flexible Model Selection**: Easy switching between different LLM providers and models

## Common Commands

### Setup
```bash
# Install dependencies
pip3 install --upgrade -r requirements.txt

# Set API key (create .env file)
echo "CHATGPT_API_KEY=your_openai_key_here" > .env
```

### Processing Documents

```bash
# Process PDF
python3 run_pageindex.py --pdf_path /path/to/document.pdf

# Process Markdown
python3 run_pageindex.py --md_path /path/to/document.md

# Custom parameters
python3 run_pageindex.py --pdf_path document.pdf \
  --model gpt-4o-2024-11-20 \
  --max-pages-per-node 5 \
  --max-tokens-per-node 15000 \
  --if-add-node-summary yes
```

### Output
Results are saved to `results/[filename]_structure.json` with hierarchical tree structure.

## Testing and Examples

### Test Documents
- Sample PDFs in `tests/pdfs/` (financial reports, academic papers, regulatory docs)
- Expected outputs in `tests/results/`
- Manual testing by comparing outputs against expected results

### Example Notebooks
- `cookbook/pageindex_RAG_simple.ipynb` - Basic vectorless RAG example
- `cookbook/vision_RAG_pageindex.ipynb` - Vision-based processing without OCR
- `cookbook/agentic_retrieval.ipynb` - Advanced retrieval examples

## Configuration Options

### API Configuration
- `--model`: Default model to use (default: gpt-4o-2024-11-20)
- `--api-provider`: API provider (openai, openrouter, default: openai)
- `--api-base-url`: Custom API base URL (for OpenRouter or other APIs)
- `--model-name`: Provider-specific model name (overrides --model)

### PDF Processing
- `--toc-check-pages`: Pages to check for table of contents (default: 20)
- `--max-pages-per-node`: Max pages per node (default: 10)
- `--max-tokens-per-node`: Max tokens per node (default: 20000)

### Markdown Processing
- `--if-thinning`: Apply tree thinning for large documents (default: no)
- `--thinning-threshold`: Min token threshold for thinning (default: 5000)
- `--summary-token-threshold`: Token threshold for summaries (default: 200)

### Output Options
- `--if-add-node-id`: Add unique IDs to nodes (default: yes)
- `--if-add-node-summary`: Generate summaries (default: yes)
- `--if-add-doc-description`: Add document description (default: no)
- `--if-add-node-text`: Include full text in output (default: no)

## Project Structure

```
PageIndex/
├── pageindex/               # Core package
│   ├── page_index.py       # PDF processing logic
│   ├── page_index_md.py    # Markdown processing
│   ├── utils.py           # Utilities and config
│   └── config.yaml        # Default configuration
├── run_pageindex.py       # CLI entry point
├── cookbook/             # Example notebooks
├── tests/                # Test documents and results
├── tutorials/            # Learning resources
└── requirements.txt      # Python dependencies
```

## Development Notes

- No formal testing framework - testing is manual with sample documents
- Uses async/await for concurrent API calls to OpenAI
- JSON-based logging for debugging and traceability
- Vision-based processing available but requires different setup (see cookbooks)
- Tree thinning optimization for large Markdown documents
- Results should match expected structures in `tests/results/`