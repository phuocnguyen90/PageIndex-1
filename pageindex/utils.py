"""
PageIndex utilities module.

This module provides backward compatibility by re-exporting functions from
the new refactored modules:

- api_client: API client functions (ChatGPT_API, etc.)
- cost_tracker: Cost tracking for API calls
- token_utils: Token counting utilities
- document_utils: Document processing utilities
- config: Configuration management

New code should import directly from the specific modules:
    from pageindex.api_client import ChatGPT_API_async
    from pageindex.cost_tracker import get_global_tracker
    from pageindex.token_utils import count_tokens
    from pageindex.document_utils import structure_to_list
"""

import logging
from dotenv import load_dotenv

load_dotenv()

# Import all functions from refactored modules for backward compatibility
from .api_client import (
    ChatGPT_API,
    ChatGPT_API_async,
    ChatGPT_API_with_finish_reason,
    make_api_call,
    resolve_model_name,
    _create_client,
    _get_model_from_env,
)

from .cost_tracker import (
    CostTracker,
    get_global_tracker,
    reset_global_tracker,
)

from .token_utils import (
    count_tokens,
    check_token_limit,
    estimate_tokens_from_pages,
    estimate_tokens_from_text_length,
)

from .document_utils import (
    # JSON utilities
    get_json_content,
    extract_json,

    # Node utilities
    write_node_id,
    get_nodes,
    structure_to_list,
    get_leaf_nodes,
    is_leaf_node,
    get_last_node,

    # PDF utilities
    extract_text_from_pdf,
    get_pdf_title,
    get_text_of_pages,
    get_first_start_page_from_text,
    get_last_start_page_from_text,
    sanitize_filename,
    get_pdf_name,
    get_page_tokens,
    get_text_of_pdf_pages,
    get_text_of_pdf_pages_with_labels,
    get_number_of_pages,
    add_node_text,
    add_node_text_with_labels,

    # Structure utilities
    list_to_tree,
    add_preface_if_needed,
    post_processing,
    clean_structure_post,
    remove_fields,
    remove_structure_text,
    convert_physical_index_to_int,
    convert_page_to_int,
    reorder_dict,
    format_structure,

    # Display utilities
    print_toc,
    print_json,

    # Logger
    JsonLogger,

    # Summary generation
    generate_node_summary,
    generate_summaries_for_structure,
    create_clean_structure_for_description,
    generate_doc_description,
)

# Import ConfigLoader
from .config import ConfigLoader

# Module-level exports for backward compatibility
__all__ = [
    # API Client
    'ChatGPT_API',
    'ChatGPT_API_async',
    'ChatGPT_API_with_finish_reason',
    'make_api_call',
    'resolve_model_name',
    '_create_client',
    '_get_model_from_env',

    # Cost Tracking
    'count_tokens',
    'CostTracker',
    'get_global_tracker',
    'reset_global_tracker',

    # Token Utilities
    'check_token_limit',
    'estimate_tokens_from_pages',
    'estimate_tokens_from_text_length',

    # Document Utilities
    'get_json_content',
    'extract_json',
    'write_node_id',
    'get_nodes',
    'structure_to_list',
    'get_leaf_nodes',
    'is_leaf_node',
    'get_last_node',
    'extract_text_from_pdf',
    'get_pdf_title',
    'get_text_of_pages',
    'get_first_start_page_from_text',
    'get_last_start_page_from_text',
    'sanitize_filename',
    'get_pdf_name',
    'get_page_tokens',
    'get_text_of_pdf_pages',
    'get_text_of_pdf_pages_with_labels',
    'get_number_of_pages',
    'add_node_text',
    'add_node_text_with_labels',
    'list_to_tree',
    'add_preface_if_needed',
    'post_processing',
    'clean_structure_post',
    'remove_fields',
    'remove_structure_text',
    'convert_physical_index_to_int',
    'convert_page_to_int',
    'reorder_dict',
    'format_structure',
    'print_toc',
    'print_json',
    'JsonLogger',
    'generate_node_summary',
    'generate_summaries_for_structure',
    'create_clean_structure_for_description',
    'generate_doc_description',

    # Configuration
    'ConfigLoader',
]
