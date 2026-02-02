"""
JSON Schemas for Structured Outputs in PageIndex.
"""

TOC_DETECTED_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "toc_detected",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "thinking": {"type": "string"},
                "toc_detected": {"type": "string", "enum": ["yes", "no"]}
            },
            "required": ["thinking", "toc_detected"],
            "additionalProperties": False
        }
    }
}

TOC_COMPLETION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "toc_completion",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "thinking": {"type": "string"},
                "completed": {"type": "string", "enum": ["yes", "no"]}
            },
            "required": ["thinking", "completed"],
            "additionalProperties": False
        }
    }
}

TOC_EXTRACTOR_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "toc_extractor",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "thinking": {"type": "string"},
                "page_index_given_in_toc": {"type": "string", "enum": ["yes", "no"]},
                "toc_content": {"type": "string"}
            },
            "required": ["thinking", "page_index_given_in_toc", "toc_content"],
            "additionalProperties": False
        }
    }
}

DETECT_PAGE_INDEX_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "detect_page_index",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "thinking": {"type": "string"},
                "starting_physical_index": {"type": ["integer", "null"]},
                "page_index_given_in_toc": {"type": "string", "enum": ["yes", "no"]}
            },
            "required": ["thinking", "starting_physical_index", "page_index_given_in_toc"],
            "additionalProperties": False
        }
    }
}

TOC_TRANSFORMER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "toc_transformer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "table_of_contents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "structure": {"type": ["string", "null"]},
                            "title": {"type": "string"},
                            "page": {"type": ["integer", "null"]}
                        },
                        "required": ["structure", "title", "page"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["table_of_contents"],
            "additionalProperties": False
        }
    }
}

CHECK_TITLE_APPEARANCE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "check_title_appearance",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "list_index": {"type": "integer"},
                "answer": {"type": "string", "enum": ["yes", "no"]},
                "title": {"type": "string"},
                "page_number": {"type": ["integer", "null"]}
            },
            "required": ["list_index", "answer", "title", "page_number"],
            "additionalProperties": False
        }
    }
}

CHECK_TITLE_APPEARANCE_IN_START_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "check_title_appearance_in_start",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "thinking": {"type": "string"},
                "start_begin": {"type": "string", "enum": ["yes", "no"]}
            },
            "required": ["thinking", "start_begin"],
            "additionalProperties": False
        }
    }
}

TOC_INDEX_EXTRACTOR_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "toc_index_extractor",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "thinking": {"type": "string"},
                "table_of_contents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "structure": {"type": ["string", "null"]},
                            "title": {"type": "string"},
                            "physical_index": {"type": ["string", "null"]}
                        },
                        "required": ["structure", "title", "physical_index"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["thinking", "table_of_contents"],
            "additionalProperties": False
        }
    }
}

FIX_INCORRECT_TOC_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fix_incorrect_toc",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "thinking": {"type": "string"},
                "fixed_toc_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "structure": {"type": ["string", "null"]},
                            "title": {"type": "string"},
                            "physical_index": {"type": ["integer", "string", "null"]}
                        },
                        "required": ["structure", "title", "physical_index"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["thinking", "fixed_toc_items"],
            "additionalProperties": False
        }
    }
}

SINGLE_TOC_ITEM_INDEX_FIXER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "single_toc_item_index_fixer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "thinking": {"type": "string"},
                "physical_index": {"type": ["string", "integer", "null"]}
            },
            "required": ["thinking", "physical_index"],
            "additionalProperties": False
        }
    }
}

GENERATE_TOC_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "generate_toc",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "thinking": {"type": "string"},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "structure": {"type": "string"},
                            "title": {"type": "string"},
                            "physical_index": {"type": "string"}
                        },
                        "required": ["structure", "title", "physical_index"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["thinking", "sections"],
            "additionalProperties": False
        }
    }
}

ADD_PAGE_NUMBER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "add_page_number",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "thinking": {"type": "string"},
                "toc_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "structure": {"type": ["string", "null"]},
                            "title": {"type": "string"},
                            "start": {"type": "string", "enum": ["yes", "no"]},
                            "physical_index": {"type": ["string", "null"]}
                        },
                        "required": ["structure", "title", "start", "physical_index"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["thinking", "toc_items"],
            "additionalProperties": False
        }
    }
}
