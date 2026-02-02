# OpenRouter Integration for PageIndex

This document describes the OpenRouter integration that has been added to PageIndex, enabling support for multiple LLM providers and flexible model selection.

## Overview

PageIndex now supports multiple LLM providers beyond just OpenAI:

- **OpenAI**: The original provider with access to GPT models
- **OpenRouter**: An open-source provider with access to 100+ models from various providers
- **Custom APIs**: Any OpenAI-compatible API endpoint

## Key Features

### 1. Multi-Provider Support
- Seamless switching between OpenAI and OpenRouter
- Automatic model name resolution for OpenRouter
- Support for custom API endpoints

### 2. Flexible Model Selection
- Use OpenAI models (gpt-4o, gpt-4o-mini, etc.)
- Use OpenRouter models (Anthropic Claude, Google Gemini, Meta Llama, etc.)
- Specify provider-specific model names

### 3. Backward Compatibility
- Existing code continues to work without changes
- Default behavior remains the same (OpenAI)

## Configuration

### Environment Variables

```bash
# For OpenAI (default)
CHATGPT_API_KEY=your_openai_key_here

# For OpenRouter
OPENROUTER_API_KEY=your_openrouter_key_here

# Optional: Custom API base URL
API_BASE_URL=https://your-custom-api.com/v1
```

### Command Line Options

```bash
# Default OpenAI
python3 run_pageindex.py --pdf_path document.pdf

# With OpenRouter
python3 run_pageindex.py --pdf_path document.pdf \
  --api-provider openrouter \
  --model claude-3-5-sonnet-20241022

# With custom API endpoint
python3 run_pageindex.py --pdf_path document.pdf \
  --api-base-url https://your-custom-api.com/v1 \
  --model your-custom-model
```

## Model Name Resolution

### OpenAI Models
- `gpt-4o` → `gpt-4o`
- `gpt-4o-mini` → `gpt-4o-mini`

### OpenRouter Models
- `claude-3-5-sonnet-20241022` → `anthropic/claude-3-5-sonnet-20241022`
- `gpt-4o` → `openai/gpt-4o`
- `llama-3.1-70b-instruct` → `meta-llama/llama-3.1-70b-instruct`

### Custom Models
- Use `--model-name` to override model resolution
- Example: `--model-name "custom/model"`

## Available Models

### OpenRouter Models (Examples)
- **Anthropic**: `anthropic/claude-3.5-sonnet`, `anthropic/claude-haiku-4.5`
- **Google**: `google/gemini-3-flash-preview`, `google/gemini-3-pro`
- **Meta**: `meta-llama/llama-3.1-70b-instruct`, `meta-llama/llama-3.1-405b`
- **Mistral**: `mistralai/mistral-large`, `mistralai/mistral-small`
- **DeepSeek**: `deepseek/deepseek-v3.2`, `deepseek/deepseek-chat`

### Model Selection Guide
1. **Cost-effective**: `anthropic/claude-haiku-4.5`, `google/gemini-3-flash`
2. **High performance**: `anthropic/claude-3.5-sonnet`, `google/gemini-3-pro`
3. **Coding**: `meta-llama/llama-3.1-coder`, `qwen/qwen3-coder`
4. **Reasoning**: `deepseek/deepseek-v3.2`, `x-ai/grok-4`

## Implementation Details

### Client Creation Logic
The `_create_client()` function determines the appropriate client based on:
1. Custom base URL (if specified)
2. Explicit OpenRouter API key
3. Available API keys in environment
4. Default to OpenAI

### Model Resolution
The `resolve_model_name()` function handles:
- Provider-specific model name prefixes
- Explicit model name overrides
- Default model passthrough

### API Function Wrappers
All API functions (`ChatGPT_API`, `ChatGPT_API_async`, `ChatGPT_API_with_finish_reason`) now support:
- Different providers
- Custom base URLs
- Flexible model selection

## Testing

Run the test suite to verify the integration:

```bash
python3 test_openrouter.py
python3 test_real_api.py
python3 test_pageindex_integration.py
```

## Migration Guide

### For Existing Users
No changes required. Your existing code will continue to work with OpenAI.

### For New Users
1. Set up your API keys
2. Choose your preferred provider
3. Select appropriate models
4. Run PageIndex as usual

## Troubleshooting

### Common Issues

1. **Invalid Model ID**
   - Check the model name is correct for your provider
   - Use `python3 -c "from openrouter import OpenRouter; client = OpenRouter(); print(client.models.list())"` to see available models

2. **API Key Issues**
   - Ensure your API key has the correct permissions
   - Check the key format (no extra spaces or newlines)

3. **Base URL Issues**
   - Verify the base URL is correct for your provider
   - Ensure the endpoint is accessible from your environment

### Debug Mode
Enable debug logging by setting:
```bash
export PYTHONPATH=/path/to/pageindex
python3 -m pageindex.utils 2>&1 | grep -A5 -B5 "Error"
```

## Future Enhancements

- Automatic model recommendation based on task type
- Dynamic model switching based on cost/performance requirements
- Support for additional providers (Anthropic, Cohere, etc.)
- Model benchmarking and performance metrics

## Contributing

To add support for new providers:
1. Update the `_create_client()` function
2. Add model name resolution for the new provider
3. Update documentation and tests
4. Add example configurations

## License

This integration is part of the PageIndex project and follows the same license terms.