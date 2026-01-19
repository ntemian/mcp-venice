# Venice AI MCP Server

MCP server providing Claude Code access to [Venice AI](https://venice.ai) - a privacy-focused AI platform with uncensored models.

## Features

- **Privacy-first**: No data storage or logging
- **Uncensored models**: No content restrictions
- **Web search**: Real-time information retrieval
- **Image generation**: Flux model support
- **OpenAI-compatible**: Standard API format

## Available Tools

| Tool | Description |
|------|-------------|
| `venice_chat` | General conversation with optional web search |
| `venice_complete` | Simple one-shot completion |
| `venice_uncensored` | Chat with no content restrictions (dolphin-mistral) |
| `venice_search` | Query with web search enabled |
| `venice_reason` | DeepSeek R1 reasoning (step-by-step) |
| `venice_code` | Code generation with Qwen Coder |
| `venice_analyze` | Code analysis (explain, review, bugs, improve, security) |
| `venice_image` | Image generation (Flux model) |
| `venice_embeddings` | Generate text embeddings |
| `venice_json` | Structured JSON output |
| `venice_summarize` | Text summarization |
| `venice_models` | List available models |

## Models

**Text Models:**
- `llama-3.3-70b` - Default, general conversation (128K context)
- `deepseek-r1-671b` - Advanced reasoning, chain-of-thought
- `qwen-2.5-coder-32b` - Code specialist
- `dolphin-2.9.3-mistral-7b` - Uncensored, no restrictions

**Image Models:**
- `flux-dev` - High-quality image generation

## Installation

```bash
cd ~/mcp-venice
npm install
npm run build
```

## Configuration

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "venice": {
      "type": "stdio",
      "command": "node",
      "args": ["/Users/ntemis/mcp-venice/dist/index.js"],
      "env": {
        "VENICE_API_KEY": "your-api-key"
      }
    }
  }
}
```

## API Key

Get your API key from: https://venice.ai/settings/api

## Usage Examples

```
# Chat with web search
venice_chat(messages: [...], web_search: "on")

# Uncensored query
venice_uncensored(prompt: "...")

# Generate image
venice_image(prompt: "A sunset over mountains", style: "photorealistic")

# Code generation
venice_code(task: "Create a REST API", language: "typescript")
```

## Links

- [Venice AI](https://venice.ai)
- [API Documentation](https://docs.venice.ai)
- [OpenAI Compatibility](https://docs.venice.ai/api-reference/api-spec)
