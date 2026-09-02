# Bifrost llama.cpp Provider Plugin

A [Bifrost](https://github.com/maximhq/bifrost) plugin that enables first-class llama.cpp support through the llama-server API.

## Features

- **Verbatim reasoning-effort passthrough**: `xhigh`, `max`, `low`, `medium`, `high` — any value is forwarded unchanged to llama.cpp
- **True SSE streaming**: Chunk-by-chunk streaming with proper OpenAI-compatible response format
- **Live model metadata**: Discovers available models, context lengths, and capabilities from your llama-server instance
- **Native llama-server extensions**: Grammar, chat templates, top-k, and other llama.cpp-specific parameters
- **Zero provider disguise**: llama.cpp is its own provider, not masquerading as OpenAI or another provider

## Requirements

- Bifrost (Docker or from source)
- llama-server running and accessible from your Bifrost instance
- Go 1.27.0 (only if building from source)

## Quick Start

### 1. Build Bifrost from source (required for plugins)

The prebuilt Bifrost binary cannot load plugins (static linking). You must build from source:

```bash
git clone https://github.com/maximhq/bifrost
cd bifrost/transports/bifrost-http
mkdir -p ui && touch ui/.keep  # UI placeholder
GOWORK=off CGO_ENABLED=1 GOTOOLCHAIN=go1.27.0 go build -ldflags="-w -s" -o bifrost-http .
```

Verify it's dynamically linked:

```bash
file bifrost-http
# Should say "dynamically linked"
```

### 2. Download or build the plugin

**Option A: Download from release**

Download `llamacpp-plugin.so` from the latest [GitHub release](https://github.com/Max9403/bifrost-llama.cpp-provider/releases).

**Option B: Build from source**

```bash
git clone https://github.com/Max9403/bifrost-llama.cpp-provider
cd bifrost-llama.cpp-provider/plugin/llamacpp-so
go build -buildmode=plugin -o llamacpp-plugin.so .
```

### 3. Configure the plugin

Add the plugin to your Bifrost configuration via the UI or config file:

```json
{
  "plugins": [
    {
      "enabled": true,
      "name": "llamacpp-plugin",
      "path": "/path/to/llamacpp-plugin.so",
      "config": {
        "provider_key": "llamacpp",
        "base_url": "http://127.0.0.1:8080"
      }
    }
  ]
}
```

### 4. Use it

Select a llama.cpp model with the `llamacpp/` prefix:

```bash
curl -X POST http://localhost:23232/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llamacpp/your-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "reasoning_effort": "high"
  }'
```

## Configuration Options

| Option | Type | Default | Description |
| -------- | ------ | --------- | ------------- |
| `provider_key` | string | `llamacpp` | Provider identifier for model routing |
| `base_url` | string | `http://127.0.0.1:8080` | llama-server base URL |
| `api_key` | string | `""` | Optional API key for llama-server auth |

## Plugin Architecture

This plugin uses Bifrost's LLM plugin hooks to intercept requests for `llamacpp/*` models and route them directly to the llama.cpp provider engine. The provider implements the full `schemas.Provider` interface with:

- Chat completion (unary + SSE stream)
- Live model listing and metadata discovery
- Verbatim `reasoning_effort` passthrough
- Native llama-server parameter support via ExtraParams
- Proper error handling and mapping to Bifrost error formats

## Building

The plugin uses Go's plugin system (`-buildmode=plugin`). Key requirements:

- Go version must match between the plugin and Bifrost binary (1.27.0)
- The plugin is compiled as a shared object (`.so`) on Linux/macOS
- The `provider/llamacpp` package implements the core provider logic
- The `plugin/llamacpp-so` package provides the Bifrost plugin entry points

## Testing

```bash
go test ./provider/...
```

## License

MIT
