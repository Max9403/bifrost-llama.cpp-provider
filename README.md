# Bifrost llama.cpp Plugin

A [Bifrost](https://github.com/maximhq/bifrost) plugin that provides verbatim reasoning-effort passthrough for llama.cpp models via llama-server.

## How It Works

The plugin uses Bifrost's `HTTPTransportPreHook` to intercept chat completion requests before Bifrost parses them. When it detects a request for a model with a configured prefix, it:

1. Moves the `reasoning_effort` field from the root of the request into `chat_template_kwargs`
2. Deletes the root-level `reasoning_effort` to prevent Bifrost from normalizing it
3. Sets the `x-bf-passthrough-extra-params` header to ensure provider-specific fields are forwarded

This preserves the caller's exact effort value (low, medium, high, xhigh, max, or any custom value) without lossy ladder-clamping.

## Requirements

- Bifrost built from source (dynamically linked, not the prebuilt npm binary)
- Go 1.27.0 (matching Bifrost's toolchain)
- Linux or macOS (Go plugins are not supported on Windows)

## Quick Start

### 1. Build Bifrost from source

```bash
git clone https://github.com/maximhq/bifrost
cd bifrost/transports/bifrost-http
mkdir -p ui && touch ui/.keep
GOWORK=off CGO_ENABLED=1 GOTOOLCHAIN=go1.27.0 go build -ldflags="-w -s" -o bifrost-http .
```

### 2. Download or build the plugin

**Download from release:**
Download `llamacpp-plugin.so` from the latest [release](https://github.com/Max9403/bifrost-llama.cpp-provider/releases).

**Build from source:**

```bash
git clone https://github.com/Max9403/bifrost-llama.cpp-provider
cd bifrost-llama.cpp-provider
cd plugin/llamacpp-so
go build -buildmode=plugin -o ../../build/llamacpp-plugin.so .
```

### 3. Configure the plugin

Add the plugin to your Bifrost configuration:

```json
{
  "plugins": [
    {
      "path": "/path/to/llamacpp-plugin.so",
      "name": "llamacpp-reasoning-effort",
      "enabled": true,
      "placement": "pre_builtin",
      "order": 0,
      "config": {
        "provider_prefixes": ["llama-local/"]
      }
    }
  ]
}
```

The `pre_builtin` placement is required so the hook runs before Bifrost's built-in processing.

### 4. Use it

Target your llama-server with a configured model prefix:

```bash
curl http://localhost:23232/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-local/your-model-name",
    "messages": [{"role": "user", "content": "Hello!"}],
    "reasoning_effort": "xhigh"
  }'
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `provider_prefixes` | array of strings | `["llama-local/"]` | List of model name prefixes to intercept. Requests for models starting with any of these prefixes will have reasoning_effort preserved verbatim. |

Example with multiple prefixes:

```json
{
  "config": {
    "provider_prefixes": ["llama-local/", "ollama/", "local-models/"]
  }
}
```

## Why This Approach?

Bifrost normalizes `reasoning_effort` values on its OpenAI-compatible path, which breaks llama.cpp models that accept custom effort levels like `xhigh` or `max`. This plugin sidesteps normalization by rewriting the request before Bifrost parses it, rather than after.

## License

MIT
