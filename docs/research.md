# Bifrost × llama.cpp Provider — Integration Research

> Research phase deliverable. Every claim below is verified against source at the pinned
> revisions in §1. No Bifrost interface, llama.cpp endpoint, or request field is invented.

## 1. Pinned versions under investigation

| Component | Revision | Notes |
| --- | --- | --- |
| Bifrost (maximhq/bifrost) | commit `7e26cffbd47cd295f35b64176bfbb721fdd0924a` (main, 2026-08-30); go-gettable core tag `core/v1.8.4` = `6d60595b574e5a89289e1449825d0854aa99eabb` | `core/schemas/{bifrost.go, chatcompletions.go, models.go, account.go, textcompletions.go, embedding.go}` and the `schemas.Provider` interface are **byte-identical** between `core/v1.8.4` and main HEAD (verified via sha256 + diff). We therefore target **`github.com/maximhq/bifrost/core v1.8.4`** for compilation and treat main-HEAD behavior as equivalent. |
| llama.cpp (ggml-org/llama.cpp) | commit `9723942adc518b43c4b95dc4dce6906903eb5e09` (build string `b10700-11-g9723942`) | Route wiring in `tools/server/server.cpp`; handlers in `tools/server/server-context.cpp`; request parsing in `tools/server/server-common.cpp`; response shaping in `tools/server/server-task.cpp`. |

Go toolchain in this environment: `go1.27.0-X:nodwarf5 linux/amd64`. Bifrost core requires `go 1.27.0`.

## 2. THE KEY FINDING (read this first)

**Bifrost does not support dynamically registering a provider or HTTP routes from a
runtime Go plugin.** This was verified by source inspection, not assumption:

1. **Provider instantiation is a hardcoded switch.**
   `core/bifrost.go:4457` `createBaseProvider(providerKey schemas.ModelProvider, config *schemas.ProviderConfig)`
   switches over the provider key:

   ```go
   case schemas.Ollama:
       return ollama.NewOllamaProvider(config, bifrost.logger)
   ...
   default:
       return nil, fmt.Errorf("unsupported provider: %s", targetProviderKey)
   ```

   There is **no exported registration API**. A repo-wide search for
   `RegisterProvider|ProviderFactory|providerRegistry|AddProvider(implementation)|SetProviderFactory`
   finds nothing in `core/`, `framework/`, or `transports/`.
   (`RegisterKnownProvider` in `core/schemas/utils.go:68` only adds a name to the model-string
   parser map used by `ParseModelString`; it does not register a provider implementation, and is
   called internally by `prepareProvider` for already-registered providers.)

2. **Provider keys are compile-time constants.**
   `core/schemas/bifrost.go:43-76`: `type ModelProvider string` + a fixed `const` block
   (`OpenAI`, `Azure`, …, `Ollama`, …, `Wafer`). `StandardProviders`/`SupportedBaseProviders`
   slices derive from it.

3. **Request flow resolves the provider queue before any plugin pre-hook can save it.**
   `core/bifrost.go:5492` `tryRequest` (and `:5755` `tryStreamRequest`) begin with:

   ```go
   provider, model, _ := req.GetRequestFields()
   pq, err := bifrost.getProviderQueue(provider)   // line 5496 / 5759 — BEFORE hooks
   ...
   preReq, shortCircuit, preCount := pipeline.RunLLMPreHooks(ctx, req)  // line 5527 / 5801
   ```

   `getProviderQueue` (line 4598) lazily calls `prepareProvider → createBaseProvider`, so a request
   with `provider: "llamacpp"` fails with `unsupported provider: llamacpp` **before** a plugin's
   `PreLLMHook` short-circuit is ever consulted. (`PreRequestHook` runs earlier — `handleRequest`
   line ~5233 / `handleStreamRequest` ~5378 — so it *can* rewrite `req.Provider` to a known key;
   see §5 for what that enables/limits.)

4. **The Go plugin API is hook-based only.**
   `framework/plugins/soplugin.go` / `soloader.go`: a `.so` plugin is loaded via `plugin.Open`
   and inspected for these exported symbols (the complete list):
   `Init(config any) error`, `GetName() string`, `Cleanup() error`,
   `HTTPTransportPreAuthHook`, `HTTPTransportPreHook`, `HTTPTransportPostHook`,
   `HTTPTransportStreamChunkHook`, `PreRequestHook`, `PreLLMHook`, `PostLLMHook`,
   `PreMCPHook`, `PostMCPHook`, `PreMCPConnectionHook`, `PostMCPConnectionHook`, `Inject`.
   None of them is a provider-registration seam. `BifrostConfig` (`core/schemas/bifrost.go:25-40`)
   has no provider-factory field either.

5. **HTTP routes are static.** The bifrost-http transport registers routes at server startup
   (`transports/bifrost-http/handlers/*.go` `RegisterRoutes`); there is no plugin-visible
   `RegisterRoute`/`AddRoute` API.

6. **HTTPTransportPreHook short-circuit responses are buffered.**
   `transports/bifrost-http/handlers/middlewares.go:794` `applyHTTPResponseToCtx` does
   `ctx.SetBody(resp.Body)` from `schemas.HTTPResponse{StatusCode, Headers, Body []byte}`
   (`core/schemas/plugin.go:93-97`) — there is no streaming-capable return type for transport
   short-circuits. (An undocumented workaround to reach the raw `*fasthttp.RequestCtx` through
   the plugin's parent context exists; see §5.3. It is fragile and not a supported API.)

### Consequence (corrected)

A **first-class** `llamacpp` provider (selectable via `provider: "llamacpp"` with
native routes in the Bifrost UI/config) **requires a minimal Bifrost core patch**.

Per the project constraint — **we do not modify, fork, or ship a patched Bifrost** —
that path is off the table. Instead this repository delivers the llama.cpp integration
through **two supported, no-Bifrost-change surfaces**, both verified against source:

1. **An importable Go provider library** (`provider/llamacpp`). A `schemas.Provider`
implementation you call directly in your Go code — zero Bifrost changes, no dispatch,
   no fork. This is the recommended path for Go applications.

2. **A Bifrost plugin** (`plugin/llamacppplugin`) that routes `provider: "llamacpp"`
   traffic through a stock Bifrost instance using only the public `schemas.LLMPlugin`
   hook API (`PreRequestHook` + `PreLLMHook`). It performs the real llama-server call
   itself and short-circuits — true SSE streaming included (the native
   `LLMPluginShortCircuit.Stream` is honored by core). It does not disguise llama.cpp
   as the OpenAI/Ollama provider and does not fall back to it. See §5.2 for the
   verified mechanism and its one documented requirement (a configured carrier key).

Nothing is disguised as a plugin, and nothing requires patching core.


## 3. Exact Bifrost integration points

### 3.1 Provider interface (what a first-class provider implements)

`core/schemas/provider.go:634` — `type Provider interface` — 56 methods. Full list at
`core/v1.8.4:core/schemas/provider.go`. Signatures (abridged; `ctx *BifrostContext` omitted in
comments, `key schemas.Key`):

```go
GetProviderKey() ModelProvider
ListModels(ctx, keys []Key, req *BifrostListModelsRequest) (*BifrostListModelsResponse, *BifrostError)
TextCompletion(ctx, key, req *BifrostTextCompletionRequest) (*BifrostTextCompletionResponse, *BifrostError)
TextCompletionStream(ctx, postHookRunner PostHookRunner, postHookSpanFinalizer func(context.Context), key, req) (chan *BifrostStreamChunk, *BifrostError)
ChatCompletion(ctx, key, req *BifrostChatRequest) (*BifrostChatResponse, *BifrostError)
ChatCompletionStream(ctx, postHookRunner, postHookSpanFinalizer, key, req) (chan *BifrostStreamChunk, *BifrostError)
Responses / ResponsesStream (same pattern; *BifrostResponsesRequest)
CountTokens(ctx, key, req *BifrostResponsesRequest) (*BifrostCountTokensResponse, *BifrostError)
Compaction(ctx, key, req *BifrostCompactionRequest) (*BifrostCompactionResponse, *BifrostError)
Embedding(ctx, key, req *BifrostEmbeddingRequest) (*BifrostEmbeddingResponse, *BifrostError)
Rerank(ctx, key, req *BifrostRerankRequest) (*BifrostRerankResponse, *BifrostError)
OCR / Speech / SpeechStream / Transcription / TranscriptionStream
ImageGeneration(+Stream), ImageEdit(+Stream), ImageVariation
VideoGeneration, VideoEdit, VideoRetrieve, VideoDownload, VideoDelete, VideoList, VideoRemix
BatchCreate, BatchList, BatchRetrieve, BatchCancel, BatchDelete, BatchResults
FileUpload, FileList, FileRetrieve, FileDelete, FileContent
CachedContentCreate/List/Retrieve/Update/Delete
ContainerCreate/List/Retrieve/Delete, ContainerFileCreate/List/Retrieve/Content/Delete
Passthrough(ctx, key, req *BifrostPassthroughRequest) (*BifrostPassthroughResponse, *BifrostError)
PassthroughStream(ctx, postHookRunner, postHookSpanFinalizer, key, req) (chan *BifrostStreamChunk, *BifrostError)
```

`PostHookRunner` (core/schemas/provider.go:633):
`func(ctx *BifrostContext, result *BifrostResponse, err *BifrostError) (*BifrostResponse, *BifrostError)`.

Most methods are implemented as explicit `NewUnsupportedOperationError` (see `core/providers/ollama/ollama.go`,
482 lines, the closest analog — local inference server with per-key base URL).

### 3.2 Provider instantiation seam (why a first-class provider needs a core patch)

The seam where a first-class provider is wired into Bifrost — and therefore why the
plugin/library approach is needed instead (this repo does not patch core):

- `core/schemas/bifrost.go:45-76` — `ModelProvider` constants; a first-class provider
  needs `LLamaCpp ModelProvider = "llamacpp"` here (+ `StandardProviders`).
- `core/bifrost.go:4457-5541` — `createBaseProvider` switch; a first-class provider needs
  `case schemas.LLamaCpp: return llamacpp.NewLLamaCppProvider(config, bifrost.logger)`.
- `core/schemas/account.go` — optional `LLamaCppKeyConfig` on `Key` (per-key base URL /
  API key, mirroring `OllamaKeyConfig` at account.go:815-817).
- `transports/config.schema.json` — provider config schema entry.

None of these changes are shipped or required by this repo. The provider package is built
so that such a patch would be small, mechanical, and low-risk if it is ever adopted
upstream, but the provider and plugin work without it today.

### 3.3 Request/response types (canonical shapes the provider maps to/from)

- `BifrostChatRequest` (`core/schemas/chatcompletions.go:14-33`): `Provider ModelProvider`,
  `Model string`, `Input []ChatMessage`, `Params *ChatParameters` (contains
  `Reasoning *ChatReasoning`, `Tools []ChatTool`, `ResponseFormat`, `MaxCompletionTokens`,
  **`ExtraParams map[string]any`** `json:"-"`, …), `Fallbacks`, `RawRequestBody []byte`.
- `ChatReasoning` (chatcompletions.go:329-334): `Enabled *bool`, **`Effort *string`** (free string),
  `MaxTokens *int`, `Display *string`. **No whitelist/enforcement in core.**
- `BifrostStreamChunk` (`core/schemas/bifrost.go:1837-1846`): one embedded response or `BifrostError`.
- `BifrostError`: `IsBifrostError bool`, `Error *ErrorField{Message,Type,Code,Param,...}`,
  `ExtraFields`, `AllowFallbacks *bool`, `UpstreamLatencyMS *float64`, … (provider.go).
- `Model` (`core/schemas/models.go:149-181`): includes **`ContextLength *int`**, `MaxInputTokens`,
  `MaxOutputTokens`, `Pricing`, `SupportedMethods` — this is how per-model context metadata is
  surfaced. `BifrostListModelsResponse.Data []Model` is what `/v1/models` returns.
- `schemas.Key` (account.go:140+): `Value SecretVar` (API key), `Models WhiteList`,
  per-provider config structs (`OllamaKeyConfig.URL`, etc.).

### 3.4 Provider-specific extension mechanism (official, verified)

Bifrost has an **official extra-params passthrough**:

- HTTP transport: `transports/bifrost-http/integrations/router.go:754-770` — when the request sets
  header `x-bf-passthrough-extra-params: true` (parsed at `transports/bifrost-http/lib/ctx.go:661-663`),
  the top-level JSON key `"extra_params"` from the request body is copied into the Bifrost request's
  `ExtraParams` map. Comment in source: *"Provider-specific fields (e.g. Bedrock guardrailConfig)
  must be nested under 'extra_params' in the request body."*
- Providers read it: e.g. `core/providers/openai/chat.go:51` `openaiReq.ExtraParams = bifrostReq.Params.ExtraParams`
  and then merge into the upstream wire body; `core/providers/anthropic/chat.go:265-420` promotes
  specific keys (thinking, top_k, speed) from `ExtraParams`.
- Go SDK: `Params.ExtraParams` is directly settable.

**Decision:** llama.cpp-specific extensions use `extra_params` (Bifrost's official mechanism):
direct llama-server top-level fields (`grammar`, `json_schema`, `chat_template_kwargs`,
`reasoning_effort`, …) plus a reserved `"llamacpp"` sub-object for provider-level options
(reasoning passthrough policy, etc.). No new Bifrost schema is needed for this.

### 3.5 Reasoning-effort normalization behavior (why a dedicated provider matters)

- Core does **not** normalize effort strings: `ChatReasoning.Effort` is `*string`, and no
  validation of `"xhigh"`/`"max"` exists in core or in the HTTP integration layer.
- **Specific providers do normalize**: `core/providers/openai/chat.go:225` and
  `core/providers/gemini/utils.go:337` call `caps.NormalizeReasoningEffort(effort, …)`
  (`core/schemas/modelcapsreasoning.go:121-171`), which *downgrades* unknown efforts onto the
  model's published ladder (e.g. `xhigh` → nearest rung). This is exactly the lossy behavior the
  task forbids for llama.cpp. A first-class `llamacpp` provider simply **does not call**
  `NormalizeReasoningEffort` and forwards `Params.Reasoning.Effort` verbatim into llama-server's
  `reasoning_effort` (which itself accepts arbitrary strings — see §4.2/§4.11).

### 3.6 Model catalog / context metadata

- `BifrostConfig.ModelCatalog ModelInfoProvider` (bifrost.go:39):
  `GetModelInfo(provider ModelProvider, model string) *Model` + `CalculateRequestCost`.
  Set at construction (no runtime setter). Plugins read it via `ctx.GetModelInfo`.
- The `Model` struct carries `ContextLength *int` (models.go:156). The catalog is editorial
  (datasheet-driven) — there is **no provider callback to contribute discovered context lengths**.
- **What a provider CAN do today:** populate `ContextLength`/`MaxInputTokens` in `ListModels`
  responses (`BifrostListModelsResponse.Data`). This is the supported path for exposing
  llama-server-derived context windows via `/v1/models`.
- **Missing (upstream contribution needed):** a provider-supplied model-catalog feed, e.g.
  an optional `ModelCatalogContributor` interface that `setModelCatalogOnContext` (bifrost.go:412)
  consults, so `ctx.GetModelInfo` and governance/context-budget logic see live llama-server
  context data. Documented in §6 of the upstream plan.

### 3.7 Go plugin runtime requirements (for the optional hook adapter)

- Linux/macOS only (`framework/plugins`; Go `plugin` package).
- Requires a **dynamically linked** Bifrost binary (statically linked binaries cannot load
  `.so` plugins — `docs/plugins/building-dynamic-binary.mdx`).
- Plugin build: `go build -buildmode=plugin`, export symbols per §2.4, load via `plugins` config.
- Plugin modules are separate Go modules requiring `github.com/maximhq/bifrost/core v1.8.x`
  (see `plugins/*/go.mod`).

### 3.8 Passthrough request type (relevant to native routes)

`schemas.BifrostPassthroughRequest` (core/schemas/passthrough.go:3-12):
`Provider, Model, Method, Path, RawQuery, UpstreamURL, Body []byte, SafeHeaders`.
Providers expose `Passthrough`/`PassthroughStream`. No provider uses it for llama-style native
routes today; it is an alternative upstream design for native routes but has no transport route
wired to it for a custom namespace (`/v1/providers/llamacpp/*`), so we treat it as: *potential*
upstream integration, *not* a current capability.

## 4. Verified llama-server API surface (target revision)

Source-verified from `9723942adc…` (`b10700-11-g9723942`). Full route/handler/field reference was
produced from source by a dedicated research pass; the material facts:

### 4.1 Routes (tools/server/server.cpp:236-304)

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/health`, `/v1/health` | Public (no API key), `{ "status": "ok" }` |
| GET | `/props` | always on; **server/model metadata incl. context + build_info** |
| POST | `/props` | requires `--props` |
| GET | `/models`, `/v1/models` | always on; **per-model metadata incl. n_ctx / n_ctx_train** |
| POST | `/completion`, `/completions` | native completions (non-OAI) |
| POST | `/v1/completions` | OAI-compatible completions |
| POST | `/chat/completions`, `/v1/chat/completions` | **primary chat route** (OAI-compatible + llama.cpp extensions) |
| POST | `/v1/chat/completions/control` | reasoning-end control |
| POST | `/v1/responses`, `/responses` | OAI Responses API |
| POST | `/v1/embeddings`, `/embeddings`, `/embedding` | requires `--embedding` |
| POST | `/v1/rerank`, `/rerank`, … | requires reranking |
| POST | `/tokenize` | always on |
| POST | `/detokenize` | always on |
| POST | `/apply-template` | templated prompt preview |
| POST | `/chat/completions/input_tokens`, `/v1/chat/completions/input_tokens` | **token counting for chat** |
| POST | `/responses/input_tokens`, `/v1/responses/input_tokens` | token counting (responses) |
| POST | `/v1/messages/count_tokens` | Anthropic-style counting |
| GET | `/slots` | requires `--slots` |
| POST | `/slots/:id_slot?action=save\|restore\|erase` | requires `--slot-save-path` |
| GET | `/metrics` | requires `--metrics` (Prometheus text) |
| GET | `/lora-adapters`, POST `/lora-adapters` | adapters |
| GET/POST/DELETE | `/v1/stream`, `/v1/streams/lookup` | resumable stream API |
| router-mode extras | `POST /models`, `/models/load`, `/models/unload`, `DELETE /models`, `GET /models/sse` | only under `--router` |

Auth: `--api-key` / `--api-key-file` (`common/arg.cpp:3504-3536`). Checks `Authorization` (Bearer
stripped) then `X-Api-Key`; 401 JSON on failure; `/health`+`/v1/health` are public. **A provider
API key maps directly to Bearer auth.** (server-http.cpp:208-251)

### 4.2 `/chat/completions` request fields (server-common.cpp:1077-1444)

Explicitly parsed: `messages`, `stream`, `tools`, `tool_choice`, `parallel_tool_calls`, `stop`,
`json_schema`, `grammar`, `response_format` (text | json_object | json_schema),
`add_generation_prompt`, `continue_final_message`, `chat_template_kwargs`, **`reasoning_effort`**,
**`reasoning_format`** (`none|auto|deepseek|deepseek-legacy`), `reasoning_budget_tokens`
(alias `thinking_budget_tokens`), `reasoning_budget_message`, `reasoning_control`,
`logprobs`, `top_logprobs`.

**Any other top-level key is passed through verbatim into llama params** (server-common.cpp:1436-1444):
`temperature`, `top_p`, `top_k`, `repeat_penalty`, `min_p`, `mirostat*`, `penalty_last_n`,
`frequency_penalty`, `presence_penalty`, `max_tokens` (→ n_predict), etc.

Constraints: `grammar` ✗ with `json_schema` (1163-1164) and ✗ with tools (1299-1301);
`tools`/non-auto `tool_choice` require `--jinja`.

### 4.3 Chat response shapes

- Non-stream OAI: `choices[{finish_reason, index, message, logprobs?}], created, model,
  system_fingerprint, object:"chat.completion", usage{prompt_tokens, completion_tokens,
  total_tokens, prompt_tokens_details.cached_tokens}, id` (server-task.cpp:414-460).
- Streaming: SSE `data: {chat.completion.chunk...}\n\n` per chunk → terminal
  `data: [DONE]\n\n` (server-context.cpp:4387-4477; format_oai_sse server-common.cpp:1586-1610).
  First delta `{role:"assistant",content:null}`; usage chunk optional. Reasoning appears as
  `reasoning_content` on message/delta where the model/template emits it.
- Native completions response: `index, content, tokens, id_slot, stop, model, tokens_predicted,
  tokens_evaluated, generation_settings, prompt, has_new_line, truncated, stop_type,
  stopping_word, tokens_cached, timings` (server-task.cpp:340-354).

### 4.4 Metadata endpoints

`GET /v1/models` (`get_res_models`, server-context.cpp:4539-4571):

- `.data[]`: OAI-style `{ id, aliases, tags, object:"model", created, owned_by:"llamacpp",
  meta: { vocab_type, n_vocab, n_ctx, n_ctx_train, n_embd, n_params, size, ftype } }`
- `.models[]`: Ollama-style list (mostly empty strings; `capabilities` meaningful).

`GET /props` (`get_res_props`, 4572-4620):
`default_generation_settings{ params, n_ctx }` (n_ctx = per-slot context = `min(CTX, model_n_ctx_train)`),
`total_slots`, `model_alias`, `model_ftype`, `model_path`, `modalities{vision,video,audio}`,
`chat_template`, `chat_template_caps`, `bos_token`, `eos_token`, **`build_info`** (llama.cpp
version string), `is_sleeping`, feature flags (`endpoint_slots`, `endpoint_props`,
`endpoint_metrics`, `ui`), optional `chat_template_tool_use`.

**Context-window provenance:** `n_ctx` (effective, per-slot) and `n_ctx_train` (GGUF-trained)
are **live server values**, not name heuristics. Same family name ≠ same context window
(the effective `n_ctx` is `min(-ctx value, n_ctx_train)`, set at launch). Both are read from
these endpoints with TTL cache; anything else → unknown.

`GET /slots` (requires `--slots`): array of `{id, n_ctx, speculative, is_processing, …}`.
`GET /metrics` (requires `--metrics`): Prometheus text, `llamacpp:*` prefix.

### 4.5 Tokenization / counting endpoints

- `POST /tokenize`: `{content, add_special?, parse_special?, with_pieces?}` →
  `{tokens: [ids]}` or `{tokens:[{id,piece}]}`.
- `POST /detokenize`: `{tokens:[ids]}` → `{content}`.
- `POST /v1/chat/completions/input_tokens`: chat-shape body → token count (for Bifrost
  `CountTokens` / context budgeting).

### 4.6 Reasoning semantics on llama-server (server-common.cpp:1323-1341, 1400-1418)

- `reasoning_effort`: **string, no whitelist in the server.** `"none"` → disables thinking;
  any other non-empty string → forwarded to the chat template as-is. ⇒ `"xhigh"`, `"max"`,
  future values pass through untouched — exactly what the task requires.
- `reasoning_format`: enum `none|auto|deepseek|deepseek-legacy`.
- `chat_template_kwargs.enable_thinking`: boolean override ("true"/"false").
- `reasoning_budget_tokens` / `thinking_budget_tokens`, `reasoning_control`,
  `reasoning_budget_message`: thinking-budget controls.

### 4.7 Stable vs version-dependent

**Stable (documented, long-lived, keep as contract):**
`/health`, `/v1/models`, `/chat/completions` (+`/v1/` alias), `/completions` (+`/v1/`),
`/tokenize`, `/detokenize`, `/props`, `/slots`, `/v1/embeddings`, `/metrics`, the OAI SSE
chunk format and `[DONE]`, `reasoning_effort`/`chat_template_kwargs`/`grammar`/`json_schema`
on chat completions.

**Version-dependent / flag-gated (must be feature-detected, never assumed):**
`/props` (POST form), `/slots` (flag `--slots`), `/metrics` (`--metrics`), `/embeddings`
(`--embedding`), `/rerank*` (pooling), `/v1/responses` + `/v1/messages` (OAI/Anthropic compat),
`/v1/chat/completions/control`, `/v1/stream*` (resumable streams), `/lora-adapters`,
router-mode model management routes, `reasoning_format` variants, `with_pieces` on tokenize,
the `meta` object in `/v1/models` (newer versions).

The provider therefore: (a) only calls the stable core set for Phase 1; (b) probes
`/props`/`/slots`/`/v1/embeddings`/`/tokenize` at metadata refresh and sets capability flags from
actual responses (`endpoint_*` booleans in `/props`) instead of trusting flags; (c) treats every
version-dependent route as an allowlisted native-route that fails with a clear "unsupported by
this llama-server build" error when absent.

## 5. Feasible integration strategies (ranked, all verified)

### 5.1 A. Importable Go provider library (PRIMARY — what this repo delivers)

A Go package `llamacpp` implementing `schemas.Provider`, built against the published
`github.com/maximhq/bifrost/core` module. Used by calling its methods directly in Go
code. No Bifrost change, no fork, no dispatch: your app owns the request/response and
keeps llama.cpp fully transparent (provider key is our own `llamacpp`, never another
provider).

Delivered as: `provider/llamacpp/` (importable) + tests against a fake llama-server.

Capabilities: chat (unary + true SSE streaming), text completion (unary + stream),
responses (delegated to chat completions), embeddings, count tokens, list models with
live context metadata, passthrough to native llama-server routes, and verbatim
reasoning-effort passthrough.

### 5.2 B. Bifrost plugin (DELIVERED — routes through a stock Bifrost instance)

Uses *supported* plugin APIs only. Two cooperating hooks in `plugin/llamacppplugin`:

1. `PreRequestHook`: detect `req.Provider == "llamacpp"` and rewrite it to a
   preconfigured **carrier** provider key (e.g. `"openai"`) so `getProviderQueue`
   succeeds. The deployment must have that carrier provider minimally configured
   (it is never called — the plugin always short-circuits).
2. `PreLLMHook`: if the request was rewritten → run the llama-server call via the
   `provider/llamacpp` package and return `LLMPluginShortCircuit{Response: ...}`
   (unary) or `{Stream: chan}` (SSE, **true streaming** — `tryStreamRequest` at
   bifrost.go:5801 consumes plugin streams chunk-by-chunk; the native
   `LLMPluginShortCircuit` carries a `Stream` field, plugin_native.go:23).

The carrier provider is never invoked and never substituted as the backend:
on llama-server failure the plugin short-circuits with the upstream error and
`AllowFallbacks=false`, so a llamacpp request can never silently run on the
carrier. The provider label is restored to `llamacpp` before the short-circuit.

Verified working mechanics: `LLMPluginShortCircuit.Stream` is honored
(core/bifrost.go:5801-5880); `PreRequestHook` mutations are committed and re-read
(core/bifrost.go:5234-5248). Verified limitations: carrier config requirement;
native llama-server-only routes (`/tokenize`, `/props`, …) are not served (they are
not part of the plugin API) — use the provider library for those.

### 5.3 C. Undocumented streaming workaround (NOT used)

Via `ctx.GetParentCtxWithUserValues().Value(fasthttp.ContextKey)` a transport plugin
can recover the `*fasthttp.RequestCtx` and use `SetBodyStreamWriter` for true SSE
streaming through `HTTPTransportPreHook`. **Not used** — it depends on transport
internals and would break across refactors. The §5.2 plugin already provides real
streaming through the supported API.

### 5.4 D. Companion native-route proxy (not shipped; future work)

A tiny standalone HTTP reverse proxy that exposes llama-server's native routes
(`/tokenize`, `/detokenize`, `/props`, `/slots`, …) against a configured
llama-server with an explicit route allowlist. Works alongside any Bifrost
install, no core changes. The provider library's `Passthrough`/`PassthroughStream`
methods already cover this for Go callers; a standalone binary is a packaging
convenience, not a correctness requirement.

## 6. Optional upstream contribution (out of scope for this repo; documented for completeness)

Because this repo deliberately does NOT ship a core patch, the following is recorded only as
the path that *would* be taken if llama.cpp were adopted as a first-class provider upstream.
It is not a deliverable here and is not applied to any Bifrost in this project.

If/when a llama.cpp provider were to be accepted upstream in `github.com/maximhq/bifrost/core`:

- `core/schemas/bifrost.go`: add `LLamaCpp ModelProvider = "llamacpp"` and append to the
  `StandardProviders` list.
- `core/bifrost.go` `createBaseProvider`: add `case schemas.LLamaCpp:` returning the new
  provider, and import the package.
- `core/schemas/account.go`: (optional) add an `LLamaCppKeyConfig` on `Key` to mirror the
  existing `OllamaKeyConfig` and support per-key base URLs/keys.
- `transports/config.schema.json`: add a `llamacpp` provider entry and key-config definition.
- New provider package under `core/providers/llamacpp/` (this repo's `provider/llamacpp` is
  already shaped for this: it implements the full `schemas.Provider` interface and only
  references a local `ProviderKey` constant that a one-line swap to `schemas.LLamaCpp`
  would replace).

This repo's `provider/llamacpp` is already built so that such an upstream patch is
mechanical and low-risk — but it is not a requirement to use or test this provider today.

## 7. Limitations (honest, verified)

1. **No runtime provider registration** in current Bifrost: a request whose `provider` is
   `llamacpp` cannot be natively dispatched by the built-in provider switch without a core
   change. This repo therefore (a) exposes the provider as a Go library you call directly,
   and (b) ships a `schemas.LLMPlugin` that rewrites the provider key and short-circuits.

2. **The plugin path requires a configured carrier provider key.** Because Bifrost creates
   its provider queue before running `PreLLMHook`, the plugin rewrites the `llamacpp` key to
   a pre-existing configured provider (the "carrier") just so the queue is found. The carrier
   is never called for llamacpp traffic — the plugin always short-circuits — but its presence
   in the config is a real, documented requirement. On llama-server failure the plugin sets
   `AllowFallbacks=false` so the carrier is not used as a silent fallback.

3. **Native llama-server-only routes are not served by the plugin.** `/tokenize`,
   `/detokenize`, `/props`, `/slots`, etc. are not part of Bifrost's plugin API; the plugin
   only handles the LLM request types. Use the provider library's
   `Passthrough`/`PassthroughStream` for those.

4. **HTTPTransportPreHook short-circuit responses are buffered** (no SSE) via the official
   API. This is why the plugin uses `PreLLMHook` streaming short-circuit instead, which
   does support true SSE.

5. **llama-server routes are flag-gated.** The provider probes, not presumes (§4.7).

6. **The `ProviderKey` is a local constant**, not a Bifrost core constant. This is a
   deliberate choice so the provider compiles against the public, unpatched
   `bifrost/core` module. When/if it is adopted upstream, that constant becomes
   `schemas.LLamaCpp` and is identical in string value.
