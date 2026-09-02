// Package llamacpp implements a first-class Bifrost provider for llama.cpp's
// llama-server HTTP API (https://github.com/ggml-org/llama.cpp/tree/master/tools/server).
// It compiles as a standalone Go module against the published
// github.com/maximhq/bifrost/core (see go.mod). It does not require any
// modification, fork, or patch of Bifrost: it is a library you call
// directly, and it is also wired through stock Bifrost by the LLMPlugin in
// plugin/llamacppplugin.
//
// Design decisions — every one is verified against source at the pinned
// revisions in docs/research.md:
//
//   - REAL provider, not a plugin, not a fork, not an OpenAI-provider
//     frontend. It implements schemas.Provider directly and maps
//     llama-server's OpenAI-compatible surface (plus llama.cpp extensions)
//     natively.
//
//   - Reasoning-effort passthrough (core requirement):
//     params.reasoning.effort is forwarded VERBATIM to llama-server's
//     `reasoning_effort` field, which has no whitelist in the server and
//     forwards arbitrary strings into the chat template ("none" disables
//     thinking). The provider NEVER calls caps.NormalizeReasoningEffort —
//     the lossy ladder-clamping the OpenAI/Gemini providers apply (e.g.
//     "xhigh" → nearest published rung) is intentionally NOT applied here.
//
//   - Provider-specific extensions use Bifrost's official ExtraParams
//     passthrough (x-bf-passthrough-extra-params header /
//     params.extra_params map). Top-level entries are merged into the
//     llama-server request body as native fields (grammar, json_schema,
//     chat_template_kwargs, top_k, ...); a reserved `llamacpp` sub-object
//     carries provider-level options (e.g. disabling effort passthrough).
//
//   - Context-window metadata is discovered from llama-server's LIVE
//     endpoints (GET /v1/models meta.n_ctx / meta.n_ctx_train; GET /props
//     default_generation_settings.n_ctx) with a TTL cache — never from
//     model-name heuristics.
//
//   - Verbatim wire fidelity is achieved through the shared OpenAI-family
//     conversion layer's documented "custom provider" mode
//     (schemas.BifrostContextKeyIsCustomProvider), under which the
//     conversion returns the request unmodified: no OpenAI field filtering,
//     no reasoning normalization — exactly the passthrough semantics
//     llama-server needs. The request is then post-processed in THIS package
//     (chat.go): provider options are stripped, reasoning_effort is promoted
//     verbatim, and the final body is what llama-server receives.
//
//   - Streaming uses llama-server's documented OpenAI-style SSE format
//     (data: {chat.completion.chunk...}\n\n ... data: [DONE]), consumed
//     chunk-by-chunk through Bifrost's standard streaming helpers, so
//     cancel/timeout/idle/governance behavior matches the built-in providers.
package llamacpp

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"time"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	schemas "github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// ProviderKey is the Bifrost ModelProvider identifier for llama.cpp.
//
// Declared locally so this module compiles against the published,
// unmodified bifrost/core. If the provider is ever adopted upstream, this
// constant becomes schemas.LLamaCpp with the identical string value.
const ProviderKey schemas.ModelProvider = "llamacpp"

// LLamaCppProvider implements schemas.Provider for llama.cpp's llama-server.
type LLamaCppProvider struct {
	logger              schemas.Logger
	client              *fasthttp.Client // unary requests
	streamingClient     *fasthttp.Client // streaming (no read timeout)
	networkConfig       schemas.NetworkConfig
	sendBackRawRequest  bool
	sendBackRawResponse bool

	mu             sync.RWMutex
	metadataCache  map[string]llamaServerMetadata
	metadataExpiry map[string]time.Time
	metadataTTL    time.Duration
}

// llamaServerMetadata is context-window and build metadata discovered from
// llama-server's live endpoints (never inferred from model names).
type llamaServerMetadata struct {
	ContextLength                  *int // effective per-slot n_ctx
	TrainedContextLength           *int // n_ctx_train (GGUF)
	BuildInfo                      string
	Aliases                        []string
	Vision, Audio, Video           bool
	HasSlots, HasMetrics, HasEmbed bool
	DiscoveredAt                   time.Time
}

// noopLogger is a discard logger used when the caller does not provide one.
type noopLogger struct{}

func (noopLogger) Debug(string, ...any)                   {}
func (noopLogger) Info(string, ...any)                    {}
func (noopLogger) Warn(string, ...any)                    {}
func (noopLogger) Error(string, ...any)                   {}
func (noopLogger) Fatal(string, ...any)                   {}
func (noopLogger) SetLevel(schemas.LogLevel)              {}
func (noopLogger) SetOutputType(schemas.LoggerOutputType) {}
func (noopLogger) LogHTTPRequest(schemas.LogLevel, string) schemas.LogEventBuilder {
	return noopLogEventBuilder{}
}

// noopLogEventBuilder is a no-op builder for loggers that don't need structured logging.
type noopLogEventBuilder struct{}

func (noopLogEventBuilder) Str(string, string) schemas.LogEventBuilder  { return noopLogEventBuilder{} }
func (noopLogEventBuilder) Int(string, int) schemas.LogEventBuilder     { return noopLogEventBuilder{} }
func (noopLogEventBuilder) Int64(string, int64) schemas.LogEventBuilder { return noopLogEventBuilder{} }
func (noopLogEventBuilder) Send()                                       {}

// NewLLamaCppProvider creates a new llama.cpp provider.
//
// Configuration:
//   - network_config.base_url: llama-server base URL, e.g. "http://127.0.0.1:8080".
//   - keys: each key's value is the API key for llama-server --api-key
//     (optional if the server runs without auth); multiple keys are supported
//     for round-robin/load balancing. Per-key base URLs arrive via the
//     upstream core patch (llamacpp_key_config) and are read reflectively so
//     this module builds against both patched and unpatched core.
func NewLLamaCppProvider(config *schemas.ProviderConfig, logger schemas.Logger) (*LLamaCppProvider, error) {
	if config == nil {
		return nil, errors.New("llamacpp: nil provider config")
	}
	if logger == nil {
		logger = noopLogger{}
	}

	config.CheckAndSetDefaults()

	requestTimeout := time.Second * time.Duration(config.NetworkConfig.DefaultRequestTimeoutInSeconds)
	client := &fasthttp.Client{
		ReadTimeout:         requestTimeout,
		WriteTimeout:        requestTimeout,
		MaxConnsPerHost:     config.NetworkConfig.MaxConnsPerHost,
		MaxIdleConnDuration: time.Second * time.Duration(config.NetworkConfig.KeepAliveTimeoutInSeconds),
		MaxConnWaitTimeout:  requestTimeout,
		MaxConnDuration:     time.Second * time.Duration(schemas.DefaultMaxConnDurationInSeconds),
		ConnPoolStrategy:    fasthttp.FIFO,
	}
	client = providerUtils.ConfigureProxy(client, config.ProxyConfig, logger)
	client = providerUtils.ConfigureDialer(client, config.NetworkConfig.AllowPrivateNetwork)
	client = providerUtils.ConfigureTLS(client, config.NetworkConfig, logger)
	streamingClient := providerUtils.BuildStreamingClient(client)
	config.NetworkConfig.BaseURL = strings.TrimRight(config.NetworkConfig.BaseURL, "/")

	return &LLamaCppProvider{
		logger:              logger,
		client:              client,
		streamingClient:     streamingClient,
		networkConfig:       config.NetworkConfig,
		sendBackRawRequest:  config.SendBackRawRequest,
		sendBackRawResponse: config.SendBackRawResponse,
		metadataCache:       make(map[string]llamaServerMetadata),
		metadataExpiry:      make(map[string]time.Time),
		metadataTTL:         30 * time.Second,
	}, nil
}

// GetProviderKey implements schemas.Provider.
func (provider *LLamaCppProvider) GetProviderKey() schemas.ModelProvider {
	return ProviderKey
}

// baseURL resolves the llama-server base URL for a key:
// per-key llamacpp_key_config.url (upstream patch) → network_config.base_url.
func (provider *LLamaCppProvider) baseURL(key schemas.Key) string {
	if url, ok := provider.keyConfigURL(key); ok {
		return strings.TrimRight(url, "/")
	}
	if provider.networkConfig.BaseURL != "" {
		return strings.TrimRight(provider.networkConfig.BaseURL, "/")
	}
	return ""
}

// keyConfigURL reads the upstream-patch per-key URL reflectively so the module
// compiles pre-patch (field absent → not found) and works post-patch.
func (provider *LLamaCppProvider) keyConfigURL(key schemas.Key) (string, bool) {
	v := reflect.ValueOf(&key).Elem().FieldByName("LLamaCppKeyConfig")
	if !v.IsValid() || v.Kind() != reflect.Ptr || v.IsNil() {
		return "", false
	}
	urlField := v.Elem().FieldByName("URL")
	if !urlField.IsValid() || urlField.Kind() != reflect.Struct {
		return "", false
	}
	if sv, ok := urlField.Interface().(schemas.SecretVar); ok {
		if u := sv.GetValue(); u != "" {
			return u, true
		}
	}
	return "", false
}

// baseURLOrError returns the resolved base URL or a BifrostError.
func (provider *LLamaCppProvider) baseURLOrError(key schemas.Key) (string, *schemas.BifrostError) {
	u := provider.baseURL(key)
	if u == "" {
		return "", providerUtils.NewBifrostOperationError(
			"no base URL configured for llamacpp: set network_config.base_url to the llama-server URL (e.g. http://127.0.0.1:8080)",
			nil)
	}
	if !strings.HasPrefix(u, "http://") && !strings.HasPrefix(u, "https://") {
		return "", providerUtils.NewConfigurationError(
			fmt.Sprintf("llamacpp base URL must start with http:// or https://, got %q", u))
	}
	return u, nil
}

// authHeader builds Bearer auth (llama-server --api-key support).
func (provider *LLamaCppProvider) authHeader(key schemas.Key) map[string]string {
	headers := make(map[string]string, 1)
	if key.Value.GetValue() != "" {
		headers["Authorization"] = "Bearer " + key.Value.GetValue()
	}
	return headers
}

// setLLamaCppCtxFlags configures the request context for verbatim passthrough:
//
//   - BifrostContextKeyIsCustomProvider: the shared OpenAI-family conversion
//     layer treats this provider as a custom (passthrough) provider — it
//     returns the request unmodified instead of applying OpenAI field
//     filtering or reasoning normalization. llama-server is NOT OpenAI;
//     verbatim is exactly the correct semantics for it.
//   - BifrostContextKeyPassthroughExtraParams: enables the official
//     extra-params → wire-body merge inside the shared request helpers
//     (the same mechanism the HTTP router uses with the
//     x-bf-passthrough-extra-params header).
func (provider *LLamaCppProvider) setLLamaCppCtxFlags(ctx *schemas.BifrostContext) {
	if ctx == nil {
		return
	}
	ctx.SetValue(schemas.BifrostContextKeyIsCustomProvider, true)
	ctx.SetValue(schemas.BifrostContextKeyPassthroughExtraParams, true)
}

// llamacppOptions are provider-level options from the reserved `llamacpp`
// sub-object in ExtraParams (removed before the body hits the wire).
type llamacppOptions struct {
	ReasoningPassthrough string // "" = passthrough (default), "none" = never send reasoning_effort
	DisableReasoning     bool   // drop all reasoning params
	IncludeUsage         *bool  // stream_options.include_usage override (default: true)
}

// prepareBifrostRequest normalizes a BifrostChatRequest for llama-server wire
// compatibility. It mutates the given request in place:
//
//  1. Extracts provider-level options from the reserved `llamacpp` sub-object
//     in ExtraParams and removes that sub-object (never sent to wire).
//  2. Computes llama-server's top-level `reasoning_effort` value (verbatim
//     passthrough from Bifrost's Reasoning.Effort, or "none" when
//     Reasoning.Enabled=false) and adds it to ExtraParams so the shared
//     request helper merges it into the wire body.
//
// This must be called BEFORE the shared OpenAI-family handlers so both
// non-streaming (HandleOpenAIChatCompletionRequest) and streaming (my custom
// closure) paths see the normalized ExtraParams.
func (provider *LLamaCppProvider) prepareBifrostRequest(req *schemas.BifrostChatRequest) llamacppOptions {
	opts := llamacppOptions{}

	if req == nil || req.Params == nil {
		return opts
	}

	extraParams := req.Params.ExtraParams
	if extraParams == nil {
		extraParams = make(map[string]interface{})
		req.Params.ExtraParams = extraParams
	}

	// 1. Extract llamacpp provider options, remove the reserved sub-object.
	raw, hasOpts := extraParams["llamacpp"]
	if hasOpts {
		delete(extraParams, "llamacpp")
		if obj, isMap := raw.(map[string]interface{}); isMap {
			if s, ok := obj["reasoning_passthrough"].(string); ok {
				opts.ReasoningPassthrough = s
			}
			if b, ok := obj["disable_reasoning"].(bool); ok {
				opts.DisableReasoning = b
			}
			if b, ok := obj["include_usage"].(bool); ok {
				opts.IncludeUsage = &b
			}
		}
	}

	// 2. Compute reasoning_effort (verbatim passthrough).
	if opts.DisableReasoning {
		// Caller explicitly disabled reasoning via provider options.
		if r := req.Params.Reasoning; r != nil {
			r.Effort = nil
			r.Enabled = nil
		}
		return opts
	}

	if opts.ReasoningPassthrough == "none" {
		// Never send reasoning_effort: clear so the shared marshaller
		// (which maps Reasoning.Effort → reasoning_effort) emits nothing.
		if r := req.Params.Reasoning; r != nil {
			r.Effort = nil
		}
		return opts
	}

	// Explicit extra_params["reasoning_effort"] wins (user's raw llama-server field).
	if existing, ok := extraParams["reasoning_effort"]; ok && existing != nil {
		return opts // Leave as-is.
	}

	if r := req.Params.Reasoning; r != nil {
		if r.Effort != nil && *r.Effort != "" {
			// Verbatim: "xhigh", "max", etc. pass through unchanged.
			extraParams["reasoning_effort"] = *r.Effort
		} else if r.Enabled != nil && !*r.Enabled {
			// Bifrost convention: Enabled=false means "thinking off" → llama-server's "none".
			extraParams["reasoning_effort"] = "none"
		}
	}

	return opts
}

// upstreamErrorMessage parses llama-server's OpenAI-style error envelope
// {"error": {"message", "type", "code"}} for clear, actionable errors.
func upstreamErrorMessage(body []byte) (string, string) {
	var parsed struct {
		Error *struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	msg := strings.TrimSpace(string(body))
	if msg == "" {
		msg = "llama-server returned an error"
	}
	if err := schemas.Unmarshal(body, &parsed); err == nil && parsed.Error != nil {
		if parsed.Error.Message != "" {
			msg = parsed.Error.Message
		}
		return msg, parsed.Error.Type
	}
	if len(msg) > 300 {
		msg = msg[:300] + "..."
	}
	return msg, ""
}

// upstreamAPIError builds a retryable upstream error (IsBifrostError=false)
// from a non-2xx llama-server response.
func upstreamAPIError(prefix string, resp *fasthttp.Response) *schemas.BifrostError {
	msg, errType := upstreamErrorMessage(resp.Body())
	codePtr := (*string)(nil)
	if errType != "" {
		codePtr = &errType
	}
	return providerUtils.NewProviderAPIError(
		fmt.Sprintf("%s upstream error: %s", prefix, msg),
		errors.New(msg),
		resp.StatusCode(),
		codePtr,
		nil,
	)
}

func isContextEOF(err error) bool {
	return err == io.EOF || errors.Is(err, io.EOF)
}

var _ = http.MethodPost
