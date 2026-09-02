package llamacpp_test

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	llamacpp "github.com/dark-eye/bifrost-llama.cpp-provider/provider/llamacpp"
	schemas "github.com/maximhq/bifrost/core/schemas"
)

// newStreamingFake builds a fake llama-server that emits SSE chat-completion
// chunks for /v1/chat/completions.
func newStreamingFake(t *testing.T) *fakeLLamaServer {
	t.Helper()
	fake := newFakeLLamaServer()
	fake.setHandler("/v1/chat/completions", func(req recordedRequest, w http.ResponseWriter) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			writeJSON(w, http.StatusInternalServerError, `{"error":"no flusher"}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		events := []string{
			`data: {"id":"chatcmpl-s1","object":"chat.completion.chunk","created":1,"model":"qwen3","choices":[{"index":0,"finish_reason":null,"delta":{"role":"assistant","reasoning_content":"Let me think"}}]}` + "\n\n",
			`data: {"id":"chatcmpl-s1","object":"chat.completion.chunk","created":1,"model":"qwen3","choices":[{"index":0,"finish_reason":null,"delta":{"role":"assistant","content":"Hello"}}]}` + "\n\n",
			`data: {"id":"chatcmpl-s1","object":"chat.completion.chunk","created":1,"model":"qwen3","choices":[{"index":0,"finish_reason":null,"delta":{"role":"assistant","content":" world"}}]}` + "\n\n",
			`data: {"id":"chatcmpl-s1","object":"chat.completion.chunk","created":1,"model":"qwen3","choices":[{"index":0,"finish_reason":"stop","delta":{"role":"assistant"}}],"usage":{"prompt_tokens":7,"completion_tokens":9,"total_tokens":16}}` + "\n\n",
			"data: [DONE]\n\n",
		}
		for _, ev := range events {
			if _, err := io.WriteString(w, ev); err != nil {
				return
			}
			flusher.Flush()
		}
	})
	return fake
}

func collectStream(t *testing.T, ch chan *schemas.BifrostStreamChunk) []string {
	t.Helper()
	var parts []string
	for chunk := range ch {
		if chunk == nil || chunk.BifrostChatResponse == nil {
			continue
		}
		for _, choice := range chunk.BifrostChatResponse.Choices {
			if choice.ChatStreamResponseChoice != nil &&
				choice.ChatStreamResponseChoice.Delta != nil &&
				choice.ChatStreamResponseChoice.Delta.Content != nil {
				parts = append(parts, *choice.ChatStreamResponseChoice.Delta.Content)
			}
		}
	}
	return parts
}

func TestChatCompletionStream_ReasoningEffortVerbatim(t *testing.T) {
	fake := newStreamingFake(t)
	defer fake.Close()

	provider := newTestProvider(t, fake)
	ctx := newCtx()
	ctx.SetValue(schemas.BifrostContextKeyIsResponsesToChatCompletionFallback, false)

	effort := "xhigh"
	ch, bifrostErr := provider.ChatCompletionStream(ctx, noopPostHookRunner, nil, testKey(), &schemas.BifrostChatRequest{
		Provider: llamacpp.ProviderKey,
		Model:    "qwen3-30b",
		Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: ptr("hi")}}},
		Params:   &schemas.ChatParameters{Reasoning: &schemas.ChatReasoning{Effort: &effort}},
	})
	if bifrostErr != nil {
		t.Fatalf("ChatCompletionStream failed: %+v", bifrostErr)
	}
	parts := collectStream(t, ch)
	if strings.Join(parts, "") != "Hello world" {
		t.Errorf("streamed content = %q, want %q", strings.Join(parts, ""), "Hello world")
	}

	req := fake.lastRequest()
	if req == nil {
		t.Fatal("no request recorded")
	}
	var body map[string]any
	_ = json.Unmarshal(req.Body, &body)
	if body["stream"] != true {
		t.Errorf("stream flag not set: %s", req.Body)
	}
	if got, _ := body["reasoning_effort"].(string); got != "xhigh" {
		t.Errorf("streaming reasoning_effort = %q, want %q (body=%s)", got, "xhigh", req.Body)
	}
}

func TestChatCompletionStream_ReasoningContentInDelta(t *testing.T) {
	fake := newStreamingFake(t)
	defer fake.Close()

	provider := newTestProvider(t, fake)
	ch, bifrostErr := provider.ChatCompletionStream(newCtx(), noopPostHookRunner, nil, testKey(), &schemas.BifrostChatRequest{
		Provider: llamacpp.ProviderKey,
		Model:    "qwen3-30b",
		Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: ptr("hi")}}},
	})
	if bifrostErr != nil {
		t.Fatalf("ChatCompletionStream failed: %+v", bifrostErr)
	}

	// Collect and verify that reasoning_content was folded into a Reasoning field.
	gotReasoning := false
	for chunk := range ch {
		if chunk == nil || chunk.BifrostChatResponse == nil {
			continue
		}
		for _, choice := range chunk.BifrostChatResponse.Choices {
			if choice.ChatStreamResponseChoice != nil && choice.ChatStreamResponseChoice.Delta != nil {
				if choice.ChatStreamResponseChoice.Delta.Reasoning != nil &&
					strings.Contains(*choice.ChatStreamResponseChoice.Delta.Reasoning, "Let me think") {
					gotReasoning = true
				}
			}
		}
	}
	if !gotReasoning {
		t.Error("reasoning_content from stream delta was not surfaced as Reasoning")
	}
}

// TestListModels_ContextMetadata verifies that /v1/models' per-model `meta`
// (ctx_size) is mapped to Bifrost's ModelInfo metadata.
func TestListModels_ContextMetadata(t *testing.T) {
	fake := newFakeLLamaServer()
	defer fake.Close()

	fake.setHandler("/v1/models", func(req recordedRequest, w http.ResponseWriter) {
		// Mirror llama-server's real /v1/models shape: meta comes from the
		// child process' loaded_info (server-context.cpp get_res_model_info).
		writeJSON(w, http.StatusOK, `{
			"data": [
				{
					"id": "qwen3-30b-a3b",
					"object": "model",
					"owned_by": "llamacpp",
					"created": 1700000000,
					"meta": {
						"n_ctx": 32768,
						"n_ctx_train": 131072,
						"n_embd": 2048,
						"n_params": 30544072832
					}
				},
				{"id": "phi-4-mini", "object": "model"}
			],
			"object": "list"
		}`)
	})

	provider := newTestProvider(t, fake)
	resp, bifrostErr := provider.ListModels(newCtx(), []schemas.Key{testKey()}, &schemas.BifrostListModelsRequest{})
	if bifrostErr != nil {
		t.Fatalf("ListModels failed: %+v", bifrostErr)
	}
	if resp == nil {
		t.Fatal("ListModels returned nil response")
	}
	if len(resp.Data) != 2 {
		t.Fatalf("expected 2 models, got %d", len(resp.Data))
	}

	// Find the model with metadata by ID (response order is not guaranteed).
	var qwen *schemas.Model
	for i := range resp.Data {
		if resp.Data[i].ID == "qwen3-30b-a3b" {
			qwen = &resp.Data[i]
		}
	}
	if qwen == nil {
		t.Fatal("qwen3-30b-a3b model not found in response")
	}
	t.Logf("qwen model id=%v ctx=%v", qwen.ID, qwen.ContextLength)
	if qwen.ContextLength == nil || *qwen.ContextLength != 32768 {
		t.Errorf("qwen model context length = %v, want 32768 (from meta.ctx_size)", qwen.ContextLength)
	}
}

// TestPassthrough_Tokenize verifies the llama-server /tokenize passthrough.
func TestPassthrough_Tokenize(t *testing.T) {
	fake := newFakeLLamaServer()
	defer fake.Close()

	fake.setHandler("/tokenize", func(req recordedRequest, w http.ResponseWriter) {
		writeJSON(w, http.StatusOK, `{"tokens":[1,2,3]}`)
	})

	provider := newTestProvider(t, fake)
	resp, bifrostErr := provider.Passthrough(newCtx(), testKey(), &schemas.BifrostPassthroughRequest{
		Method: "POST",
		Path:   "/tokenize",
		Body:   []byte(`{"content":"hello"}`),
	})
	if bifrostErr != nil {
		t.Fatalf("Passthrough failed: %+v", bifrostErr)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}
	if !strings.Contains(string(resp.Body), `"tokens":[1,2,3]`) {
		t.Errorf("unexpected passthrough body: %s", string(resp.Body))
	}

	req := fake.lastRequest()
	if req == nil || req.Path != "/tokenize" {
		t.Fatalf("expected /tokenize request, got %+v", req)
	}
}

// TestPassthrough_MethodNotAllowed verifies the method allowlist.
func TestPassthrough_MethodNotAllowed(t *testing.T) {
	fake := newFakeLLamaServer()
	defer fake.Close()

	provider := newTestProvider(t, fake)
	_, bifrostErr := provider.Passthrough(newCtx(), testKey(), &schemas.BifrostPassthroughRequest{
		Method: "DELETE",
		Path:   "/health",
	})
	if bifrostErr == nil {
		t.Fatal("expected bifrost error for DELETE method, got nil")
	}
}
