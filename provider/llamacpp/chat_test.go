package llamacpp_test

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	llamacpp "github.com/dark-eye/bifrost-llama.cpp-provider/provider/llamacpp"
	schemas "github.com/maximhq/bifrost/core/schemas"
)

// newTestProvider builds a provider wired to a fake llama-server.
func newTestProvider(t *testing.T, fake *fakeLLamaServer) *llamacpp.LLamaCppProvider {
	t.Helper()
	config := &schemas.ProviderConfig{
		NetworkConfig: schemas.NetworkConfig{
			BaseURL:                        fake.URL,
			AllowPrivateNetwork:            true,
			DefaultRequestTimeoutInSeconds: 15,
			StreamIdleTimeoutInSeconds:     30,
			MaxConnsPerHost:                10,
			KeepAliveTimeoutInSeconds:      60,
		},
	}
	provider, err := llamacpp.NewLLamaCppProvider(config, nil)
	if err != nil {
		t.Fatalf("NewLLamaCppProvider failed: %v", err)
	}
	return provider
}

// noopPostHookRunner is an identity PostHookRunner for tests (Bifrost's
// real one runs plugin hooks; tests have none).
func noopPostHookRunner(ctx *schemas.BifrostContext, result *schemas.BifrostResponse, err *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError) {
	return result, err
}

func newCtx() *schemas.BifrostContext {
	return schemas.NewBifrostContext(context.Background(), time.Now().Add(30*time.Second))
}

func testKey() schemas.Key {
	return schemas.Key{
		ID:    "test-key",
		Name:  "test",
		Value: *schemas.NewSecretVar("test-api-key"),
	}
}

// TestChatCompletion_ReasoningEffortVerbatim is the core requirement:
// params.reasoning.effort must reach llama-server's `reasoning_effort`
// VERBATIM — including non-standard values like "xhigh" and "max" that the
// OpenAI/Gemini providers would silently downgrade.
func TestChatCompletion_ReasoningEffortVerbatim(t *testing.T) {
	fake := newFakeLLamaServer()
	defer fake.Close()

	fake.setHandler("/v1/chat/completions", func(req recordedRequest, w http.ResponseWriter) {
		writeJSON(w, 200, `{
			"id": "chatcmpl-1",
			"object": "chat.completion",
			"created": 1700000000,
			"model": "qwen3-30b-a3b",
			"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hello"}}],
			"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
		}`)
	})

	provider := newTestProvider(t, fake)
	ctx := newCtx()

	for _, effort := range []string{"xhigh", "max", "custom_effort_2026"} {
		ctx = newCtx()
		resp, bifrostErr := provider.ChatCompletion(ctx, testKey(), &schemas.BifrostChatRequest{
			Provider: llamacpp.ProviderKey,
			Model:    "qwen3-30b-a3b",
			Input: []schemas.ChatMessage{
				{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: &[]string{"test"}[0]}},
			},
			Params: &schemas.ChatParameters{
				Reasoning: &schemas.ChatReasoning{Effort: &[]string{effort}[0]},
			},
		})
		if bifrostErr != nil {
			t.Fatalf("ChatCompletion(effort=%q) failed: %+v", effort, bifrostErr)
		}
		if resp == nil || len(resp.Choices) == 0 ||
			resp.Choices[0].ChatNonStreamResponseChoice == nil ||
			resp.Choices[0].ChatNonStreamResponseChoice.Message == nil ||
			resp.Choices[0].ChatNonStreamResponseChoice.Message.Content == nil ||
			resp.Choices[0].ChatNonStreamResponseChoice.Message.Content.ContentStr == nil ||
			*resp.Choices[0].ChatNonStreamResponseChoice.Message.Content.ContentStr != "hello" {
			t.Fatalf("unexpected response for effort=%q: %+v", effort, resp)
		}

		req := fake.lastRequest()
		if req == nil {
			t.Fatalf("no request recorded")
		}
		var body map[string]any
		if err := json.Unmarshal(req.Body, &body); err != nil {
			t.Fatalf("bad wire body: %v", err)
		}
		got, ok := body["reasoning_effort"].(string)
		if !ok {
			t.Fatalf("effort=%q: reasoning_effort missing from wire body: %s", effort, req.Body)
		}
		if got != effort {
			t.Errorf("effort NOT verbatim: sent=%q got=%q (body=%s)", effort, got, req.Body)
		}
		// The reserved llamacpp options object must never leak to the wire.
		if _, leaked := body["llamacpp"]; leaked {
			t.Errorf("llamacpp provider options leaked onto the wire: %s", req.Body)
		}
	}
}

// TestChatCompletion_NoEffortNoField ensures reasoning_effort is absent when
// the caller did not provide reasoning params.
func TestChatCompletion_NoEffortNoField(t *testing.T) {
	fake := newFakeLLamaServer()
	defer fake.Close()

	fake.setHandler("/v1/chat/completions", func(req recordedRequest, w http.ResponseWriter) {
		writeJSON(w, 200, `{
			"id": "chatcmpl-2", "object": "chat.completion", "created": 1,
			"model": "m",
			"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
			"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
		}`)
	})

	provider := newTestProvider(t, fake)
	_, bifrostErr := provider.ChatCompletion(newCtx(), testKey(), &schemas.BifrostChatRequest{
		Provider: llamacpp.ProviderKey,
		Model:    "m",
		Input: []schemas.ChatMessage{
			{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: ptr("hi")}},
		},
	})
	if bifrostErr != nil {
		t.Fatalf("ChatCompletion failed: %+v", bifrostErr)
	}
	req := fake.lastRequest()
	var body map[string]any
	if err := json.Unmarshal(req.Body, &body); err != nil {
		t.Fatalf("bad body: %v", err)
	}
	if _, present := body["reasoning_effort"]; present {
		t.Errorf("reasoning_effort should not be sent when not requested: %s", req.Body)
	}
}

// TestChatCompletion_EnabledFalseSendsNone: reasoning.enabled=false maps to
// llama-server's documented "none" spelling.
func TestChatCompletion_EnabledFalseSendsNone(t *testing.T) {
	fake := newFakeLLamaServer()
	defer fake.Close()

	fake.setHandler("/v1/chat/completions", func(req recordedRequest, w http.ResponseWriter) {
		writeJSON(w, 200, `{
			"id": "chatcmpl-3", "object": "chat.completion", "created": 1,
			"model": "m",
			"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
			"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
		}`)
	})

	provider := newTestProvider(t, fake)
	false_ := false
	_, bifrostErr := provider.ChatCompletion(newCtx(), testKey(), &schemas.BifrostChatRequest{
		Provider: llamacpp.ProviderKey,
		Model:    "m",
		Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: ptr("hi")}}},
		Params:   &schemas.ChatParameters{Reasoning: &schemas.ChatReasoning{Enabled: &false_}},
	})
	if bifrostErr != nil {
		t.Fatalf("ChatCompletion failed: %+v", bifrostErr)
	}
	req := fake.lastRequest()
	var body map[string]any
	_ = json.Unmarshal(req.Body, &body)
	if got, _ := body["reasoning_effort"].(string); got != "none" {
		t.Errorf("expected reasoning_effort=none for enabled=false, got %q", got)
	}
}

// TestChatCompletion_ExtraParamsPassthrough verifies Bifrost's official
// extra_params mechanism reaches llama-server as native top-level fields
// (grammar, top_k, chat_template_kwargs...).
func TestChatCompletion_ExtraParamsPassthrough(t *testing.T) {
	fake := newFakeLLamaServer()
	defer fake.Close()

	fake.setHandler("/v1/chat/completions", func(req recordedRequest, w http.ResponseWriter) {
		writeJSON(w, 200, `{
			"id": "chatcmpl-4", "object": "chat.completion", "created": 1,
			"model": "m",
			"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
			"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
		}`)
	})

	provider := newTestProvider(t, fake)
	grammar := "(expr: (term ('+' term)*))"
	topK := 40
	_, bifrostErr := provider.ChatCompletion(newCtx(), testKey(), &schemas.BifrostChatRequest{
		Provider: llamacpp.ProviderKey,
		Model:    "m",
		Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: ptr("hi")}}},
		Params: &schemas.ChatParameters{
			ExtraParams: map[string]any{
				"grammar":              grammar,
				"top_k":                topK,
				"llamacpp":             map[string]any{"include_usage": false},
				"chat_template_kwargs": map[string]any{"enable_thinking": false},
			},
		},
	})
	if bifrostErr != nil {
		t.Fatalf("ChatCompletion failed: %+v", bifrostErr)
	}
	req := fake.lastRequest()
	var body map[string]any
	if err := json.Unmarshal(req.Body, &body); err != nil {
		t.Fatalf("bad body: %v (body=%s)", err, req.Body)
	}
	if got, _ := body["grammar"].(string); got != grammar {
		t.Errorf("grammar not passed through: got %q", got)
	}
	if got, ok := body["top_k"].(float64); !ok || int(got) != topK {
		t.Errorf("top_k not passed through: got %v", body["top_k"])
	}
	if _, ok := body["chat_template_kwargs"].(map[string]any); !ok {
		t.Errorf("chat_template_kwargs not passed through: %s", req.Body)
	}
	if _, leaked := body["llamacpp"]; leaked {
		t.Errorf("llamacpp options leaked to wire: %s", req.Body)
	}
}

// TestChatCompletion_UpstreamError verifies the error envelope is parsed and
// the status code is preserved.
func TestChatCompletion_UpstreamError(t *testing.T) {
	fake := newFakeLLamaServer()
	defer fake.Close()

	fake.setHandler("/v1/chat/completions", func(req recordedRequest, w http.ResponseWriter) {
		writeJSON(w, 400, `{"error":{"message":"tools param requires --jinja flag","type":"invalid_request_error"}}`)
	})

	provider := newTestProvider(t, fake)
	_, bifrostErr := provider.ChatCompletion(newCtx(), testKey(), &schemas.BifrostChatRequest{
		Provider: llamacpp.ProviderKey,
		Model:    "m",
		Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: ptr("hi")}}},
	})
	if bifrostErr == nil {
		t.Fatal("expected bifrost error, got nil")
	}
	if bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != 400 {
		t.Errorf("expected status 400, got %v", bifrostErr.StatusCode)
	}
	msg := ""
	if bifrostErr.Error != nil {
		msg = bifrostErr.Error.Message
	}
	if !strings.Contains(msg, "--jinja") {
		t.Errorf("upstream error message lost: %q", msg)
	}
}

// TestChatCompletion_AuthHeader verifies Bearer auth for --api-key servers.
func TestChatCompletion_AuthHeader(t *testing.T) {
	fake := newFakeLLamaServer()
	defer fake.Close()

	fake.setHandler("/v1/chat/completions", func(req recordedRequest, w http.ResponseWriter) {
		writeJSON(w, 200, `{
			"id": "chatcmpl-5", "object": "chat.completion", "created": 1,
			"model": "m",
			"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
			"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
		}`)
	})

	provider := newTestProvider(t, fake)
	_, bifrostErr := provider.ChatCompletion(newCtx(), testKey(), &schemas.BifrostChatRequest{
		Provider: llamacpp.ProviderKey,
		Model:    "m",
		Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: ptr("hi")}}},
	})
	if bifrostErr != nil {
		t.Fatalf("ChatCompletion failed: %+v", bifrostErr)
	}
	req := fake.lastRequest()
	if got := req.Headers["Authorization"]; got != "Bearer test-api-key" {
		t.Errorf("Authorization header = %q, want %q", got, "Bearer test-api-key")
	}
}

func ptr(s string) *string { return &s }
