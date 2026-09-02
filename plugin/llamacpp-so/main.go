package main

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/maximhq/bifrost/core/schemas"
)

// Plugin configuration
var providerPrefixes []string

func Init(config any) error {
	fmt.Println("llamacpp-reasoning-effort plugin: Init called")

	// Default: intercept llama-local/ models
	providerPrefixes = []string{"llama-local/"}

	if cfg, ok := config.(map[string]any); ok {
		if prefixes, ok := cfg["provider_prefixes"].([]any); ok {
			providerPrefixes = nil
			for _, p := range prefixes {
				if prefix, ok := p.(string); ok && prefix != "" {
					providerPrefixes = append(providerPrefixes, prefix)
				}
			}
		}
	}

	fmt.Printf("llamacpp-reasoning-effort plugin: intercepting models with prefixes: %v\n", providerPrefixes)
	return nil
}

func GetName() string {
	return "llamacpp-reasoning-effort"
}

func Cleanup() error {
	fmt.Println("llamacpp-reasoning-effort plugin: Cleanup called")
	return nil
}

// HTTPTransportPreHook intercepts requests BEFORE Bifrost parses them,
// moving reasoning_effort into chat_template_kwargs to prevent normalization.
func HTTPTransportPreHook(ctx *schemas.BifrostContext, req *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
	// Only intercept chat completion endpoints
	if req.Path != "/v1/chat/completions" && req.Path != "/openai/chat/completions" {
		return nil, nil
	}

	// Parse request body
	var body map[string]interface{}
	if err := json.Unmarshal(req.Body, &body); err != nil {
		return nil, nil
	}

	// Extract model name
	model, _ := body["model"].(string)
	if !matchesPrefix(model) {
		return nil, nil
	}

	// Extract reasoning_effort
	effort, ok := body["reasoning_effort"].(string)
	if !ok || effort == "" {
		return nil, nil
	}

	// Move into chat_template_kwargs
	templateKwargs, ok := body["chat_template_kwargs"].(map[string]interface{})
	if !ok {
		templateKwargs = make(map[string]interface{})
	}
	templateKwargs["reasoning_effort"] = effort
	body["chat_template_kwargs"] = templateKwargs

	// Delete root-level reasoning_effort to prevent normalization
	delete(body, "reasoning_effort")

	// Re-serialize
	newBody, err := json.Marshal(body)
	if err != nil {
		return nil, nil
	}
	req.Body = newBody

	// Enable passthrough of provider-specific fields
	if req.Headers == nil {
		req.Headers = make(map[string]string)
	}
	req.Headers["x-bf-passthrough-extra-params"] = "true"

	ctx.Log(schemas.LogLevelDebug, fmt.Sprintf("moved reasoning_effort into chat_template_kwargs for model %s", model))

	return nil, nil
}

// PreLLMHook not used (we handle everything at HTTP transport level)
func PreLLMHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	return req, nil, nil
}

// PostLLMHook not used
func PostLLMHook(ctx *schemas.BifrostContext, resp *schemas.BifrostResponse, bifrostErr *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	return resp, bifrostErr, nil
}

func matchesPrefix(model string) bool {
	for _, prefix := range providerPrefixes {
		if strings.HasPrefix(model, prefix) {
			return true
		}
	}
	return false
}
