package main

import (
	"fmt"

	"github.com/dark-eye/bifrost-llama.cpp-provider/provider/llamacpp"
	"github.com/maximhq/bifrost/core/schemas"
)

var (
	providerInstance *llamacpp.LLamaCppProvider
	providerKey      = schemas.ModelProvider("llamacpp")
)

func Init(config any) error {
	fmt.Println("llamacpp plugin: Init called")

	providerKey = schemas.ModelProvider("llamacpp")
	baseURL := "http://127.0.0.1:8080"

	if cfg, ok := config.(map[string]any); ok {
		if pk, ok := cfg["provider_key"].(string); ok && pk != "" {
			providerKey = schemas.ModelProvider(pk)
		}
		if bu, ok := cfg["base_url"].(string); ok && bu != "" {
			baseURL = bu
		}
	}

	providerConfig := &schemas.ProviderConfig{
		NetworkConfig: schemas.NetworkConfig{
			BaseURL: baseURL,
		},
	}

	var err error
	providerInstance, err = llamacpp.NewLLamaCppProvider(providerConfig, nil)
	if err != nil {
		return fmt.Errorf("llamacpp plugin: failed to create provider: %w", err)
	}

	fmt.Printf("llamacpp plugin: initialized with base_url=%s\n", baseURL)
	return nil
}

func GetName() string {
	return "llamacpp-plugin"
}

func Cleanup() error {
	fmt.Println("llamacpp plugin: Cleanup called")
	return nil
}

func PreRequestHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) error {
	provider, _, _ := req.GetRequestFields()
	if provider != providerKey {
		return nil
	}
	return nil
}

func PreLLMHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	provider, model, _ := req.GetRequestFields()

	if provider != providerKey {
		return req, nil, nil
	}

	if req.RequestType != schemas.ChatCompletionRequest &&
		req.RequestType != schemas.ChatCompletionStreamRequest {
		return req, nil, nil
	}

	if providerInstance == nil {
		return req, &schemas.LLMPluginShortCircuit{
			Error: &schemas.BifrostError{
				Error: &schemas.ErrorField{
					Message: "llamacpp provider not initialized",
				},
			},
		}, nil
	}

	chatReq := &schemas.BifrostChatRequest{
		Model:  model,
		Input:  req.ChatRequest.Input,
		Params: req.ChatRequest.Params,
	}

	key := schemas.Key{
		ID:    "llamacpp-plugin",
		Name:  "llamacpp-plugin",
		Value: *schemas.NewSecretVar(""),
	}

	chatResp, chatErr := providerInstance.ChatCompletion(ctx, key, chatReq)

	if chatErr != nil {
		var message string
		if chatErr.Error != nil {
			message = chatErr.Error.Message
		} else {
			message = "llamacpp provider error"
		}

		return req, &schemas.LLMPluginShortCircuit{
			Error: &schemas.BifrostError{
				Error: &schemas.ErrorField{
					Message: message,
				},
			},
		}, nil
	}

	bifrostResp := &schemas.BifrostResponse{
		ChatResponse: chatResp,
	}

	return req, &schemas.LLMPluginShortCircuit{
		Response: bifrostResp,
	}, nil
}

func PostLLMHook(ctx *schemas.BifrostContext, result *schemas.BifrostResponse, err *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	return result, err, nil
}
