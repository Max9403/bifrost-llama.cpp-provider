package llamacpp

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"time"

	openai "github.com/maximhq/bifrost/core/providers/openai"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	schemas "github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
	"github.com/valyala/fasthttp"
)

// parseLlamaServerChatResponse maps llama-server's OAI chat-completions
// response into Bifrost's canonical chat response.
//
// llama-server's reasoning arrives as `reasoning_content` on the assistant
// message and as `reasoning_content` on stream deltas (tools/server/
// server-chat.cpp). Bifrost's ChatAssistantMessage /
// ChatStreamResponseChoiceDelta unmarshalers both fold `reasoning_content`
// into Reasoning, so decoding straight into the Bifrost type is correct and
// lossless — no custom mapping required.
func (provider *LLamaCppProvider) parseLlamaServerChatResponse(
	respBody []byte,
	out *schemas.BifrostChatResponse,
	rawRequest []byte,
	sendBackRawRequest bool,
	sendBackRawResponse bool,
) (rawReq, rawResp interface{}, bifrostErr *schemas.BifrostError) {
	if len(respBody) == 0 {
		return rawRequest, respBody, providerUtils.NewBifrostOperationError(
			schemas.ErrProviderResponseUnmarshal,
			errors.New("llamacpp: empty response body"))
	}
	if err := schemas.Unmarshal(respBody, out); err != nil {
		return rawRequest, respBody, providerUtils.NewBifrostOperationError(
			schemas.ErrProviderResponseUnmarshal, err)
	}
	out.ExtraFields.Provider = provider.GetProviderKey()
	if out.Model == "" {
		out.Model = "llamacpp"
	}
	if out.Object == "" {
		out.Object = "chat.completion"
	}
	if out.Created == 0 {
		out.Created = int(time.Now().Unix())
	}
	if sendBackRawRequest {
		out.ExtraFields.RawRequest = rawRequest
	}
	if sendBackRawResponse {
		out.ExtraFields.RawResponse = respBody
	}
	return rawRequest, respBody, nil
}

// llamaServerErrorConverter maps llama-server error responses (OpenAI-style
// {"error": {"message","type","code"}}) into a retryable BifrostError with
// the upstream status code preserved.
var llamaServerErrorConverter openai.ErrorConverter = func(resp *fasthttp.Response) *schemas.BifrostError {
	return upstreamAPIError("llamacpp", resp)
}

// ChatCompletion performs a chat completion request to llama-server.
func (provider *LLamaCppProvider) ChatCompletion(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostChatRequest) (*schemas.BifrostChatResponse, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	provider.setLLamaCppCtxFlags(ctx)

	// Normalize the request for llama-server wire compatibility (strip llamacpp,
	// compute reasoning_effort). This mutates request.Params in place so the
	// shared handler picks up the normalized ExtraParams.
	if request != nil {
		provider.prepareBifrostRequest(request)
	}

	return openai.HandleOpenAIChatCompletionRequest(
		ctx,
		provider.client,
		baseURL+providerUtils.GetPathFromContext(ctx, "/v1/chat/completions"),
		request,
		provider.authHeader(key),
		provider.networkConfig.ExtraHeaders,
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		provider.GetProviderKey(),
		func(respBody []byte, response *schemas.BifrostChatResponse, reqBody []byte, sendBackRawRequest bool, sendBackRawResponse bool) (interface{}, interface{}, *schemas.BifrostError) {
			return provider.parseLlamaServerChatResponse(respBody, response, reqBody, sendBackRawRequest, sendBackRawResponse)
		},
		llamaServerErrorConverter,
		nil, // signer
		provider.logger,
	)
}

// ChatCompletionStream performs a streaming chat completion request to
// llama-server using its SSE format (data: <json>\n\n … data: [DONE]).
//
// True chunk-by-chunk streaming through Bifrost's post-hook pipeline: each SSE
// delta becomes a BifrostStreamChunk; the trailing usage is attached to the
// final synthetic chunk exactly like the built-in providers do.
func (provider *LLamaCppProvider) ChatCompletionStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, postHookSpanFinalizer func(context.Context), key schemas.Key, request *schemas.BifrostChatRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	provider.setLLamaCppCtxFlags(ctx)

	// Normalize the request for llama-server wire compatibility.
	if request != nil {
		provider.prepareBifrostRequest(request)
	}

	providerUtils.SetStreamIdleTimeoutIfEmpty(ctx, provider.networkConfig.StreamIdleTimeoutInSeconds)

	sendBackRawRequest := providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest)
	sendBackRawResponse := providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse)

	jsonBody, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
		ctx,
		request,
		func() (providerUtils.RequestBodyWithExtraParams, error) {
			reqBody := openai.ToOpenAIChatRequest(ctx, request)
			if reqBody == nil {
				return nil, errors.New("llamacpp: failed to convert chat request")
			}
			// llama-server requires stream=true for SSE responses.
			stream := true
			reqBody.Stream = &stream
			return reqBody, nil
		})
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	resp.StreamBody = true
	defer fasthttp.ReleaseRequest(req)

	activeClient := providerUtils.PrepareResponseStreaming(ctx, provider.streamingClient, resp)

	req.Header.SetMethod(http.MethodPost)
	req.SetRequestURI(baseURL + providerUtils.GetPathFromContext(ctx, "/v1/chat/completions"))
	req.Header.SetContentType("application/json")
	providerUtils.SetExtraHeaders(ctx, req, provider.networkConfig.ExtraHeaders, nil)
	for k, v := range provider.authHeader(key) {
		req.Header.Set(k, v)
	}
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Cache-Control", "no-cache")
	if !providerUtils.ApplyLargePayloadRequestBodyWithModelNormalization(ctx, req, provider.GetProviderKey()) {
		req.SetBody(jsonBody)
	}

	startTime := time.Now()
	err := providerUtils.DoStreamingRequest(ctx, activeClient, req, resp)
	latency := time.Since(startTime)
	if err != nil {
		defer providerUtils.ReleaseStreamingResponse(ctx, resp)
		if errors.Is(err, context.Canceled) {
			return nil, providerUtils.EnrichError(ctx, &schemas.BifrostError{
				IsBifrostError: false,
				Error:          &schemas.ErrorField{Type: schemas.Ptr(schemas.RequestCancelled), Message: schemas.ErrRequestCancelled, Error: err},
			}, jsonBody, nil, sendBackRawRequest, sendBackRawResponse, latency)
		}
		if errors.Is(err, fasthttp.ErrTimeout) || errors.Is(err, context.DeadlineExceeded) {
			return nil, providerUtils.EnrichError(ctx, providerUtils.NewBifrostTimeoutError(schemas.ErrProviderRequestTimedOut, err), jsonBody, nil, sendBackRawRequest, sendBackRawResponse, latency)
		}
		return nil, providerUtils.EnrichError(ctx, providerUtils.NewBifrostUpstreamConnectionError(schemas.ErrProviderDoRequest, err), jsonBody, nil, sendBackRawRequest, sendBackRawResponse, latency)
	}

	ctx.SetValue(schemas.BifrostContextKeyProviderResponseHeaders, providerUtils.ExtractProviderResponseHeaders(resp))

	if resp.StatusCode() != fasthttp.StatusOK {
		defer providerUtils.ReleaseStreamingResponse(ctx, resp)
		providerUtils.MaterializeStreamErrorBody(ctx, resp)
		return nil, providerUtils.EnrichError(ctx, llamaServerErrorConverter(resp), jsonBody, nil, sendBackRawRequest, sendBackRawResponse, latency)
	}

	if providerUtils.SetupStreamingPassthrough(ctx, resp) {
		responseChan := make(chan *schemas.BifrostStreamChunk)
		providerUtils.CloseStream(ctx, responseChan)
		return responseChan, nil
	}

	responseChan := make(chan *schemas.BifrostStreamChunk, schemas.DefaultStreamBufferSize)

	go func() {
		defer providerUtils.EnsureStreamFinalizerCalled(ctx, postHookSpanFinalizer)
		defer func() {
			if ctx.Err() == context.Canceled {
				providerUtils.HandleStreamCancellation(ctx, postHookRunner, responseChan, provider.logger, postHookSpanFinalizer, jsonBody)
			} else if ctx.Err() == context.DeadlineExceeded {
				providerUtils.HandleStreamTimeout(ctx, postHookRunner, responseChan, provider.logger, postHookSpanFinalizer, jsonBody)
			}
			providerUtils.CloseStream(ctx, responseChan)
		}()
		defer providerUtils.ReleaseStreamingResponse(ctx, resp)

		reader, releaseGzip := providerUtils.DecompressStreamBody(resp)
		defer releaseGzip()

		reader, stopIdleTimeout := providerUtils.NewIdleTimeoutReader(reader, resp.BodyStream(), providerUtils.GetStreamIdleTimeout(ctx), ctx)
		defer stopIdleTimeout()

		stopCancellation := providerUtils.SetupStreamCancellation(ctx, resp.BodyStream(), provider.logger)
		defer stopCancellation()

		reader, drained := providerUtils.DrainNonSSEStreamReader(resp, reader)
		if drained {
			ctx.SetValue(schemas.BifrostContextKeyStreamEndIndicator, true)
			providerUtils.ProcessAndSendError(ctx, postHookRunner, errors.New("llamacpp: provider returned non-SSE response for streaming request"), responseChan, provider.logger, postHookSpanFinalizer)
			return
		}

		sseReader := providerUtils.GetSSEDataReader(ctx, reader)

		chunkIndex := -1
		usage := &schemas.BifrostLLMUsage{}
		ctx.SetValue(schemas.BifrostContextKeyStreamAccumulatedUsage, usage)
		lastChunkTime := startTime

		var finishReason *string
		var messageID string
		var modelName string
		var created int
		forwardedTerminalFinishReason := false

		for {
			if ctx.Err() != nil {
				return
			}
			data, readErr := sseReader.ReadDataLine()
			if readErr != nil {
				if ctx.Err() != nil {
					return
				}
				if isContextEOF(readErr) {
					break
				}
				ctx.SetValue(schemas.BifrostContextKeyStreamEndIndicator, true)
				provider.logger.Warn("llamacpp: error reading stream: %v", readErr)
				providerUtils.ProcessAndSendError(ctx, postHookRunner, readErr, responseChan, provider.logger, postHookSpanFinalizer)
				return
			}
			jsonData := string(data)

			// Inline error object ({"error": {...}})
			if errorNode := gjson.Get(jsonData, "error"); errorNode.Exists() {
				var bifrostErr schemas.BifrostError
				if err := json.Unmarshal([]byte(jsonData), &bifrostErr); err == nil {
					if bifrostErr.Error != nil && bifrostErr.Error.Message != "" {
						ctx.SetValue(schemas.BifrostContextKeyStreamEndIndicator, true)
						providerUtils.ProcessAndSendBifrostError(ctx, postHookRunner, providerUtils.EnrichError(ctx, &bifrostErr, jsonBody, nil, sendBackRawRequest, sendBackRawResponse, latency), responseChan, provider.logger, postHookSpanFinalizer)
						return
					}
				}
			}

			var response schemas.BifrostChatResponse
			if err := json.Unmarshal([]byte(jsonData), &response); err != nil {
				provider.logger.Warn("llamacpp: failed to parse stream chunk: %v", err)
				continue
			}
			if response.Choices == nil {
				response.Choices = []schemas.BifrostResponseChoice{}
			}
			response.ExtraFields.Provider = provider.GetProviderKey()

			if response.Usage != nil {
				accumulateLLamaUsage(usage, response.Usage)
				response.Usage = nil
			}
			if response.Model != "" {
				modelName = response.Model
			}
			if len(response.Choices) == 0 {
				continue
			}

			choice := response.Choices[0]
			if choice.FinishReason != nil && *choice.FinishReason != "" {
				finishReason = choice.FinishReason
			}
			if response.ID != "" && messageID == "" {
				messageID = response.ID
			}
			if response.Created != 0 && created == 0 {
				created = response.Created
			}

			if choice.ChatStreamResponseChoice != nil &&
				choice.ChatStreamResponseChoice.Delta != nil &&
				((choice.ChatStreamResponseChoice.Delta.Content != nil && *choice.ChatStreamResponseChoice.Delta.Content != "") ||
					choice.ChatStreamResponseChoice.Delta.Reasoning != nil ||
					len(choice.ChatStreamResponseChoice.Delta.ReasoningDetails) > 0 ||
					len(choice.ChatStreamResponseChoice.Delta.ToolCalls) > 0) {
				if choice.FinishReason != nil && *choice.FinishReason != "" {
					forwardedTerminalFinishReason = true
				}
				chunkIndex++
				response.ExtraFields.ChunkIndex = chunkIndex
				response.ExtraFields.Latency = time.Since(lastChunkTime).Milliseconds()
				lastChunkTime = time.Now()
				if sendBackRawResponse {
					response.ExtraFields.RawResponse = jsonData
				}
				providerUtils.ProcessAndSendResponse(ctx, postHookRunner, providerUtils.GetBifrostResponseForStreamResponse(nil, &response, nil, nil, nil, nil), responseChan, postHookSpanFinalizer)
			}
		}

		// Truncation detection: llama-server ends with [DONE]; without it the
		// stream is treated as truncated (mirrors built-in providers).
		terminalSignalSeen := providerUtils.SSEStreamEndedOnMarker(sseReader) || finishReason != nil
		if !terminalSignalSeen {
			providerUtils.SendStreamTruncatedError(ctx, postHookRunner, responseChan, provider.logger, postHookSpanFinalizer, jsonBody)
			return
		}

		finalFinishReason := finishReason
		if forwardedTerminalFinishReason {
			finalFinishReason = nil
		}
		finalResponse := providerUtils.CreateBifrostChatCompletionChunkResponse(messageID, usage, finalFinishReason, chunkIndex, modelName, created)
		finalResponse.ExtraFields.Provider = provider.GetProviderKey()
		if sendBackRawRequest {
			providerUtils.ParseAndSetRawRequest(&finalResponse.ExtraFields, jsonBody)
		}
		finalResponse.ExtraFields.Latency = time.Since(startTime).Milliseconds()
		ctx.SetValue(schemas.BifrostContextKeyStreamEndIndicator, true)
		providerUtils.ProcessAndSendResponse(ctx, postHookRunner, providerUtils.GetBifrostResponseForStreamResponse(nil, finalResponse, nil, nil, nil, nil), responseChan, postHookSpanFinalizer)
	}()

	return responseChan, nil
}

// Responses performs a completion request using the Responses API by
// delegating to the chat completion endpoint (llama-server's /v1/responses is
// version-dependent; chat completions is the stable, documented contract).
func (provider *LLamaCppProvider) Responses(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostResponsesRequest) (*schemas.BifrostResponsesResponse, *schemas.BifrostError) {
	chatResponse, err := provider.ChatCompletion(ctx, key, request.ToChatRequest())
	if err != nil {
		return nil, err
	}
	return chatResponse.ToBifrostResponsesResponse(), nil
}

// ResponsesStream performs a streaming Responses API request via the chat
// completions SSE stream.
func (provider *LLamaCppProvider) ResponsesStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, postHookSpanFinalizer func(context.Context), key schemas.Key, request *schemas.BifrostResponsesRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	ctx.SetValue(schemas.BifrostContextKeyIsResponsesToChatCompletionFallback, true)
	return provider.ChatCompletionStream(ctx, postHookRunner, postHookSpanFinalizer, key, request.ToChatRequest())
}

// CountTokens counts tokens using llama-server's stable
// /v1/chat/completions/input_tokens endpoint (returns {"input_tokens": N,
// "object": "response.input_tokens"}).
func (provider *LLamaCppProvider) CountTokens(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostResponsesRequest) (*schemas.BifrostCountTokensResponse, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	chatReq := request.ToChatRequest()
	if chatReq == nil || len(chatReq.Input) == 0 {
		return nil, providerUtils.NewBifrostOperationError(
			"llamacpp: count-tokens requires a non-empty input", nil)
	}

	messagesBody, err := schemas.Marshal(chatReq.Input)
	if err != nil {
		return nil, providerUtils.NewBifrostOperationError("llamacpp: failed to marshal messages for token counting", err)
	}
	countBody, err := schemas.Marshal(map[string]interface{}{
		"model":    chatReq.Model,
		"messages": json.RawMessage(messagesBody),
	})
	if err != nil {
		return nil, providerUtils.NewBifrostOperationError("llamacpp: failed to marshal count-tokens request", err)
	}

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)

	req.Header.SetMethod(http.MethodPost)
	req.SetRequestURI(baseURL + "/v1/chat/completions/input_tokens")
	req.Header.SetContentType("application/json")
	for k, v := range provider.authHeader(key) {
		req.Header.Set(k, v)
	}
	providerUtils.SetExtraHeaders(ctx, req, provider.networkConfig.ExtraHeaders, nil)
	req.SetBody(countBody)

	startTime := time.Now()
	_, bifrostErr, wait := providerUtils.MakeRequestWithContext(ctx, provider.client, req, resp)
	defer wait()
	latency := time.Since(startTime)
	if bifrostErr != nil {
		return nil, providerUtils.EnrichError(ctx, bifrostErr, countBody, nil, true, true, latency)
	}
	if resp.StatusCode() != fasthttp.StatusOK {
		return nil, upstreamAPIError("llamacpp", resp)
	}

	var countResp struct {
		InputTokens int    `json:"input_tokens"`
		Object      string `json:"object"`
	}
	if err := schemas.Unmarshal(resp.Body(), &countResp); err != nil {
		return nil, providerUtils.NewBifrostOperationError(schemas.ErrProviderResponseUnmarshal, err)
	}

	total := countResp.InputTokens
	return &schemas.BifrostCountTokensResponse{
		Object:      countResp.Object,
		Model:       chatReq.Model,
		InputTokens: countResp.InputTokens,
		TotalTokens: &total,
		ExtraFields: schemas.BifrostResponseExtraFields{
			Provider: provider.GetProviderKey(),
			Latency:  latency.Milliseconds(),
		},
	}, nil
}

// Embedding performs an embedding request to llama-server's /v1/embeddings
// (requires llama-server started with --embedding).
func (provider *LLamaCppProvider) Embedding(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostEmbeddingRequest) (*schemas.BifrostEmbeddingResponse, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	if request == nil || request.Input == nil {
		return nil, providerUtils.NewBifrostOperationError(
			"llamacpp: embedding request requires input", nil)
	}

	var inputs []string
	switch {
	case request.Input.Text != nil:
		inputs = []string{*request.Input.Text}
	case len(request.Input.Texts) > 0:
		inputs = request.Input.Texts
	default:
		return nil, providerUtils.NewBifrostOperationError(
			"llamacpp: embedding input must be text or texts (token-id embedding is not supported by llama-server)", nil)
	}

	extraParams := map[string]interface{}{}
	if request.Params != nil {
		if request.Params.EncodingFormat != nil {
			extraParams["encoding_format"] = *request.Params.EncodingFormat
		}
		if request.Params.Dimensions != nil {
			extraParams["dimensions"] = *request.Params.Dimensions
		}
		for k, v := range request.Params.ExtraParams {
			extraParams[k] = v
		}
	}

	embedBody, err := schemas.Marshal(map[string]interface{}{
		"model": request.Model,
		"input": inputs,
	})
	if err != nil {
		return nil, providerUtils.NewBifrostOperationError("llamacpp: failed to marshal embedding request", err)
	}
	if len(extraParams) > 0 {
		embedBody, err = providerUtils.MergeExtraParamsIntoJSON(embedBody, extraParams)
		if err != nil {
			return nil, providerUtils.NewBifrostOperationError("llamacpp: failed to merge embedding extra params", err)
		}
	}

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)

	req.Header.SetMethod(http.MethodPost)
	req.SetRequestURI(baseURL + "/v1/embeddings")
	req.Header.SetContentType("application/json")
	for k, v := range provider.authHeader(key) {
		req.Header.Set(k, v)
	}
	providerUtils.SetExtraHeaders(ctx, req, provider.networkConfig.ExtraHeaders, nil)
	req.SetBody(embedBody)

	startTime := time.Now()
	_, bifrostErr, wait := providerUtils.MakeRequestWithContext(ctx, provider.client, req, resp)
	defer wait()
	latency := time.Since(startTime)
	if bifrostErr != nil {
		return nil, providerUtils.EnrichError(ctx, bifrostErr, embedBody, nil, true, true, latency)
	}
	if resp.StatusCode() != fasthttp.StatusOK {
		return nil, upstreamAPIError("llamacpp", resp)
	}

	var embedResp schemas.BifrostEmbeddingResponse
	if err := schemas.Unmarshal(resp.Body(), &embedResp); err != nil {
		return nil, providerUtils.NewBifrostOperationError(schemas.ErrProviderResponseUnmarshal, err)
	}
	embedResp.ExtraFields.Provider = provider.GetProviderKey()
	embedResp.ExtraFields.Latency = latency.Milliseconds()
	return &embedResp, nil
}

// TextCompletion performs a text completion request to llama-server.
// llama-server's completions surface is OpenAI-compatible; reasoning params do
// not apply here (chat templating does not run), so no effort promotion.
func (provider *LLamaCppProvider) TextCompletion(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostTextCompletionRequest) (*schemas.BifrostTextCompletionResponse, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	provider.setLLamaCppCtxFlags(ctx)
	return openai.HandleOpenAITextCompletionRequest(
		ctx,
		provider.client,
		baseURL+providerUtils.GetPathFromContext(ctx, "/v1/completions"),
		request,
		provider.authHeader(key),
		provider.networkConfig.ExtraHeaders,
		provider.GetProviderKey(),
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		nil,
		llamaServerErrorConverter,
		provider.logger,
	)
}

// TextCompletionStream performs a streaming text completion request to
// llama-server.
func (provider *LLamaCppProvider) TextCompletionStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, postHookSpanFinalizer func(context.Context), key schemas.Key, request *schemas.BifrostTextCompletionRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	provider.setLLamaCppCtxFlags(ctx)
	return openai.HandleOpenAITextCompletionStreaming(
		ctx,
		provider.streamingClient,
		baseURL+providerUtils.GetPathFromContext(ctx, "/v1/completions"),
		request,
		provider.authHeader(key),
		provider.networkConfig.ExtraHeaders,
		provider.networkConfig.StreamIdleTimeoutInSeconds,
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		provider.GetProviderKey(),
		llamaServerErrorConverter,
		postHookRunner,
		nil,
		nil,
		provider.logger,
		postHookSpanFinalizer,
	)
}

func accumulateLLamaUsage(dst *schemas.BifrostLLMUsage, src *schemas.BifrostLLMUsage) {
	if src == nil {
		return
	}
	if src.PromptTokens > dst.PromptTokens {
		dst.PromptTokens = src.PromptTokens
	}
	if src.CompletionTokens > dst.CompletionTokens {
		dst.CompletionTokens = src.CompletionTokens
	}
	if src.TotalTokens > dst.TotalTokens {
		dst.TotalTokens = src.TotalTokens
	}
	if calculated := src.PromptTokens + src.CompletionTokens; calculated > dst.TotalTokens {
		dst.TotalTokens = calculated
	}
	if src.PromptTokensDetails != nil {
		dst.PromptTokensDetails = src.PromptTokensDetails
	}
	if src.CompletionTokensDetails != nil {
		dst.CompletionTokensDetails = src.CompletionTokensDetails
	}
}
