package llamacpp

import (
	"context"
	"strings"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	schemas "github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// Passthrough exposes llama-server's raw HTTP surface through Bifrost's
// passthrough mechanism. The provider's own native routes are:
//
//	GET  /health
//	GET  /props
//	POST /tokenize
//	POST /detokenize
//	GET  /slots
//	POST /chat/completions/input_tokens
//	POST /responses/input_tokens
//	POST /v1/messages/count_tokens
//	POST /apply-template
//
// These are stable llama-server routes. The handler forwards verbatim (method,
// path, query, body, safe headers, auth) to the configured llama-server.
//
// Streaming passthrough (PassthroughStream) forwards raw response bytes for
// SSE endpoints (e.g. llama-server's /v1/stream replay API) so callers can
// stream without re-implementing the protocol.

// passthroughAllowedMethods is the method allowlist for the native routes.
var passthroughAllowedMethods = map[string]bool{
	"GET":  true,
	"POST": true,
}

// Passthrough executes a non-streaming passthrough request to llama-server.
func (provider *LLamaCppProvider) Passthrough(ctx *schemas.BifrostContext, key schemas.Key, req *schemas.BifrostPassthroughRequest) (*schemas.BifrostPassthroughResponse, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	if !passthroughAllowedMethods[strings.ToUpper(req.Method)] {
		return nil, providerUtils.NewBifrostOperationError(
			"llamacpp: passthrough method not allowed: "+req.Method, nil)
	}

	fasthttpReq := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(fasthttpReq)
	defer fasthttp.ReleaseResponse(resp)

	fasthttpReq.Header.SetMethod(req.Method)
	fasthttpReq.SetRequestURI(baseURL + req.Path)
	if req.RawQuery != "" {
		fasthttpReq.Header.SetRequestURI(baseURL + req.Path + "?" + req.RawQuery)
	}

	providerUtils.SetExtraHeaders(ctx, fasthttpReq, provider.networkConfig.ExtraHeaders, nil)
	for k, v := range req.SafeHeaders {
		fasthttpReq.Header.Set(k, v)
	}
	if key.Value.GetValue() != "" {
		fasthttpReq.Header.Set("Authorization", "Bearer "+key.Value.GetValue())
	}
	fasthttpReq.SetBody(req.Body)

	latency, bifrostErr, wait := providerUtils.MakeRequestWithContext(ctx, provider.client, fasthttpReq, resp)
	defer wait()
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	headers := providerUtils.ExtractProviderResponseHeaders(resp)
	ctx.SetValue(schemas.BifrostContextKeyProviderResponseHeaders, headers)

	body, err := providerUtils.CheckAndDecodeBody(resp)
	if err != nil {
		return nil, providerUtils.NewBifrostOperationError("llamacpp: failed to decode passthrough response body", err)
	}

	return &schemas.BifrostPassthroughResponse{
		StatusCode: resp.StatusCode(),
		Headers:    headers,
		Body:       body,
		ExtraFields: schemas.BifrostResponseExtraFields{
			Latency:                 latency.Milliseconds(),
			Provider:                provider.GetProviderKey(),
			ProviderResponseHeaders: headers,
			PassthroughPath:         req.Path,
		},
	}, nil
}

// PassthroughStream executes a streaming passthrough, forwarding raw response
// bytes as BifrostStreamChunks for SSE llama-server endpoints.
func (provider *LLamaCppProvider) PassthroughStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, postHookSpanFinalizer func(context.Context), key schemas.Key, req *schemas.BifrostPassthroughRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	if !passthroughAllowedMethods[strings.ToUpper(req.Method)] {
		return nil, providerUtils.NewBifrostOperationError(
			"llamacpp: passthrough stream method not allowed: "+req.Method, nil)
	}

	providerUtils.SetStreamIdleTimeoutIfEmpty(ctx, provider.networkConfig.StreamIdleTimeoutInSeconds)

	fasthttpReq := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	resp.StreamBody = true
	defer fasthttp.ReleaseRequest(fasthttpReq)

	fasthttpReq.Header.SetMethod(req.Method)
	fasthttpReq.SetRequestURI(baseURL + req.Path)
	if req.RawQuery != "" {
		fasthttpReq.Header.SetRequestURI(baseURL + req.Path + "?" + req.RawQuery)
	}

	providerUtils.SetExtraHeaders(ctx, fasthttpReq, provider.networkConfig.ExtraHeaders, nil)
	for k, v := range req.SafeHeaders {
		fasthttpReq.Header.Set(k, v)
	}
	if key.Value.GetValue() != "" {
		fasthttpReq.Header.Set("Authorization", "Bearer "+key.Value.GetValue())
	}
	fasthttpReq.SetBody(req.Body)

	err := providerUtils.DoStreamingRequest(ctx, provider.streamingClient, fasthttpReq, resp)
	if err != nil {
		defer providerUtils.ReleaseStreamingResponse(ctx, resp)
		return nil, providerUtils.NewBifrostUpstreamConnectionError(schemas.ErrProviderDoRequest, err)
	}

	ctx.SetValue(schemas.BifrostContextKeyProviderResponseHeaders, providerUtils.ExtractProviderResponseHeaders(resp))

	if resp.StatusCode() != fasthttp.StatusOK {
		defer providerUtils.ReleaseStreamingResponse(ctx, resp)
		providerUtils.MaterializeStreamErrorBody(ctx, resp)
		return nil, upstreamAPIError("llamacpp", resp)
	}

	return providerUtils.StreamPassthrough(
		ctx,
		postHookRunner,
		postHookSpanFinalizer,
		resp,
		resp.BodyStream(),
		providerUtils.PassthroughStreamParams{
			StatusCode:       resp.StatusCode(),
			Headers:          providerUtils.ExtractProviderResponseHeaders(resp),
			Path:             req.Path,
			RawRequest:       req.Body,
			CancellationBody: req.Body,
			Logger:           provider.logger,
		},
	), nil
}
