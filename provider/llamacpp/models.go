package llamacpp

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"time"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	schemas "github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// listModelsByKey fetches the model list for one key and enriches each entry
// with live context-window metadata (meta.n_ctx / meta.n_ctx_train) from
// llama-server's GET /v1/models response — never from model-name heuristics.
func (provider *LLamaCppProvider) listModelsByKey(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostListModelsRequest) (*schemas.BifrostListModelsResponse, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)

	req.Header.SetMethod(http.MethodGet)
	req.SetRequestURI(baseURL + providerUtils.GetPathFromContext(ctx, "/v1/models"))
	providerUtils.SetExtraHeaders(ctx, req, provider.networkConfig.ExtraHeaders, nil)
	for k, v := range provider.authHeader(key) {
		req.Header.Set(k, v)
	}

	startTime := time.Now()
	_, bifrostErr, wait := providerUtils.MakeRequestWithContext(ctx, provider.client, req, resp)
	defer wait()
	latency := time.Since(startTime)
	if bifrostErr != nil {
		return nil, providerUtils.EnrichError(ctx, bifrostErr, nil, nil,
			providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
			providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
			latency)
	}
	if resp.StatusCode() != fasthttp.StatusOK {
		return nil, upstreamAPIError("llamacpp", resp)
	}

	// llama-server /v1/models response:
	// { "data": [ { "id", "aliases", "tags", "object":"model", "created",
	//   "owned_by":"llamacpp", "meta": { "vocab_type","n_vocab","n_ctx",
	//   "n_ctx_train","n_embd","n_params","size","ftype" } } ],
	//   "models": [ ... ] }
	var modelsResp struct {
		Data []struct {
			ID      string   `json:"id"`
			Aliases []string `json:"aliases"`
			Tags    []string `json:"tags"`
			Created int64    `json:"created"`
			OwnedBy string   `json:"owned_by"`
			Context *int     `json:"n_ctx"`
			Meta    *struct {
				VocabType string `json:"vocab_type"`
				NVocab    int64  `json:"n_vocab"`
				NCtx      int    `json:"n_ctx"`
				NCtxTrain int    `json:"n_ctx_train"`
				NEmbd     int    `json:"n_embd"`
				NParams   int64  `json:"n_params"`
				Size      int64  `json:"size"`
				FType     string `json:"ftype"`
			} `json:"meta"`
		} `json:"data"`
	}
	if err := schemas.Unmarshal(resp.Body(), &modelsResp); err != nil {
		return nil, providerUtils.NewBifrostOperationError(schemas.ErrProviderResponseUnmarshal, err)
	}

	data := make([]schemas.Model, 0, len(modelsResp.Data))
	for _, m := range modelsResp.Data {
		model := schemas.Model{
			ID:               m.ID,
			OwnedBy:          schemas.Ptr(m.OwnedBy),
			SupportedMethods: []string{"chat_completion", "text_completion", "count_tokens"},
			ProviderExtra:    nil,
		}
		if m.Created != 0 {
			created := m.Created
			model.Created = &created
		}
		if len(m.Aliases) > 0 {
			model.Alias = &m.Aliases[0]
		}

		// Live context-window metadata. Effective per-slot n_ctx comes from
		// meta.n_ctx (or the top-level n_ctx on older builds). Trained window
		// is n_ctx_train. Both are server-provided, not guessed.
		var effectiveCtx, trainedCtx *int
		if m.Meta != nil {
			if m.Meta.NCtx > 0 {
				v := m.Meta.NCtx
				effectiveCtx = &v
			}
			if m.Meta.NCtxTrain > 0 {
				v := m.Meta.NCtxTrain
				trainedCtx = &v
			}
		} else if m.Context != nil && *m.Context > 0 {
			v := *m.Context
			effectiveCtx = &v
		}
		if effectiveCtx != nil {
			model.ContextLength = effectiveCtx
			// MaxInputTokens is derived from the live server value. We use the
			// effective n_ctx (min(ctx, n_ctx_train)) as the practical cap.
			model.MaxInputTokens = effectiveCtx
		}
		_ = trainedCtx

		data = append(data, model)
	}

	return &schemas.BifrostListModelsResponse{
		Data: data,
		ExtraFields: schemas.BifrostResponseExtraFields{
			Provider: provider.GetProviderKey(),
			Latency:  latency.Milliseconds(),
		},
	}, nil
}

// ListModels performs a list models request, aggregating across keys.
func (provider *LLamaCppProvider) ListModels(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostListModelsRequest) (*schemas.BifrostListModelsResponse, *schemas.BifrostError) {
	return providerUtils.HandleMultipleListModelsRequests(
		ctx,
		keys,
		request,
		provider.listModelsByKey,
	)
}

// discoverMetadata fetches live server/model metadata from /props and caches it.
// This backs ModelInfoProvider-style lookups and any capability checks that
// depend on the server's advertised context window or features.
func (provider *LLamaCppProvider) discoverMetadata(ctx context.Context, baseURL string, key schemas.Key) (*llamaServerMetadata, *schemas.BifrostError) {
	provider.mu.RLock()
	if cached, ok := provider.metadataCache[baseURL]; ok {
		if expiry, ok := provider.metadataExpiry[baseURL]; ok && time.Now().Before(expiry) {
			provider.mu.RUnlock()
			return &cached, nil
		}
	}
	provider.mu.RUnlock()

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)

	req.Header.SetMethod(http.MethodGet)
	req.SetRequestURI(strings.TrimRight(baseURL, "/") + "/props")
	for k, v := range provider.authHeader(key) {
		req.Header.Set(k, v)
	}

	_, bifrostErr, wait := providerUtils.MakeRequestWithContext(ctx, provider.client, req, resp)
	defer wait()
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	if resp.StatusCode() != fasthttp.StatusOK {
		return nil, upstreamAPIError("llamacpp", resp)
	}

	var props struct {
		DefaultGenerationSettings struct {
			NCtx int `json:"n_ctx"`
		} `json:"default_generation_settings"`
		TotalSlots      int    `json:"total_slots"`
		ModelPath       string `json:"model_path"`
		BuildInfo       string `json:"build_info"`
		IsSleeping      bool   `json:"is_sleeping"`
		EndpointSlots   bool   `json:"endpoint_slots"`
		EndpointMetrics bool   `json:"endpoint_metrics"`
		Modalities      struct {
			Vision bool `json:"vision"`
			Video  bool `json:"video"`
			Audio  bool `json:"audio"`
		} `json:"modalities"`
	}
	if err := schemas.Unmarshal(resp.Body(), &props); err != nil {
		return nil, providerUtils.NewBifrostOperationError(schemas.ErrProviderResponseUnmarshal, err)
	}

	meta := &llamaServerMetadata{
		BuildInfo:    props.BuildInfo,
		HasSlots:     props.EndpointSlots,
		HasMetrics:   props.EndpointMetrics,
		Vision:       props.Modalities.Vision,
		Audio:        props.Modalities.Audio,
		Video:        props.Modalities.Video,
		DiscoveredAt: time.Now(),
	}
	if props.DefaultGenerationSettings.NCtx > 0 {
		nctx := props.DefaultGenerationSettings.NCtx
		meta.ContextLength = &nctx
	}

	provider.mu.Lock()
	provider.metadataCache[baseURL] = *meta
	provider.metadataExpiry[baseURL] = time.Now().Add(provider.metadataTTL)
	provider.mu.Unlock()

	return meta, nil
}

// ModelInfo returns the live context metadata for a model, for callers that
// need context-window sizing without issuing a list models request.
// This is a provider-level helper; when the upstream core patch lands a
// ModelCatalogContributor hook, this is the method it should call.
func (provider *LLamaCppProvider) ModelInfo(ctx context.Context, key schemas.Key) (*llamaServerMetadata, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	return provider.discoverMetadata(ctx, baseURL, key)
}

var _ = errors.New
