package llamacpp_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"sync"

	"github.com/valyala/fasthttp"
)

// fakeLLamaServer is a self-contained test double for llama.cpp's llama-server.
// It records every request (method, path, headers, raw body) and serves
// scripted responses so tests can assert exact wire behavior without a real
// llama-server or full Bifrost instance.
type fakeLLamaServer struct {
	*httptest.Server

	mu       sync.Mutex
	requests []recordedRequest

	// pathHandler overrides default responses for a specific path.
	pathHandler map[string]func(req recordedRequest, w http.ResponseWriter)
}

type recordedRequest struct {
	Method  string
	Path    string
	Body    []byte
	Headers map[string]string
}

func newFakeLLamaServer() *fakeLLamaServer {
	f := &fakeLLamaServer{
		pathHandler: make(map[string]func(recordedRequest, http.ResponseWriter)),
	}

	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := make([]byte, 0, 64*1024)
		if r.Body != nil {
			body, _ = io.ReadAll(r.Body)
		}
		headers := make(map[string]string)
		for k, v := range r.Header {
			if len(v) > 0 {
				headers[k] = v[0]
			}
		}

		req := recordedRequest{
			Method:  r.Method,
			Path:    r.URL.Path,
			Body:    body,
			Headers: headers,
		}

		f.mu.Lock()
		f.requests = append(f.requests, req)
		handler, hasHandler := f.pathHandler[r.URL.Path]
		f.mu.Unlock()

		if hasHandler {
			handler(req, w)
			return
		}
		f.defaultResponse(req, w)
	}))
	return f
}

// setHandler overrides the default response for a path.
func (f *fakeLLamaServer) setHandler(path string, handler func(recordedRequest, http.ResponseWriter)) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.pathHandler[path] = handler
}

// lastRequest returns the most recent recorded request.
func (f *fakeLLamaServer) lastRequest() *recordedRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.requests) == 0 {
		return nil
	}
	return &f.requests[len(f.requests)-1]
}

// allRequests returns a copy of all recorded requests.
func (f *fakeLLamaServer) allRequests() []recordedRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]recordedRequest, len(f.requests))
	copy(out, f.requests)
	return out
}

// defaultResponse writes a minimal 200 response for unhandled paths.
func (f *fakeLLamaServer) defaultResponse(req recordedRequest, w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}

// writeJSON is a helper to write a JSON body with status code.
func writeJSON(w http.ResponseWriter, status int, body string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write([]byte(body))
}

// writeSSE is a helper to write SSE stream with proper flushing.
func writeSSE(w http.ResponseWriter, events ...string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSON(w, http.StatusInternalServerError, `{"error":"no flusher"}`)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	for _, event := range events {
		_, _ = w.Write([]byte(event))
		flusher.Flush()
	}
}

var _ = fasthttp.StatusOK
