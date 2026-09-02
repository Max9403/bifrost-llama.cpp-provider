package llamacpp

import (
	schemas "github.com/maximhq/bifrost/core/schemas"
)

// Compile-time assertion that LLamaCppProvider satisfies the
// schemas.Provider interface. If the upstream core changes the interface
// (added/removed methods, signature changes), this fails the build with a
// clear message instead of a runtime type-assertion failure in core dispatch.
var _ schemas.Provider = (*LLamaCppProvider)(nil)
