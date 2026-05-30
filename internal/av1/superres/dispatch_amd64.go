// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64

package superres

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best super-res row kernel on amd64.
//
// TODO: add an SSE4.1 / AVX2 polyphase variant (per-output-pixel 8-tap dot
// product with PMADDWD and a rounding shift). Until then the pure-Go reference
// is used. Any future variant must stay bit-exact with upscaleRowPureGo and be
// gated on the matching cpu.Detected flag.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	upscaleRowImpl = upscaleRowPureGo
}
