// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64

package restoration

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best Wiener passes on amd64. No tuned variant
// ships yet, so the pure-Go reference is used. This file only keeps the
// dispatch wiring symmetric across architectures.
//
// TODO: add SSE4.1 / AVX2 variants (PMADDWD for the 7-tap MAC, PSRAD-based
// rounding, PACKUSDW + PMINUW for the clamp). Each would be gated on the
// matching cpu.Detected flag.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	wienerHorizontalImpl = wienerHorizontal
	wienerVerticalImpl = wienerVertical
}
