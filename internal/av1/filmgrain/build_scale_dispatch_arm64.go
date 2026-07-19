// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package filmgrain

import (
	"os"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// buildScaleUseNEON selects the NEON scaling-LUT gather kernel. It is resolved
// once, at package init, before any decoder goroutine starts, and must not be
// mutated concurrently with live decoding. As with applyGrainSegmentUseNEON the
// dispatch is a plain bool rather than a func pointer so buildScaleRow8 /
// buildScaleRow10 below stay concrete (non-indirect) calls: escape analysis can
// then see that neither implementation retains its slice arguments, which keeps
// the caller's scale scratch buffer on the stack and the apply path zero-alloc.
//
// GOAV1_DISABLE_FILMGRAIN_SCALE_ASM forces the pure-Go reference; it is a
// measurement kill-switch that mirrors GOAV1_DISABLE_WARP_ASM.
var buildScaleUseNEON = cpu.Detected.NEON &&
	os.Getenv("GOAV1_DISABLE_FILMGRAIN_SCALE_ASM") == ""

func buildScaleRow10(scale []uint16, src []uint16, lut []uint8) {
	if buildScaleUseNEON {
		buildScaleRow10NEON(scale, src, lut)
		return
	}
	buildScaleRow10PureGo(scale, src, lut)
}
