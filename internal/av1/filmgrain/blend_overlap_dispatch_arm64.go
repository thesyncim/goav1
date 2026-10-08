// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package filmgrain

import (
	"os"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// blendGrainRowUseNEON selects the NEON overlap-blend kernel. It is resolved
// once, at package init, before any decoder goroutine starts, and must not be
// mutated concurrently with live decoding. As with applyGrainSegmentUseNEON and
// buildScaleUseNEON the dispatch is a plain bool rather than a func pointer so
// blendGrainRow below stays a concrete (non-indirect) call: escape analysis can
// then see that neither implementation retains its slice arguments, which keeps
// the caller's blend scratch buffer on the stack and the apply path zero-alloc.
//
// GOAV1_DISABLE_FILMGRAIN_BLEND_ASM forces the pure-Go reference; it is a
// measurement kill-switch that mirrors GOAV1_DISABLE_FILMGRAIN_SCALE_ASM.
var blendGrainRowUseNEON = cpu.Detected.NEON &&
	os.Getenv("GOAV1_DISABLE_FILMGRAIN_BLEND_ASM") == ""

func blendGrainRow(dst []int16, prev []int16, cur []int16, prevWeight int, curWeight int, grainMin int, grainMax int) {
	if blendGrainRowUseNEON {
		blendGrainRowNEON(dst, prev, cur, prevWeight, curWeight, grainMin, grainMax)
		return
	}
	blendGrainRowPureGo(dst, prev, cur, prevWeight, curWeight, grainMin, grainMax)
}
