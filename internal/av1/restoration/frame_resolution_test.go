// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package restoration

import (
	"errors"
	"testing"
)

// TestRestorationScratchScalesWithUnitDimensions confirms that the scratch
// length helpers grow with the per-unit width/height. This is the contract
// frame-level callers (including SVC enhancement layers running at their own
// resolution) rely on when partitioning bordered uint16 buffers into
// processing-unit slices.
func TestRestorationScratchScalesWithUnitDimensions(t *testing.T) {
	for _, tc := range []struct {
		name   string
		width  int
		height int
	}{
		{"min", 1, 1},
		{"tail-h", 16, 64},
		{"tail-w", 64, 16},
		{"max-unit", ProcUnitSize, ProcUnitSize},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gotW, err := WienerScratchLen(tc.width, tc.height)
			if err != nil {
				t.Fatalf("WienerScratchLen: %v", err)
			}
			wantW := tc.width * (tc.height + 2*WienerHalfwin)
			if gotW != wantW {
				t.Fatalf("WienerScratchLen=%d want %d", gotW, wantW)
			}
			gotS, err := SelfguidedScratchLen(tc.width, tc.height)
			if err != nil {
				t.Fatalf("SelfguidedScratchLen: %v", err)
			}
			dgdStride := tc.width + 2*SGRProjBorderHorz
			dgdLen := dgdStride * (tc.height + 2*SGRProjBorderVert)
			fltLen := tc.width * tc.height
			bufStride := sgrBufferStride(tc.width)
			bufLen := bufStride * (tc.height + 2*SGRProjBorderVert)
			wantS := dgdLen + 2*fltLen + 2*bufLen
			if gotS != wantS {
				t.Fatalf("SelfguidedScratchLen=%d want %d", gotS, wantS)
			}
		})
	}
	// Per-unit invariant: anything larger than the AV1 processing unit must be
	// rejected by both entry points. The frame-level scheduler must therefore
	// tile each plane into <=64x64 units before invoking restoration, no matter
	// how large the enhancement-layer frame is.
	if _, err := WienerScratchLen(ProcUnitSize+1, 1); !errors.Is(err, ErrInvalidRestoration) {
		t.Fatalf("WienerScratchLen oversized=%v want %v", err, ErrInvalidRestoration)
	}
	if _, err := SelfguidedScratchLen(1, ProcUnitSize+1); !errors.Is(err, ErrInvalidRestoration) {
		t.Fatalf("SelfguidedScratchLen oversized=%v want %v", err, ErrInvalidRestoration)
	}
}

// TestApplyRestorationCovers1280x720Frame walks Wiener and self-guided
// restoration over every processing unit that tiles a 1280x720 luma plane,
// which is a representative SVC enhancement-layer resolution. The 1280x720
// plane is decomposed into 64x64 processing units plus a 16-pixel-tall tail
// row (720 = 11*64 + 16). For each tile we feed a bordered slice of the
// caller-owned frame buffer to ApplyWienerRestoration / ApplySelfguidedRestoration
// and confirm that:
//   - Every unit invocation succeeds for both filters.
//   - Output samples remain inside the bit-depth range.
//   - A single shared scratch buffer sized for the worst-case unit can be
//     reused across every tile (i.e. scratch reuse is unit-bounded, not
//     frame-bounded), which is the invariant the frame-level pipeline relies
//     on when running at enhancement-layer resolution.
func TestApplyRestorationCovers1280x720Frame(t *testing.T) {
	const (
		frameWidth  = 1280
		frameHeight = 720
		bitDepth    = uint8(8)
	)
	max := uint16((1 << bitDepth) - 1)

	// Allocate one logical "frame" that has enough border for both filters.
	// SGR needs the larger border of the two.
	border := SGRProjBorderHorz
	if WienerHalfwin > border {
		border = WienerHalfwin
	}
	stride := frameWidth + 2*border
	bordered := stride * (frameHeight + 2*border)
	src := make([]uint16, bordered)
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0xc720)
	for i := range src {
		src[i] = uint16(rnd.pseudoUniform(int(max) + 1))
	}

	frameOrigin := border*stride + border

	// Worst-case scratch is the maximum unit size (ProcUnitSize x ProcUnitSize).
	wienerScratch, err := WienerScratchLen(ProcUnitSize, ProcUnitSize)
	if err != nil {
		t.Fatal(err)
	}
	sgrScratch, err := SelfguidedScratchLen(ProcUnitSize, ProcUnitSize)
	if err != nil {
		t.Fatal(err)
	}
	wienerBuf := make([]uint16, wienerScratch)
	sgrBuf := make([]int32, sgrScratch)
	wienerDst := make([]uint16, ProcUnitSize*ProcUnitSize)
	sgrDst := make([]uint16, ProcUnitSize*ProcUnitSize)

	wienerInfo := DefaultWienerInfo()

	tiles := 0
	for top := 0; top < frameHeight; top += ProcUnitSize {
		h := ProcUnitSize
		if top+h > frameHeight {
			h = frameHeight - top
		}
		for left := 0; left < frameWidth; left += ProcUnitSize {
			w := ProcUnitSize
			if left+w > frameWidth {
				w = frameWidth - left
			}
			tileOrigin := frameOrigin + top*stride + left
			dstW := w
			dstH := h

			// Wiener path.
			needW, err := WienerScratchLen(w, h)
			if err != nil {
				t.Fatalf("WienerScratchLen tile=%dx%d: %v", w, h, err)
			}
			if needW > wienerScratch {
				t.Fatalf("wiener scratch grew with tile: need=%d cap=%d", needW, wienerScratch)
			}
			if err := ApplyWienerRestoration(src, stride, tileOrigin, wienerDst, dstW, w, h, wienerInfo, bitDepth, wienerBuf[:needW]); err != nil {
				t.Fatalf("ApplyWienerRestoration tile=(%d,%d) %dx%d: %v", left, top, w, h, err)
			}
			for i := 0; i < dstW*dstH; i++ {
				if wienerDst[i] > max {
					t.Fatalf("wiener tile=(%d,%d) dst[%d]=%d > max=%d", left, top, i, wienerDst[i], max)
				}
			}

			// Self-guided path.
			needS, err := SelfguidedScratchLen(w, h)
			if err != nil {
				t.Fatalf("SelfguidedScratchLen tile=%dx%d: %v", w, h, err)
			}
			if needS > sgrScratch {
				t.Fatalf("sgr scratch grew with tile: need=%d cap=%d", needS, sgrScratch)
			}
			if err := ApplySelfguidedRestoration(src, stride, tileOrigin, sgrDst, dstW, w, h, 0, [2]int{20, -7}, bitDepth, sgrBuf[:needS]); err != nil {
				t.Fatalf("ApplySelfguidedRestoration tile=(%d,%d) %dx%d: %v", left, top, w, h, err)
			}
			for i := 0; i < dstW*dstH; i++ {
				if sgrDst[i] > max {
					t.Fatalf("sgr tile=(%d,%d) dst[%d]=%d > max=%d", left, top, i, sgrDst[i], max)
				}
			}
			tiles++
		}
	}
	// 1280 = 20*64 (no width tail). 720 = 11*64 + 16, so 12 rows of tiles.
	if want := 20 * 12; tiles != want {
		t.Fatalf("tile count=%d want %d", tiles, want)
	}
}
