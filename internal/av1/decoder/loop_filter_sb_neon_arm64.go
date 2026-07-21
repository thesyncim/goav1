//go:build arm64 && !purego

package decoder

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/lfmask"
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
)

const (
	loopFilterMaskLineLumaHorizontal = iota
	loopFilterMaskLineLumaVertical
	loopFilterMaskLineChromaHorizontal
	loopFilterMaskLineChromaVertical
)

// loopFilterMaskLineNEONCtx is the Go ABI adapter for the pinned dav1d ARM64
// whole-mask loop-filter kernels. Field offsets are consumed directly by
// loop_filter_sb_neon_arm64.s; do not reorder.
type loopFilterMaskLineNEONCtx struct {
	dst         *byte
	stride      uintptr
	mask        unsafe.Pointer
	level       *uint8
	levelStride uintptr
	lut         *lfmask.FilterLUT
	kind        uintptr
}

//go:noescape
func loopFilterMaskLineNEONAsm(ctx *loopFilterMaskLineNEONCtx)

func filterMaskLine8Available() bool { return true }

// filterLumaMaskLine8Trusted consumes one dav1d-format luma mask line and
// filters all selected 4x4 positions in four-lane groups. q0Base identifies
// the first output sample, levelIndex identifies the current [Yv,Yh,U,V]
// cell, and component is 0 for column edges or 1 for row edges.
func filterLumaMaskLine8Trusted(dst frame.Plane, q0Base int, mask *[3]uint32, levels [][4]uint8, levelIndex, component, levelStride int, edge loopfilter.Edge, lut *lfmask.FilterLUT) {
	kind := loopFilterMaskLineLumaHorizontal
	if edge == loopfilter.EdgeVertical {
		kind = loopFilterMaskLineLumaVertical
	}
	ctx := loopFilterMaskLineNEONCtx{
		dst:         &dst.Pix[q0Base],
		stride:      uintptr(dst.Stride),
		mask:        unsafe.Pointer(&mask[0]),
		level:       &levels[levelIndex][component],
		levelStride: uintptr(levelStride),
		lut:         lut,
		kind:        uintptr(kind),
	}
	loopFilterMaskLineNEONAsm(&ctx)
}

// filterChromaMaskLine8Trusted is the chroma counterpart. component is 2 for
// U or 3 for V in the shared dav1d-format level grid.
func filterChromaMaskLine8Trusted(dst frame.Plane, q0Base int, mask *[2]uint32, levels [][4]uint8, levelIndex, component, levelStride int, edge loopfilter.Edge, lut *lfmask.FilterLUT) {
	kind := loopFilterMaskLineChromaHorizontal
	if edge == loopfilter.EdgeVertical {
		kind = loopFilterMaskLineChromaVertical
	}
	ctx := loopFilterMaskLineNEONCtx{
		dst:         &dst.Pix[q0Base],
		stride:      uintptr(dst.Stride),
		mask:        unsafe.Pointer(&mask[0]),
		level:       &levels[levelIndex][component],
		levelStride: uintptr(levelStride),
		lut:         lut,
		kind:        uintptr(kind),
	}
	loopFilterMaskLineNEONAsm(&ctx)
}
