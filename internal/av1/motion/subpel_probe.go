package motion

import "github.com/thesyncim/goav1/internal/av1/frame"

// subpel_probe.go is the motion-search probe fast path: the encoder's subpel
// refinement predicts the same n x n luma block at several vectors within one
// pixel of a full-pel start, so the per-call geometry validation of the
// general predictor is hoisted to construction time and each probe dispatches
// straight to the convolve kernels. Predictions are bit-identical to
// PredictInterPlaneBlockFromOriginWithFilterBitDepth with EIGHTTAP filters.

// LumaSubpelProber predicts n x n luma probes around one full-pel origin.
type LumaSubpelProber struct {
	ref          frame.Plane
	ox, oy       int
	n            int
	valid        bool
	clampedValid bool
	scratch      ConvolveScratch
}

const (
	// Libaom av1_find_best_sub_pixel_tree starts with hstep=4, then for the
	// realtime low-precision path runs the hstep=2 quarter-pel round. Its
	// first-level plus second_level_check_v2 pattern can probe up to 8 eighth-pel
	// units from the full-pel start in the half-pel round, then another 4 in the
	// quarter-pel round.
	lumaSubpelProbeDeltaLimit  int16 = 12
	lumaSubpelProbeMinRefPixel       = -2
	lumaSubpelProbeMaxRefPixel       = 1
)

// Init prepares probes around the full-pel origin (ox, oy) of the reference
// plane, reusing the prober's convolve scratch across blocks. Validity covers
// every low-precision libaom-tree delta plus the eight-tap support; Predict
// reports false outside that.
func (p *LumaSubpelProber) Init(ref frame.Plane, ox, oy, n int) {
	const fo = filterTaps/2 - 1
	p.ref, p.ox, p.oy, p.n = ref, ox, oy, n
	p.clampedValid = n > 0 && n <= maxBlockSize &&
		planeRegionFits(ref, 1, 0, 0, ref.Width, ref.Height)
	p.valid = p.clampedValid &&
		ox+lumaSubpelProbeMinRefPixel-fo >= 0 && oy+lumaSubpelProbeMinRefPixel-fo >= 0 &&
		ox+lumaSubpelProbeMaxRefPixel+n+fo+1 <= ref.Width && oy+lumaSubpelProbeMaxRefPixel+n+fo+1 <= ref.Height
}

// Predict fills dst (stride n) with the EIGHTTAP prediction for the probe at
// delta (1/8-pel units, within the source-shaped low-precision libaom tree
// range) and reports whether the fast path covered it; callers fall back to
// the validated predictor when it returns false.
func (p *LumaSubpelProber) Predict(dst []byte, delta Vector) bool {
	if !p.clampedValid || !lumaSubpelProbeDeltaInRange(delta) {
		return false
	}
	// referenceOriginQ4 semantics at luma scale: position in Q4 units.
	posX := int64(p.ox)*16 + int64(delta.Col)*2
	posY := int64(p.oy)*16 + int64(delta.Row)*2
	refX := int(posX >> 4)
	refY := int(posY >> 4)
	subX := int(posX & 15)
	subY := int(posY & 15)
	dstPlane := frame.Plane{Pix: dst, Stride: p.n, Width: p.n, Height: p.n}
	clamped := !p.valid
	switch {
	case subX != 0 && subY != 0:
		xKernel := regularSubpelKernel(p.n, subX)
		yKernel := regularSubpelKernel(p.n, subY)
		if clamped {
			convolve2D8ClampedWithScratchImpl(dstPlane, p.ref, 0, 0, refX, refY, p.n, p.n, xKernel, yKernel, &p.scratch)
			break
		}
		convolve2D8WithScratchImpl(dstPlane, p.ref, 0, 0, refX, refY, p.n, p.n, xKernel, yKernel, &p.scratch)
	case subX != 0:
		xKernel := regularSubpelKernel(p.n, subX)
		if clamped {
			convolveX8ClampedImpl(dstPlane, p.ref, 0, 0, refX, refY, p.n, p.n, xKernel)
			break
		}
		convolveX8Impl(dstPlane, p.ref, 0, 0, refX, refY, p.n, p.n, xKernel)
	case subY != 0:
		yKernel := regularSubpelKernel(p.n, subY)
		if clamped {
			convolveY8ClampedImpl(dstPlane, p.ref, 0, 0, refX, refY, p.n, p.n, yKernel)
			break
		}
		convolveY8Impl(dstPlane, p.ref, 0, 0, refX, refY, p.n, p.n, yKernel)
	default:
		if clamped {
			return copyPlaneBlockClamped(dstPlane, p.ref, 1, 0, 0, refX, refY, p.n, p.n) == nil
		}
		for r := range p.n {
			src := (refY+r)*p.ref.Stride + refX
			copy(dst[r*p.n:r*p.n+p.n], p.ref.Pix[src:src+p.n])
		}
	}
	return true
}

func (p *LumaSubpelProber) Predict8x8(dst []byte, delta Vector) bool {
	const n = 8
	if !p.clampedValid || p.n != n || !lumaSubpelProbeDeltaInRange(delta) {
		return false
	}
	posX := int64(p.ox)*16 + int64(delta.Col)*2
	posY := int64(p.oy)*16 + int64(delta.Row)*2
	refX := int(posX >> 4)
	refY := int(posY >> 4)
	subX := int(posX & 15)
	subY := int(posY & 15)
	dstPlane := frame.Plane{Pix: dst, Stride: n, Width: n, Height: n}
	clamped := !p.valid
	switch {
	case subX != 0 && subY != 0:
		xKernel := subpelFilters8[subX]
		yKernel := subpelFilters8[subY]
		if clamped {
			convolve2D8ClampedWithScratchImpl(dstPlane, p.ref, 0, 0, refX, refY, n, n, xKernel, yKernel, &p.scratch)
			break
		}
		convolve2D8WithScratchImpl(dstPlane, p.ref, 0, 0, refX, refY, n, n, xKernel, yKernel, &p.scratch)
	case subX != 0:
		xKernel := subpelFilters8[subX]
		if clamped {
			convolveX8ClampedImpl(dstPlane, p.ref, 0, 0, refX, refY, n, n, xKernel)
			break
		}
		convolveX8Impl(dstPlane, p.ref, 0, 0, refX, refY, n, n, xKernel)
	case subY != 0:
		yKernel := subpelFilters8[subY]
		if clamped {
			convolveY8ClampedImpl(dstPlane, p.ref, 0, 0, refX, refY, n, n, yKernel)
			break
		}
		convolveY8Impl(dstPlane, p.ref, 0, 0, refX, refY, n, n, yKernel)
	default:
		return p.Predict(dst, delta)
	}
	return true
}

func (p *LumaSubpelProber) Predict16x16(dst []byte, delta Vector) bool {
	const n = 16
	if !p.clampedValid || p.n != n || !lumaSubpelProbeDeltaInRange(delta) {
		return false
	}
	posX := int64(p.ox)*16 + int64(delta.Col)*2
	posY := int64(p.oy)*16 + int64(delta.Row)*2
	refX := int(posX >> 4)
	refY := int(posY >> 4)
	subX := int(posX & 15)
	subY := int(posY & 15)
	dstPlane := frame.Plane{Pix: dst, Stride: n, Width: n, Height: n}
	clamped := !p.valid
	switch {
	case subX != 0 && subY != 0:
		xKernel := subpelFilters8[subX]
		yKernel := subpelFilters8[subY]
		if clamped {
			convolve2D8ClampedWithScratchImpl(dstPlane, p.ref, 0, 0, refX, refY, n, n, xKernel, yKernel, &p.scratch)
			break
		}
		convolve2D8WithScratchImpl(dstPlane, p.ref, 0, 0, refX, refY, n, n, xKernel, yKernel, &p.scratch)
	case subX != 0:
		xKernel := subpelFilters8[subX]
		if clamped {
			convolveX8ClampedImpl(dstPlane, p.ref, 0, 0, refX, refY, n, n, xKernel)
			break
		}
		convolveX8Impl(dstPlane, p.ref, 0, 0, refX, refY, n, n, xKernel)
	case subY != 0:
		yKernel := subpelFilters8[subY]
		if clamped {
			convolveY8ClampedImpl(dstPlane, p.ref, 0, 0, refX, refY, n, n, yKernel)
			break
		}
		convolveY8Impl(dstPlane, p.ref, 0, 0, refX, refY, n, n, yKernel)
	default:
		return p.Predict(dst, delta)
	}
	return true
}

func (p *LumaSubpelProber) Predict32x32(dst []byte, delta Vector) bool {
	const n = 32
	if !p.clampedValid || p.n != n || !lumaSubpelProbeDeltaInRange(delta) {
		return false
	}
	posX := int64(p.ox)*16 + int64(delta.Col)*2
	posY := int64(p.oy)*16 + int64(delta.Row)*2
	refX := int(posX >> 4)
	refY := int(posY >> 4)
	subX := int(posX & 15)
	subY := int(posY & 15)
	dstPlane := frame.Plane{Pix: dst, Stride: n, Width: n, Height: n}
	clamped := !p.valid
	switch {
	case subX != 0 && subY != 0:
		xKernel := subpelFilters8[subX]
		yKernel := subpelFilters8[subY]
		if clamped {
			convolve2D8ClampedWithScratchImpl(dstPlane, p.ref, 0, 0, refX, refY, n, n, xKernel, yKernel, &p.scratch)
			break
		}
		convolve2D8WithScratchImpl(dstPlane, p.ref, 0, 0, refX, refY, n, n, xKernel, yKernel, &p.scratch)
	case subX != 0:
		xKernel := subpelFilters8[subX]
		if clamped {
			convolveX8ClampedImpl(dstPlane, p.ref, 0, 0, refX, refY, n, n, xKernel)
			break
		}
		convolveX8Impl(dstPlane, p.ref, 0, 0, refX, refY, n, n, xKernel)
	case subY != 0:
		yKernel := subpelFilters8[subY]
		if clamped {
			convolveY8ClampedImpl(dstPlane, p.ref, 0, 0, refX, refY, n, n, yKernel)
			break
		}
		convolveY8Impl(dstPlane, p.ref, 0, 0, refX, refY, n, n, yKernel)
	default:
		return p.Predict(dst, delta)
	}
	return true
}

func (p *LumaSubpelProber) Predict64x64(dst []byte, delta Vector) bool {
	const n = 64
	if !p.clampedValid || p.n != n || !lumaSubpelProbeDeltaInRange(delta) {
		return false
	}
	posX := int64(p.ox)*16 + int64(delta.Col)*2
	posY := int64(p.oy)*16 + int64(delta.Row)*2
	refX := int(posX >> 4)
	refY := int(posY >> 4)
	subX := int(posX & 15)
	subY := int(posY & 15)
	dstPlane := frame.Plane{Pix: dst, Stride: n, Width: n, Height: n}
	clamped := !p.valid
	switch {
	case subX != 0 && subY != 0:
		xKernel := subpelFilters8[subX]
		yKernel := subpelFilters8[subY]
		if clamped {
			convolve2D8ClampedWithScratchImpl(dstPlane, p.ref, 0, 0, refX, refY, n, n, xKernel, yKernel, &p.scratch)
			break
		}
		convolve2D8WithScratchImpl(dstPlane, p.ref, 0, 0, refX, refY, n, n, xKernel, yKernel, &p.scratch)
	case subX != 0:
		xKernel := subpelFilters8[subX]
		if clamped {
			convolveX8ClampedImpl(dstPlane, p.ref, 0, 0, refX, refY, n, n, xKernel)
			break
		}
		convolveX8Impl(dstPlane, p.ref, 0, 0, refX, refY, n, n, xKernel)
	case subY != 0:
		yKernel := subpelFilters8[subY]
		if clamped {
			convolveY8ClampedImpl(dstPlane, p.ref, 0, 0, refX, refY, n, n, yKernel)
			break
		}
		convolveY8Impl(dstPlane, p.ref, 0, 0, refX, refY, n, n, yKernel)
	default:
		return p.Predict(dst, delta)
	}
	return true
}

func regularSubpelKernel(blockSize int, subpelQ4 int) [filterTaps]int16 {
	if blockSize <= 4 {
		return subpelFilters4[subpelQ4]
	}
	return subpelFilters8[subpelQ4]
}

func lumaSubpelProbeDeltaInRange(delta Vector) bool {
	return delta.Col >= -lumaSubpelProbeDeltaLimit &&
		delta.Col <= lumaSubpelProbeDeltaLimit &&
		delta.Row >= -lumaSubpelProbeDeltaLimit &&
		delta.Row <= lumaSubpelProbeDeltaLimit
}
