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
	ref     frame.Plane
	ox, oy  int
	n       int
	valid   bool
	scratch ConvolveScratch
}

// Init prepares probes around the full-pel origin (ox, oy) of the reference
// plane, reusing the prober's convolve scratch across blocks. Validity covers
// every delta within one pixel plus the eight-tap support; Predict reports
// false outside that.
func (p *LumaSubpelProber) Init(ref frame.Plane, ox, oy, n int) {
	const fo = filterTaps/2 - 1
	p.ref, p.ox, p.oy, p.n = ref, ox, oy, n
	p.valid = n > 0 && n <= maxBlockSize &&
		ref.Stride >= ref.Width && len(ref.Pix) >= ref.Stride*(ref.Height-1)+ref.Width &&
		ox-1-fo >= 0 && oy-1-fo >= 0 &&
		ox+1+n+fo+1 <= ref.Width && oy+1+n+fo+1 <= ref.Height
}

// Predict fills dst (stride n) with the EIGHTTAP prediction for the probe at
// delta (1/8-pel units, within +-8 of the origin) and reports whether the
// fast path covered it; callers fall back to the validated predictor when it
// returns false.
func (p *LumaSubpelProber) Predict(dst []byte, delta Vector) bool {
	if !p.valid || delta.Col < -8 || delta.Col > 8 || delta.Row < -8 || delta.Row > 8 {
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
	switch {
	case subX != 0 && subY != 0:
		xKernel := regularSubpelKernel(p.n, subX)
		yKernel := regularSubpelKernel(p.n, subY)
		convolve2D8WithScratchImpl(dstPlane, p.ref, 0, 0, refX, refY, p.n, p.n, xKernel, yKernel, &p.scratch)
	case subX != 0:
		xKernel := regularSubpelKernel(p.n, subX)
		convolveX8Impl(dstPlane, p.ref, 0, 0, refX, refY, p.n, p.n, xKernel)
	case subY != 0:
		yKernel := regularSubpelKernel(p.n, subY)
		convolveY8Impl(dstPlane, p.ref, 0, 0, refX, refY, p.n, p.n, yKernel)
	default:
		for r := range p.n {
			src := (refY+r)*p.ref.Stride + refX
			copy(dst[r*p.n:r*p.n+p.n], p.ref.Pix[src:src+p.n])
		}
	}
	return true
}

func (p *LumaSubpelProber) Predict8x8(dst []byte, delta Vector) bool {
	const n = 8
	if !p.valid || p.n != n || delta.Col < -8 || delta.Col > 8 || delta.Row < -8 || delta.Row > 8 {
		return false
	}
	posX := int64(p.ox)*16 + int64(delta.Col)*2
	posY := int64(p.oy)*16 + int64(delta.Row)*2
	refX := int(posX >> 4)
	refY := int(posY >> 4)
	subX := int(posX & 15)
	subY := int(posY & 15)
	dstPlane := frame.Plane{Pix: dst, Stride: n, Width: n, Height: n}
	switch {
	case subX != 0 && subY != 0:
		xKernel := subpelFilters8[subX]
		yKernel := subpelFilters8[subY]
		convolve2D8WithScratchImpl(dstPlane, p.ref, 0, 0, refX, refY, n, n, xKernel, yKernel, &p.scratch)
	case subX != 0:
		xKernel := subpelFilters8[subX]
		convolveX8Impl(dstPlane, p.ref, 0, 0, refX, refY, n, n, xKernel)
	case subY != 0:
		yKernel := subpelFilters8[subY]
		convolveY8Impl(dstPlane, p.ref, 0, 0, refX, refY, n, n, yKernel)
	default:
		return p.Predict(dst, delta)
	}
	return true
}

func (p *LumaSubpelProber) Predict16x16(dst []byte, delta Vector) bool {
	const n = 16
	if !p.valid || p.n != n || delta.Col < -8 || delta.Col > 8 || delta.Row < -8 || delta.Row > 8 {
		return false
	}
	posX := int64(p.ox)*16 + int64(delta.Col)*2
	posY := int64(p.oy)*16 + int64(delta.Row)*2
	refX := int(posX >> 4)
	refY := int(posY >> 4)
	subX := int(posX & 15)
	subY := int(posY & 15)
	dstPlane := frame.Plane{Pix: dst, Stride: n, Width: n, Height: n}
	switch {
	case subX != 0 && subY != 0:
		xKernel := subpelFilters8[subX]
		yKernel := subpelFilters8[subY]
		convolve2D8WithScratchImpl(dstPlane, p.ref, 0, 0, refX, refY, n, n, xKernel, yKernel, &p.scratch)
	case subX != 0:
		xKernel := subpelFilters8[subX]
		convolveX8Impl(dstPlane, p.ref, 0, 0, refX, refY, n, n, xKernel)
	case subY != 0:
		yKernel := subpelFilters8[subY]
		convolveY8Impl(dstPlane, p.ref, 0, 0, refX, refY, n, n, yKernel)
	default:
		return p.Predict(dst, delta)
	}
	return true
}

func (p *LumaSubpelProber) Predict32x32(dst []byte, delta Vector) bool {
	const n = 32
	if !p.valid || p.n != n || delta.Col < -8 || delta.Col > 8 || delta.Row < -8 || delta.Row > 8 {
		return false
	}
	posX := int64(p.ox)*16 + int64(delta.Col)*2
	posY := int64(p.oy)*16 + int64(delta.Row)*2
	refX := int(posX >> 4)
	refY := int(posY >> 4)
	subX := int(posX & 15)
	subY := int(posY & 15)
	dstPlane := frame.Plane{Pix: dst, Stride: n, Width: n, Height: n}
	switch {
	case subX != 0 && subY != 0:
		xKernel := subpelFilters8[subX]
		yKernel := subpelFilters8[subY]
		convolve2D8WithScratchImpl(dstPlane, p.ref, 0, 0, refX, refY, n, n, xKernel, yKernel, &p.scratch)
	case subX != 0:
		xKernel := subpelFilters8[subX]
		convolveX8Impl(dstPlane, p.ref, 0, 0, refX, refY, n, n, xKernel)
	case subY != 0:
		yKernel := subpelFilters8[subY]
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
