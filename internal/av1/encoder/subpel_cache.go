package encoder

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/motion"
)

// subpel_cache.go evaluates the libaom subpel tree's candidates for one
// refinement with the fast luma predictor (pred8.go), producing exactly the
// SADs the motion-package prober path produces: full-pel and single-axis
// candidates are fused predict+SAD passes, and 2-D candidates share one
// first-pass intermediate per horizontal phase. The intermediate of an 8-tap
// horizontal pass depends only on the phase and the absolute sample column,
// so one buffer per phase covering every column and row the tree can reach
// (deltas within +-12 eighth-pels: integer offsets -2..1) serves every 2-D
// candidate with that phase; only its vertical pass is per candidate.

const (
	subpelCacheIMStride = 80 // int16 per row: covers n+8 columns for n <= 64
	subpelCacheIMRows   = 64 + 10
	subpelCacheMaxDelta = 12
)

type subpelProbeCache struct {
	ok        bool
	ref       frame.Plane
	src       []byte // the block's top-left source sample onwards
	srcStride int
	ox, oy, n int
	imValid   [3]bool // horizontal Q4 phases 4, 8, 12
	im        [3][subpelCacheIMRows * subpelCacheIMStride]int16
	scratch   [pred8IMLen]int16
}

func (st *lossyEncodeState) ensureSubpelProbeCache() {
	if st.subpelCache == nil && pred8Impl != nil && pred8IMImpl != nil && pred8VIMImpl != nil && pred8KernelsOK {
		st.subpelCache = new(subpelProbeCache)
	}
}

// init prepares the cache for refinements of the n x n block at (px, py)
// starting from the full-pel vector startMV. ok stays false (callers use the
// prober) unless the vector kernels exist, n is 16, 32 or 64, startMV is
// full-pel and every reachable footprint lies inside the plane.
func (c *subpelProbeCache) init(ref frame.Plane, src []byte, srcStride, px, py, n int, startMV motion.Vector) {
	c.ok = false
	if pred8Impl == nil || pred8IMImpl == nil || !pred8KernelsOK || n&15 != 0 || n > 64 ||
		startMV.Row&7 != 0 || startMV.Col&7 != 0 {
		return
	}
	ox, oy := px+int(startMV.Col>>3), py+int(startMV.Row>>3)
	// Rows oy-5 .. oy+n+4 feed the intermediates; columns ox-5 .. ox+n+17
	// cover the n+8 intermediate columns from ox-2 with the 16-byte loads.
	if ox < 5 || oy < 5 || ox+n+18 > ref.Width || oy+n+5 > ref.Height ||
		(oy+n+5)*ref.Stride+ox+n+18 > len(ref.Pix) || len(src) < py*srcStride+px+(n-1)*srcStride+n+16 {
		return
	}
	c.ok = true
	c.ref = ref
	c.src = src[py*srcStride+px:]
	c.srcStride = srcStride
	c.ox, c.oy, c.n = ox, oy, n
	c.imValid = [3]bool{}
}

// sad returns the SAD of the regular-filter prediction at delta (eighth-pel
// units from the start vector) against the source block.
func (c *subpelProbeCache) sad(delta motion.Vector) (int, bool) {
	if !c.ok || delta.Row < -subpelCacheMaxDelta || delta.Row > subpelCacheMaxDelta ||
		delta.Col < -subpelCacheMaxDelta || delta.Col > subpelCacheMaxDelta {
		return 0, false
	}
	posX, posY := int(delta.Col)*2, int(delta.Row)*2
	ix, fx := posX>>4, posX&15
	iy, fy := posY>>4, posY&15
	refX, refY := c.ox+ix, c.oy+iy
	n := c.n
	if fx == 0 || fy == 0 {
		return pred8SAD(c.src, c.srcStride, &c.ref, refX, refY, n, n, fx, fy, &c.scratch)
	}
	slot := fx>>2 - 1
	im := unsafe.Pointer(&c.im[slot][0])
	k := &pred8Kernels[motion.InterpEightTapRegular]
	if !c.imValid[slot] {
		// Rows oy-5 .. oy+n+4, columns ox-2 .. ox+n+5.
		pred8IMImpl(im, subpelCacheIMStride*2,
			unsafe.Pointer(&c.ref.Pix[(c.oy-5)*c.ref.Stride+c.ox-2]), c.ref.Stride, n+8, n+10, &k[fx])
		c.imValid[slot] = true
	}
	// The candidate's first intermediate row is refY-3 = oy+iy-3, i.e. row
	// iy+2 of the buffer; its first column refX is column ix+2.
	at := unsafe.Add(im, ((iy+2)*subpelCacheIMStride+ix+2)*2)
	return pred8VIMImpl(nil, 0, unsafe.Pointer(&c.src[0]), c.srcStride, at, subpelCacheIMStride*2, n, n, &k[fy]), true
}
