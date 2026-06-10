package encoder

import (
	"sync"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

// hme.go is the hierarchical motion-estimation pre-pass: a quarter-resolution
// coarse search per 32x32 region whose vector seeds the full-resolution
// refinement windows. The full-pel diamond alone covers +-8px, so panning
// faster than that loses motion compensation entirely; the quarter-res
// search covers +-32px at a sixteenth of the SAD cost and the seeded window
// recovers the exact vector.

// hmeState owns the reusable pyramid planes and the per-region seed grid.
// The reference pyramid is the previous frame's source pyramid (recycled by
// swap): seeds only recenter the exact full-resolution search, so the
// source-vs-reconstruction difference is irrelevant and each frame builds
// one quarter plane instead of two.
type hmeState struct {
	srcQ  []byte
	refQ  []byte
	qw    int
	qh    int
	seeds []motion.Vector // full-pel even offsets per 32x32 region
	cols  int
	rows  int
	armed bool

	// bandCut counts regions per search band whose best quarter-res SAD
	// stayed high - no match anywhere in the +-32px reach. A frame where
	// most regions fail is a scene cut.
	bandCut [hmeBands]int
}

// hmeBands is the row-band fan-out of the pyramid build and region search.
const hmeBands = 4

// hmeCutRegionSAD marks a region as unmatched: ~25 per quarter-res sample
// over an 8x8 block, far above textured-content match SADs.
const hmeCutRegionSAD = 8 * 8 * 25

// prime seeds the reference pyramid from a frame that has no predecessor
// (the keyframe restarting the chain).
func (h *hmeState) prime(src SourceFrame420) {
	w, ht := src.Width, src.Height
	qw, qh := w/4, ht/4
	if qw < 8 || qh < 8 {
		h.armed = false
		return
	}
	if len(h.srcQ) < qw*qh {
		h.srcQ = make([]byte, qw*qh)
		h.refQ = make([]byte, qw*qh)
	}
	h.qw, h.qh = qw, qh
	buildQuarterPlane(h.refQ, src.Y, src.YStride, qw, qh)
	h.armed = true
}

// buildQuarterPlane box-averages src 4:1 in both dimensions into dst.
func buildQuarterPlane(dst []byte, src []byte, stride, qw, qh int) {
	for qy := 0; qy < qh; qy++ {
		base := qy * 4 * stride
		drow := qy * qw
		for qx := 0; qx < qw; qx++ {
			s := 0
			for r := 0; r < 4; r++ {
				row := base + r*stride + qx*4
				s += int(src[row]) + int(src[row+1]) + int(src[row+2]) + int(src[row+3])
			}
			dst[drow+qx] = uint8((s + 8) >> 4)
		}
	}
}

// run rebuilds the source pyramid, searches it against the previous frame's
// (the reference pyramid), and fills the seed grid. The quarter-res search
// probes a stride-2 raster over +-8 quarter pels (+-32 full pels) and
// refines +-1, on 8x8 quarter blocks (32x32 regions).
func (h *hmeState) run(src SourceFrame420) {
	w, ht := src.Width, src.Height
	qw, qh := w/4, ht/4
	if qw < 8 || qh < 8 || !h.armed {
		h.cols, h.rows = 0, 0
		return
	}
	cols := (w + 31) / 32
	rows := (ht + 31) / 32
	if len(h.seeds) < cols*rows {
		h.seeds = make([]motion.Vector, cols*rows)
	}
	h.qw, h.qh, h.cols, h.rows = qw, qh, cols, rows
	// Build and search fan out over row bands: both passes write disjoint
	// rows and the search reads only completed pyramid planes.
	var wg sync.WaitGroup
	for b := range hmeBands {
		q0 := b * qh / hmeBands
		q1 := (b + 1) * qh / hmeBands
		if q0 >= q1 {
			continue
		}
		wg.Add(1)
		go func(q0, q1 int) {
			defer wg.Done()
			buildQuarterPlane(h.srcQ[q0*qw:], src.Y[q0*4*src.YStride:], src.YStride, qw, q1-q0)
		}(q0, q1)
	}
	wg.Wait()
	defer func() { h.srcQ, h.refQ = h.refQ, h.srcQ }()

	for b := range hmeBands {
		r0 := b * rows / hmeBands
		r1 := (b + 1) * rows / hmeBands
		if r0 >= r1 {
			continue
		}
		h.bandCut[b] = 0
		wg.Add(1)
		go func(b, r0, r1 int) {
			defer wg.Done()
			h.searchRows(b, r0, r1)
		}(b, r0, r1)
	}
	wg.Wait()
}

// cutDetected reports whether the last run looks like a scene cut: at least
// sixty percent of the regions found no quarter-res match.
func (h *hmeState) cutDetected() bool {
	total := h.cols * h.rows
	if total == 0 {
		return false
	}
	cut := 0
	for _, c := range h.bandCut {
		cut += c
	}
	return cut*10 >= total*6
}

// searchRows fills the seed grid for region rows [r0, r1).
func (h *hmeState) searchRows(band, r0, r1 int) {
	qw, qh, cols := h.qw, h.qh, h.cols
	for ry := r0; ry < r1; ry++ {
		for rx := 0; rx < cols; rx++ {
			// Clamp partial edge regions onto the last whole quarter block.
			qx := min(rx*8, qw-8)
			qy := min(ry*8, qh-8)
			srcBlock := h.srcQ[qy*qw+qx:]
			zero := sad8x8DualImpl(srcBlock, qw, h.refQ[qy*qw+qx:], qw)
			bestDX, bestDY, bestSAD := 0, 0, zero
			// Static fast path mirrors the full-res search's bar.
			if zero > 8*8*2 {
				minDX, maxDX := max(-8, -qx), min(8, qw-8-qx)
				minDY, maxDY := max(-8, -qy), min(8, qh-8-qy)
				for dy := minDY; dy <= maxDY; dy += 2 {
					for dx := minDX; dx <= maxDX; dx += 2 {
						if dx == 0 && dy == 0 {
							continue
						}
						if s := sad8x8DualImpl(srcBlock, qw, h.refQ[(qy+dy)*qw+qx+dx:], qw); s < bestSAD {
							bestSAD, bestDX, bestDY = s, dx, dy
						}
					}
				}
				for _, cand := range [4][2]int{{bestDX + 1, bestDY}, {bestDX - 1, bestDY}, {bestDX, bestDY + 1}, {bestDX, bestDY - 1}} {
					dx, dy := cand[0], cand[1]
					if dx < minDX || dx > maxDX || dy < minDY || dy > maxDY {
						continue
					}
					if s := sad8x8DualImpl(srcBlock, qw, h.refQ[(qy+dy)*qw+qx+dx:], qw); s < bestSAD {
						bestSAD, bestDX, bestDY = s, dx, dy
					}
				}
			}
			// Quarter-pel offsets scale to multiples of four full pels,
			// keeping the even-offset chroma alignment invariant.
			h.seeds[ry*cols+rx] = motion.Vector{Row: int16(bestDY * 4), Col: int16(bestDX * 4)}
			if bestSAD > hmeCutRegionSAD {
				h.bandCut[band]++
			}
		}
	}
}

// seedAt returns the full-pel seed for the 32x32 region containing (px, py).
func (h *hmeState) seedAt(px, py int) (int, int) {
	if h.cols == 0 {
		return 0, 0
	}
	rx := min(px/32, h.cols-1)
	ry := min(py/32, h.rows-1)
	s := h.seeds[ry*h.cols+rx]
	return int(s.Col), int(s.Row)
}
