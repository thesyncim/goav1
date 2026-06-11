package encoder

import (
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
	srcQ     []byte
	refQ     []byte
	qw       int
	qh       int
	seeds    []motion.Vector // full-pel even offsets per 32x32 region
	seedSADs []int32         // best quarter-res SAD per region
	cols     int
	rows     int
	armed    bool

	// bandCut counts regions per search band whose best quarter-res SAD
	// stayed high - no match anywhere in the +-32px reach. A frame where
	// most regions fail is a scene cut.
	bandCut [hmeBands]int

	// Persistent band workers: spawning goroutines per frame allocates a
	// closure and often a fresh stack each time, so the four workers park
	// on a job channel for the encoder's lifetime instead.
	work    chan hmeJob
	done    chan struct{}
	started bool
	// bandStatic counts regions whose zero-motion match was already below
	// the search bar - the static-content signal for the layer offset.
	bandStatic [hmeBands]int
}

// hmeBands is the row-band fan-out of the pyramid build and region search.
const hmeBands = 4

// hmeJob is one band's worth of work for the persistent workers: a build
// slice of the quarter pyramid or a search slice of the region grid.
type hmeJob struct {
	build     bool
	band      int
	r0, r1    int
	src       []byte
	srcStride int
}

// startWorkers launches the persistent band workers once.
func (h *hmeState) startWorkers() {
	if h.started {
		return
	}
	h.work = make(chan hmeJob, hmeBands)
	h.done = make(chan struct{}, hmeBands)
	for range hmeBands {
		go func() {
			for j := range h.work {
				if j.build {
					buildQuarterPlane(h.srcQ[j.r0*h.qw:], j.src[j.r0*4*j.srcStride:], j.srcStride, h.qw, j.r1-j.r0)
				} else {
					h.searchRows(j.band, j.r0, j.r1)
				}
				h.done <- struct{}{}
			}
		}()
	}
	h.started = true
}

// hmeCutRegionSAD marks a region as unmatched: ~25 per quarter-res sample
// over an 8x8 block, far above textured-content match SADs.
const hmeCutRegionSAD = 8 * 8 * 25

// hmeTrustRegionSAD marks a region's seed as trusted: at or below ~4 per
// quarter-res sample the coarse match is clean, so the true full-pel vector
// sits within the seed's four-pel quantization and a narrowed refinement
// window cannot lose it.
const hmeTrustRegionSAD = 8 * 8 * 4

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
		h.seedSADs = make([]int32, cols*rows)
	}
	h.qw, h.qh, h.cols, h.rows = qw, qh, cols, rows
	h.startWorkers()
	// Build and search fan out over row bands: both passes write disjoint
	// rows and the search reads only completed pyramid planes.
	jobs := 0
	for b := range hmeBands {
		q0 := b * qh / hmeBands
		q1 := (b + 1) * qh / hmeBands
		if q0 >= q1 {
			continue
		}
		h.work <- hmeJob{build: true, r0: q0, r1: q1, src: src.Y, srcStride: src.YStride}
		jobs++
	}
	for range jobs {
		<-h.done
	}
	defer func() { h.srcQ, h.refQ = h.refQ, h.srcQ }()

	jobs = 0
	for b := range hmeBands {
		r0 := b * rows / hmeBands
		r1 := (b + 1) * rows / hmeBands
		if r0 >= r1 {
			continue
		}
		h.bandCut[b] = 0
		h.bandStatic[b] = 0
		h.work <- hmeJob{band: b, r0: r0, r1: r1}
		jobs++
	}
	for range jobs {
		<-h.done
	}
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
			if zero <= 8*8*2 {
				h.bandStatic[band]++
			}
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
			h.seedSADs[ry*cols+rx] = int32(bestSAD)
			if bestSAD > hmeCutRegionSAD {
				h.bandCut[band]++
			}
		}
	}
}

// staticFraction reports the share of regions whose zero-motion match was
// already clean in the last run, in 256ths.
func (h *hmeState) staticFraction() int {
	total := h.cols * h.rows
	if total == 0 {
		return 0
	}
	static := 0
	for _, c := range h.bandStatic {
		static += c
	}
	return static * 256 / total
}

// seedAt returns the full-pel seed for the 32x32 region containing (px, py)
// and whether the quarter-res match was clean enough to trust the seed with
// a narrowed refinement window.
func (h *hmeState) seedAt(px, py int) (int, int, bool) {
	if h.cols == 0 {
		return 0, 0, false
	}
	rx := min(px/32, h.cols-1)
	ry := min(py/32, h.rows-1)
	s := h.seeds[ry*h.cols+rx]
	return int(s.Col), int(s.Row), h.seedSADs[ry*h.cols+rx] <= hmeTrustRegionSAD
}
