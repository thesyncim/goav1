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

func (h *hmeState) close() {
	if h.work != nil {
		close(h.work)
		h.work = nil
	}
	h.done = nil
	h.started = false
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
	buildQuarterPlaneArch(dst, src, stride, qw, qh)
}

func buildQuarterPlanePureGo(dst []byte, src []byte, stride, qw, qh int) {
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
// runs the same exhaustive mesh primitive as the full-pel search over 8x8
// quarter blocks (32x32 regions).
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
	h.srcQ, h.refQ = h.refQ, h.srcQ
}

func (h *hmeState) runSerial(src SourceFrame420) {
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
	buildQuarterPlane(h.srcQ, src.Y, src.YStride, qw, qh)
	for b := range hmeBands {
		h.bandCut[b] = 0
		h.bandStatic[b] = 0
		r0 := b * rows / hmeBands
		r1 := (b + 1) * rows / hmeBands
		if r0 < r1 {
			h.searchRows(b, r0, r1)
		}
	}
	h.srcQ, h.refQ = h.refQ, h.srcQ
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
			zero := sad8x8Dual(srcBlock, qw, h.refQ[qy*qw+qx:], qw)
			bestDX, bestDY, bestSAD := hmeQuarterMeshSearch(srcBlock, h.refQ, qw, qw, qh, qx, qy)
			if zero <= 8*8*2 {
				h.bandStatic[band]++
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

func hmeQuarterMeshSearch(srcBlock, ref []byte, stride, width, height, qx, qy int) (int, int, int) {
	bestDX, bestDY, _ := hmeQuarterExhaustiveMeshSearch(srcBlock, ref, stride, width, height, qx, qy, 0, 0, 8, 2)
	return hmeQuarterExhaustiveMeshSearch(srcBlock, ref, stride, width, height, qx, qy, bestDX, bestDY, 1, 1)
}

// Ported from libaom av1/encoder/mcomp.c exhaustive_mesh_search. HME supplies
// the quarter-resolution mesh pattern used for realtime seed generation.
func hmeQuarterExhaustiveMeshSearch(srcBlock, ref []byte, stride, width, height, qx, qy int, startDX, startDY, searchRange, step int) (int, int, int) {
	if step < 1 {
		step = 1
	}
	startDX = min(max(startDX, -qx), width-8-qx)
	startDY = min(max(startDY, -qy), height-8-qy)
	minDX, maxDX := -qx, width-8-qx
	minDY, maxDY := -qy, height-8-qy

	startCol := max(-searchRange, minDX-startDX)
	endCol := min(searchRange, maxDX-startDX)
	startRow := max(-searchRange, minDY-startDY)
	endRow := min(searchRange, maxDY-startDY)

	bestDX, bestDY := startDX, startDY
	bestRawSAD := sad8x8Dual(srcBlock, stride, ref[(qy+startDY)*stride+qx+startDX:], stride)
	bestCost := bestRawSAD + hmeQuarterMVCost(startDX, startDY, width, height)
	colStep := step
	if step <= 1 {
		colStep = 4
	}
	for row := startRow; row <= endRow; row += step {
		col := startCol
		if step > 1 {
			for ; col+3*step <= endCol; col += step * 4 {
				dy := startDY + row
				dx0 := startDX + col
				dx1 := dx0 + step
				dx2 := dx1 + step
				dx3 := dx2 + step
				ref0 := ref[(qy+dy)*stride+qx+dx0:]
				ref1 := ref[(qy+dy)*stride+qx+dx1:]
				ref2 := ref[(qy+dy)*stride+qx+dx2:]
				ref3 := ref[(qy+dy)*stride+qx+dx3:]
				s0, s1, s2, s3 := sad8x8x4(srcBlock, ref0, ref1, ref2, ref3, stride)
				hmeQuarterUpdateBest(s0, dx0, dy, width, height, &bestCost, &bestRawSAD, &bestDX, &bestDY)
				hmeQuarterUpdateBest(s1, dx1, dy, width, height, &bestCost, &bestRawSAD, &bestDX, &bestDY)
				hmeQuarterUpdateBest(s2, dx2, dy, width, height, &bestCost, &bestRawSAD, &bestDX, &bestDY)
				hmeQuarterUpdateBest(s3, dx3, dy, width, height, &bestCost, &bestRawSAD, &bestDX, &bestDY)
			}
			for ; col <= endCol; col += step {
				dx, dy := startDX+col, startDY+row
				s := sad8x8Dual(srcBlock, stride, ref[(qy+dy)*stride+qx+dx:], stride)
				hmeQuarterUpdateBest(s, dx, dy, width, height, &bestCost, &bestRawSAD, &bestDX, &bestDY)
			}
			continue
		}
		for ; col+3 <= endCol; col += colStep {
			dy := startDY + row
			dx0 := startDX + col
			dx1 := dx0 + 1
			dx2 := dx1 + 1
			dx3 := dx2 + 1
			ref0 := ref[(qy+dy)*stride+qx+dx0:]
			ref1 := ref[(qy+dy)*stride+qx+dx1:]
			ref2 := ref[(qy+dy)*stride+qx+dx2:]
			ref3 := ref[(qy+dy)*stride+qx+dx3:]
			s0, s1, s2, s3 := sad8x8x4(srcBlock, ref0, ref1, ref2, ref3, stride)
			hmeQuarterUpdateBest(s0, dx0, dy, width, height, &bestCost, &bestRawSAD, &bestDX, &bestDY)
			hmeQuarterUpdateBest(s1, dx1, dy, width, height, &bestCost, &bestRawSAD, &bestDX, &bestDY)
			hmeQuarterUpdateBest(s2, dx2, dy, width, height, &bestCost, &bestRawSAD, &bestDX, &bestDY)
			hmeQuarterUpdateBest(s3, dx3, dy, width, height, &bestCost, &bestRawSAD, &bestDX, &bestDY)
		}
		for i := 0; i < endCol-col; i++ {
			dx, dy := startDX+col+i, startDY+row
			s := sad8x8Dual(srcBlock, stride, ref[(qy+dy)*stride+qx+dx:], stride)
			hmeQuarterUpdateBest(s, dx, dy, width, height, &bestCost, &bestRawSAD, &bestDX, &bestDY)
		}
	}
	return bestDX, bestDY, bestRawSAD
}

func hmeQuarterUpdateBest(rawSAD, dx, dy, width, height int, bestCost, bestRawSAD, bestDX, bestDY *int) {
	if rawSAD >= *bestCost {
		return
	}
	cost := rawSAD + hmeQuarterMVCost(dx, dy, width, height)
	if cost < *bestCost {
		*bestCost, *bestRawSAD, *bestDX, *bestDY = cost, rawSAD, dx, dy
	}
}

func hmeQuarterMVCost(dx, dy, width, height int) int {
	minFrame := min(width, height) * 4
	lambda := 32
	if minFrame >= 720 {
		lambda = 8
	} else if minFrame >= 480 {
		lambda = 15
	}
	// Libaom's MV_COST_L1_* mvsad cost is
	// (lambda * GET_MV_SUBPEL(abs(row)+abs(col))) >> 3; in the quarter plane
	// the searched offsets are full-pel already, so *8 and >>3 cancel.
	return lambda * (abs(dx) + abs(dy))
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
