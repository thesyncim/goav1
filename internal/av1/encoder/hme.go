package encoder

import "sync"

// hme.go is the encoder's frame-level motion estimation, SVT-AV1's ME
// process (Source/Lib/Codec/motion_estimation.c svt_aom_motion_estimation_b64)
// run once per inter frame before mode decision, on source pictures (open
// loop) like SVT's pa_me:
//
//   - the source pyramid and the per-SB HME search center (svt_hme.go);
//   - the zz SAD (init_zz_sad) and the per-SB full-pel sweep with per-PU
//     argmins (pframe_depth_removal.go realtimeSBMultiSizeME) for EVERY
//     superblock, partial edge ones included (SVT runs the ME on the padded
//     picture; here partial SBs and search positions that leave the frame
//     read edge-replicated windows, the padding SVT's buffers carry);
//   - compute_distortion's per-SB statistics (me_8x8_distortion,
//     me_8x8_cost_variance, dist_64x64...).
//
// Mode decision reads the per-PU vectors (pframe_motion.go), the partitioner
// and depth removal read the statistics, and cyclic refresh reads
// me_8x8_distortion as SVT's cyclic_sb_qp_derivation does.

// hmeState owns the source pyramids, the previous source (the open-loop ME
// reference), the frame's ME results and the band workers that compute them.
type hmeState struct {
	svt svtPyramid

	// prevY is the active reference source's luma at prevStride (the
	// reference of the open-loop sweep and zz SAD): a view of the source
	// kept with the reference slot the frame predicts from (refY), the way
	// SVT's pa_me searches each reference picture's input.
	prevY      []byte
	prevStride int
	prevW      int
	prevH      int
	havePrev   bool
	refY       [hmeRefSlots]hmeRefLuma

	// fme is the current frame's ME result.
	fme frameME

	// win is per-band scratch for partial and edge superblocks.
	win [hmeBands]meWindow

	work     chan hmeJob
	done     chan struct{}
	started  bool
	workerWG sync.WaitGroup
}

// frameME is one inter frame's open-loop ME, per 64x64 superblock in raster
// order.
type frameME struct {
	cols, rows int
	valid      bool
	isRef      bool
	sb         []realtimeSBMultiSizeME
	// l1SAD is the HME center's level-1 SAD (0 when HME was skipped).
	l1SAD []uint32
	// me8x8Dist is compute_distortion's me_8x8_distortion: the per-PU 8x8
	// best SAD sum normalized to a 64x64 area.
	me8x8Dist []uint64
}

// hmeBands is the SB-row fan-out of the frame ME.
const hmeBands = 4

type hmeJob struct {
	band   int
	r0, r1 int
	src    SourceFrame420
}

func (h *hmeState) startWorkers() {
	if h.started {
		return
	}
	work := make(chan hmeJob, hmeBands)
	done := make(chan struct{}, hmeBands)
	h.work, h.done = work, done
	for range hmeBands {
		h.workerWG.Add(1)
		go func() {
			defer h.workerWG.Done()
			for j := range work {
				h.meRows(j.band, j.src, j.r0, j.r1)
				done <- struct{}{}
			}
		}()
	}
	h.started = true
}

func (h *hmeState) close() {
	work := h.work
	if work == nil {
		return
	}
	close(work)
	h.workerWG.Wait()
	h.work = nil
	h.done = nil
	h.started = false
}

// hmeRefSlots is the number of reference sources kept: LAST (slot 0) and
// the L1T3 middle layer (slot 1).
const hmeRefSlots = 2

// hmeRefLuma is one reference slot's source luma.
type hmeRefLuma struct {
	y            []byte
	stride, w, h int
	valid        bool
}

// prime makes src (a keyframe) the reference source of every slot.
func (h *hmeState) prime(src SourceFrame420) {
	h.svt.build(src)
	for s := range h.refY {
		h.storeRef(s, src)
	}
	h.selectRef(0)
	h.fme.valid = false
}

// storeRef keeps src (whose pyramid is the current one) as slot s's
// reference source.
func (h *hmeState) storeRef(s int, src SourceFrame420) {
	r := &h.refY[s]
	n := src.YStride * src.Height
	if len(r.y) < n {
		r.y = make([]byte, n)
	}
	copy(r.y[:n], src.Y[:n])
	r.stride, r.w, r.h, r.valid = src.YStride, src.Width, src.Height, true
	h.svt.storeRef(s)
}

// selectRef makes slot s the active ME reference.
func (h *hmeState) selectRef(s int) {
	r := &h.refY[s]
	h.prevY, h.prevStride, h.prevW, h.prevH, h.havePrev = r.y, r.stride, r.w, r.h, r.valid
	h.svt.selectRef(s)
}

// primeFromRecon seeds the open-loop reference with a reconstruction when no
// previous source exists (the standalone P-frame API).
func (h *hmeState) primeFromRecon(ref SourceFrame420) {
	h.prime(ref)
}

// run is the frame's ME on the band workers; isRef is me_ctx->is_ref,
// ref the reference slot the frame predicts from and store the slot it
// refreshes (-1 for a non-reference picture).
func (h *hmeState) run(src SourceFrame420, isRef bool, ref, store int) {
	h.prepare(src, isRef, ref)
	if h.fme.valid {
		h.startWorkers()
		jobs := 0
		for b := range hmeBands {
			r0 := b * h.fme.rows / hmeBands
			r1 := (b + 1) * h.fme.rows / hmeBands
			if r0 >= r1 {
				continue
			}
			h.work <- hmeJob{band: b, r0: r0, r1: r1, src: src}
			jobs++
		}
		for range jobs {
			<-h.done
		}
	}
	h.finish(src, store)
}

// runSerial is run on the calling goroutine.
func (h *hmeState) runSerial(src SourceFrame420, isRef bool, ref, store int) {
	h.prepare(src, isRef, ref)
	if h.fme.valid {
		h.meRows(0, src, 0, h.fme.rows)
	}
	h.finish(src, store)
}

// finish keeps a reference picture's source in the slot it refreshes.
func (h *hmeState) finish(src SourceFrame420, store int) {
	if store >= 0 && store < hmeRefSlots {
		h.storeRef(store, src)
	}
}

func (h *hmeState) prepare(src SourceFrame420, isRef bool, ref int) {
	h.svt.build(src)
	h.selectRef(min(max(ref, 0), hmeRefSlots-1))
	f := &h.fme
	f.valid = false
	if !h.havePrev || h.prevW != src.Width || h.prevH != src.Height || h.prevStride != src.YStride ||
		!h.svt.ready() || src.Width < 64 || src.Height < 64 {
		return
	}
	f.cols, f.rows = (src.Width+63)/64, (src.Height+63)/64
	n := f.cols * f.rows
	if len(f.sb) < n {
		f.sb = make([]realtimeSBMultiSizeME, n)
		f.l1SAD = make([]uint32, n)
		f.me8x8Dist = make([]uint64, n)
	}
	f.isRef = isRef
	f.valid = true
}

func (h *hmeState) meRows(band int, src SourceFrame420, r0, r1 int) {
	f := &h.fme
	for r := r0; r < r1; r++ {
		for c := 0; c < f.cols; c++ {
			i := r*f.cols + c
			f.l1SAD[i], f.me8x8Dist[i] = h.motionEstimateSB(&f.sb[i], src, c*64, r*64, f.isRef, &h.win[band])
		}
	}
}

// sbME returns the frame ME of the superblock containing luma (px, py).
func (h *hmeState) sbME(px, py int) *realtimeSBMultiSizeME {
	if h == nil || !h.fme.valid {
		return nil
	}
	c, r := px/64, py/64
	if c >= h.fme.cols || r >= h.fme.rows {
		return nil
	}
	return &h.fme.sb[r*h.fme.cols+c]
}

// hmeCutSAD is the per-sample level-1 HME SAD above which a superblock found
// no match; a frame where most superblocks found none is a scene cut.
const hmeCutSAD = 25

// cutDetected reports a scene cut: at least 60% of the superblocks' HME
// centers left a level-1 SAD above hmeCutSAD per half-resolution sample.
// SVT-AV1 at preset 12 rtc runs with scene-change detection off; this keeps
// the encoder's opt-in scene-cut keyframes on the SVT HME data.
func (h *hmeState) cutDetected() bool {
	f := &h.fme
	if !f.valid {
		return false
	}
	n := f.cols * f.rows
	cut := 0
	for _, s := range f.l1SAD[:n] {
		if s > 32*32*hmeCutSAD {
			cut++
		}
	}
	return cut*10 >= n*6
}
