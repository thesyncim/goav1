package threading

import (
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/thesyncim/goav1/internal/av1/dsp"
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/reconstruct"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// frameWorkWavefrontMinSBRowsPerWorker is the SB-row count, per available
// worker, a tile must clear before the deferred SB-row wavefront engages.
// Below it the fused single-thread path wins (no buffering, no goroutines), so
// the wavefront stays off. It is 2: the diagonal must fill before it drains.
const frameWorkWavefrontMinSBRowsPerWorker = 2

const (
	frameWorkReconMaxIndex        = int32(1<<31 - 1)
	frameWorkReconMaxWavefrontCol = frameWorkReconMaxIndex - 2
)

func frameWorkReconIndex(v int) (int32, bool) {
	if v < 0 || v > int(frameWorkReconMaxIndex) {
		return 0, false
	}
	return int32(v), true
}

func frameWorkReconWavefrontCol(v int) (int32, bool) {
	if v < 0 || v > int(frameWorkReconMaxWavefrontCol) {
		return 0, false
	}
	return int32(v), true
}

const (
	frameWorkReconPaletteY uint8 = 1 << iota
	frameWorkReconPaletteUV
	frameWorkReconMotionExtras
	frameWorkReconSubChroma
)

// frameWorkReconPaletteBinding records which buffered reconstruction event
// should be rebound to which value-arena palette maps once event collection is
// complete. Keeping the binding out of frameWorkReconEvent avoids growing every
// non-palette event for a rare screen-content path.
type frameWorkReconPaletteBinding struct {
	visitIndex     int32
	paletteIndex   int32
	extrasIndex    int32
	subChromaIndex int32
	mask           uint8
}

// frameWorkDeferredSlots are the value-arena slots bufferDeferredVisit copied
// a visit's out-of-line payloads into; mask says which are set.
type frameWorkDeferredSlots struct {
	palette, extras, subChroma int
	mask                       uint8
}

// FrameWorkTileResidualCDFs groups the caller-owned entropy states needed to
// walk block syntax, decode transform trees, and decode residual coefficients.
type FrameWorkTileResidualCDFs struct {
	Loop          tile.BlockLoopCDFs
	Coeff         tile.BlockCoeffCDFs
	TransformType *tile.TransformTypeCDFs
	Restoration   tile.RestorationCDFs
}

// FrameWorkTileResidualCDFStorage owns the default entropy states needed for a
// full block-loop, transform, and coefficient pass. Callers can keep one per
// worker/tile context and pass CDFs() into DecodeAndReconstructJobResiduals.
type FrameWorkTileResidualCDFStorage struct {
	Partition             tile.PartitionCDFs
	Mode                  tile.BlockModeCDFs
	Intra                 tile.IntraModeCDFs
	InterRef              tile.InterRefCDFs
	InterMode             tile.InterModeCDFs
	MV                    tile.MVCDFs
	DV                    tile.MVCDFs
	Interp                tile.InterpFilterCDFs
	Motion                tile.MotionModeCDFs
	Blend                 tile.CompoundBlendCDFs
	Transform             tile.TransformCDFs
	TransformType         tile.TransformTypeCDFs
	Coeff                 tile.CoeffCDFs
	DeltaQ                entropy.CDF
	DeltaLF               entropy.CDF
	DeltaLFMulti          [tile.FrameLoopFilterCount]entropy.CDF
	RestorationSwitchable entropy.CDF
	RestorationWiener     entropy.CDF
	RestorationSGRProj    entropy.CDF
}

// InitDefault seeds the storage with the CDF defaults ported from libaom/dav1d.
func (s *FrameWorkTileResidualCDFStorage) InitDefault(baseQIndex uint8) error {
	if s == nil {
		return ErrInvalidBatch
	}
	defaults, err := defaultTileResidualCDFStorage(baseQIndex)
	if err != nil {
		return err
	}
	copyCDFStorage(s, defaults)
	return nil
}

// buildDefault fills s with the defaults; InitDefault copies them from a
// per-q-context cache built once by this.
func (s *FrameWorkTileResidualCDFStorage) buildDefault(baseQIndex uint8) error {
	if err := s.Partition.InitDefault(); err != nil {
		return err
	}
	if err := s.Mode.InitDefault(); err != nil {
		return err
	}
	if err := s.Intra.InitDefault(); err != nil {
		return err
	}
	if err := s.InterRef.InitDefault(); err != nil {
		return err
	}
	if err := s.InterMode.InitDefault(); err != nil {
		return err
	}
	if err := s.MV.InitDefault(); err != nil {
		return err
	}
	if err := s.DV.InitDefault(); err != nil {
		return err
	}
	if err := s.Interp.InitDefault(); err != nil {
		return err
	}
	if err := s.Motion.InitDefault(); err != nil {
		return err
	}
	if err := s.Blend.InitDefault(); err != nil {
		return err
	}
	if err := s.Transform.InitDefault(); err != nil {
		return err
	}
	if err := s.TransformType.InitDefault(); err != nil {
		return err
	}
	if err := s.Coeff.InitDefault(baseQIndex); err != nil {
		return err
	}
	if err := s.DeltaQ.InitDefaultDelta(); err != nil {
		return err
	}
	if err := s.DeltaLF.InitDefaultDelta(); err != nil {
		return err
	}
	for i := range tile.FrameLoopFilterCount {
		if err := s.DeltaLFMulti[i].InitDefaultDelta(); err != nil {
			return err
		}
	}
	if err := tile.InitDefaultRestorationCDFs(tile.RestorationCDFs{
		Switchable: &s.RestorationSwitchable,
		Wiener:     &s.RestorationWiener,
		SGRProj:    &s.RestorationSGRProj,
	}); err != nil {
		return err
	}
	return nil
}

// CDFs returns the pointer view consumed by the residual driver.
func (s *FrameWorkTileResidualCDFStorage) CDFs() FrameWorkTileResidualCDFs {
	if s == nil {
		return FrameWorkTileResidualCDFs{}
	}
	delta := tile.DeltaCDFs{
		Q:  &s.DeltaQ,
		LF: &s.DeltaLF,
	}
	for i := range tile.FrameLoopFilterCount {
		delta.LFMulti[i] = &s.DeltaLFMulti[i]
	}
	return FrameWorkTileResidualCDFs{
		Loop: tile.BlockLoopCDFs{
			Partition: &s.Partition,
			Mode:      &s.Mode,
			Intra:     &s.Intra,
			InterRef:  &s.InterRef,
			InterMode: &s.InterMode,
			MV:        &s.MV,
			DV:        &s.DV,
			Interp:    &s.Interp,
			Motion:    &s.Motion,
			Blend:     &s.Blend,
			Transform: &s.Transform,
			Coeff:     &s.Coeff,
			Delta:     delta,
		},
		Coeff: tile.BlockCoeffCDFs{
			Transform: &s.Transform,
			Coeff:     &s.Coeff,
		},
		TransformType: &s.TransformType,
		Restoration: tile.RestorationCDFs{
			Switchable: &s.RestorationSwitchable,
			Wiener:     &s.RestorationWiener,
			SGRProj:    &s.RestorationSGRProj,
		},
	}
}

// InitTileResidualCDFStorage seeds storage from the frame context selected for
// this tile group, falling back to AV1 defaults when the batch is used outside
// decoder-managed frame work.
func (b *FrameWorkBatch) InitTileResidualCDFStorage(storage *FrameWorkTileResidualCDFStorage) error {
	if storage == nil {
		return ErrInvalidBatch
	}
	if b.InitialTileResidualCDFs != nil {
		copyCDFStorage(storage, b.InitialTileResidualCDFs)
		return nil
	}
	return storage.InitDefault(b.Quantization.BaseQIdx)
}

// RetainTileResidualCDFStorage publishes storage as the adapted frame context
// only for the context_update_tile_id job when CDF updates are enabled.
func (b *FrameWorkBatch) RetainTileResidualCDFStorage(index int, state *tile.DecodeState, storage *FrameWorkTileResidualCDFStorage) error {
	return b.retainTileResidualCDFStorage(index, state, storage, false)
}

// RetainTileResidualCDFStorageOwned is the private decoder retention entry.
// Its caller has successfully decoded into initialized, exclusively owned
// storage. Caller hooks and generic batch entry points retain full validation.
func RetainTileResidualCDFStorageOwned(b *FrameWorkBatch, index int, state *tile.DecodeState, storage *FrameWorkTileResidualCDFStorage, req *FrameWorkTileResidualRequest, compact *FrameWorkLoopFilterCompactState) error {
	if b == nil {
		return ErrInvalidBatch
	}
	trusted := FrameWorkLoopFilterCompactStateActive(compact) &&
		FrameWorkLoopFilterCompactMatchesMasks(b.LoopFilterMasks, compact) &&
		frameWorkOwnedCDFCounts(b, req)
	return b.retainTileResidualCDFStorage(index, state, storage, trusted)
}

func frameWorkOwnedCDFCounts(b *FrameWorkBatch, req *FrameWorkTileResidualRequest) bool {
	return req != nil && b.WavefrontWorkers == 0 &&
		req.AfterBlock == nil && req.Loop.BeforeSuperblock == nil
}

func (b *FrameWorkBatch) retainTileResidualCDFStorage(index int, state *tile.DecodeState, storage *FrameWorkTileResidualCDFStorage, trusted bool) error {
	if index < 0 || index >= len(b.Jobs) || state == nil || storage == nil {
		return ErrInvalidBatch
	}
	if !state.RetainFrameContext {
		return nil
	}
	updates, err := b.JobUpdatesFrameContext(index)
	if err != nil {
		return err
	}
	if !updates {
		return nil
	}
	if b.RetainedTileResidualCDFs == nil || b.RetainedTileResidualCDFsValid == nil {
		return ErrInvalidBatch
	}
	if trusted {
		storage.resetCDFCountsTrusted()
	} else if err := storage.ResetCDFCounts(); err != nil {
		return err
	}
	copyCDFStorage(b.RetainedTileResidualCDFs, storage)
	*b.RetainedTileResidualCDFsValid = true
	if ft := b.FrameThread; ft != nil && ft.CDFOut != nil {
		ft.CDFOut.Publish(FrameProgressDone)
	}
	return nil
}

// FrameWorkTileResidualScratch is caller-owned state reused while decoding one
// tile job's block loop and coefficient contexts.
type FrameWorkTileResidualScratch struct {
	Loop         tile.BlockLoopScratch
	LoopContext  tile.BlockLoopContextCarrier
	Coeff        tile.BlockCoeffScratch
	CoeffContext tile.CoeffEntropyContext
	CFL          FrameWorkCFLPredictionScratch

	controller       frameWorkTileResidualLoopController
	beforeSuperblock tile.BlockLoopSuperblockVisitor
	stats            FrameWorkTileResidualStats
	geomCache        frameWorkJobGeometryCache

	// lfMask carries the reusable decode-time loop-filter mask build state
	// (dav1d decomp_tx scratch + carried above/left tx-context). It is reset
	// once per job and holds capacity across jobs for zero steady-state allocs.
	lfMask frameWorkLoopFilterMaskScratch

	// reconVisits, reconCBI and cfArena are the frame-thread buffers of
	// dav1d's two-pass decode (f->frame_thread.b, .cbi and .cf): the parse
	// appends each block's visit, each coded TXB's cbi entry and its
	// coefficients, and reconstruction re-walks the blocks' transform layout
	// reading them back in the same order.
	// paletteArena holds deep-copied palette colour maps for buffered visits
	// whose prediction is palette-coded: the decoded BlockLoopVisit.Prediction
	// carries YMap/UVMap pointers into a single per-tile PaletteModeScratch that
	// later blocks overwrite, so a buffered visit must own its own copy. Palette
	// events store indexes while the arena is still appendable; pointers into the
	// final value arena are installed immediately before replay. All buffers
	// retain capacity across tiles and are sliced to zero length per job by
	// resetDeferredReconBuffers.
	reconVisits []tile.BlockLoopVisit
	reconCBI    []frameWorkReconCBI
	cfArena     []int32
	// cfOverflow reports that a TXB did not fit the job's cf reservation; it
	// then decoded into cfSpare.
	cfOverflow bool
	cfSpare    [1024]int32
	// cfDirty bounds the arena prefix earlier jobs may have left
	// coefficients in, including the slot a failed decode_coefs wrote past
	// the committed length.
	cfDirty int
	// coefBuf is the fused path's dav1d t->cf coefficient buffer.
	coefBuf         coefBuffer
	paletteArena    []tile.PaletteModeScratch
	paletteBindings []frameWorkReconPaletteBinding
	// extrasArena holds the copied warp/OBMC payloads of buffered visits
	// (BlockPredictionModeResult.Extras points into per-tile decode state);
	// they are bound like the palette maps.
	extrasArena []tile.BlockPredictionMotionExtras
	// subChromaArena holds the copied sub8x8 chroma cells, bound the same way.
	subChromaArena []tile.SubChromaInterResult

	// wavefront holds the reusable SB-row wavefront scheduling state used when
	// the deferred reconstruction replay runs across multiple goroutines. It is
	// lazily allocated only on the multi-worker wavefront path (the fused
	// single-thread and serial deferred replay never touch it, so they pay
	// nothing); it is a pointer so the scratch struct stays copyable (the
	// wavefront state embeds sync/atomic counters, which must not be copied). Its
	// slices and per-goroutine state retain capacity across frames.
	wavefront *frameWorkReconWavefront
}

// PreallocCallbackScratch primes reusable method-value callbacks used by the
// block-loop adapter so the first restoration tile does not allocate them.
func (s *FrameWorkTileResidualScratch) PreallocCallbackScratch() {
	if s == nil || s.beforeSuperblock != nil {
		return
	}
	s.beforeSuperblock = s.controller.BeforeSuperblock
}

// PreallocLoopFilterMaskScratch reserves the decode-time loop-filter mask build
// above-context arrays (aY/aUV, one uint8 per MI column) so the first frame that
// builds masks reuses them instead of allocating. cols is the frame's MI-column
// count; lfMask.reset re-slices and fills these each job.
func (s *FrameWorkTileResidualScratch) PreallocLoopFilterMaskScratch(cols int) {
	if s == nil || cols <= 0 {
		return
	}
	if cap(s.lfMask.aY) < cols {
		s.lfMask.aY = make([]uint8, cols)
	}
	if cap(s.lfMask.aUV) < cols {
		s.lfMask.aUV = make([]uint8, cols)
	}
	tile.PreallocBlockLoopRefmvsScratch(&s.Loop, cols)
}

// frameWorkReconWavefront owns the reusable per-tile SB-row bucketing and
// wavefront scheduling state. rowStart partitions the superblock spans into
// per-SB-row ranges ([rowStart[r], rowStart[r+1])); done[r] counts the SBs
// completed in row r so a goroutine reconstructing row r+1 can gate its reads of
// the upper-right neighbor. states holds one reconstruction state per wavefront
// goroutine (each with its own CFL state machine, prediction/residual scratch,
// and geometry cache) so the goroutines never share mutable reconstruction
// state. Numeric per-worker scratch is carved from flat arenas to avoid one
// backing allocation per worker. All buffers retain capacity across frames.
type frameWorkReconWavefront struct {
	rowStart       []int32
	sbSpans        []frameWorkReconSB
	build          frameWorkReconSpanBuilder
	done           []atomic.Int32
	states         []frameWorkReconState
	predict        []FrameWorkPredictionScratch
	inter          []FrameWorkInterPredictionScratch
	cfl            []FrameWorkCFLPredictionScratch
	int32Arena     []int32
	residualArena  []int16
	int32Stride    uint16
	residualStride uint16
	doneAlloc      []atomic.Int32
	progress       *frameWorkReconWavefrontProgress

	// pub and published (guarded by progress.mu) hand parsed SB rows to the
	// lanes: rows [0, published) of pub are complete.
	pub       frameWorkReconView
	published int
	lanes     int
	wg        sync.WaitGroup
	firstErr  atomic.Pointer[wavefrontError]

	// frameThread is the frame-thread split this wavefront belongs to, or nil.
	// Its lanes wait for the reference surfaces before the first row.
	frameThread *FrameThreadRecon
	// reconTile is the tile whose finished SB rows frameThread.Rows gets.
	reconTile int32
	// cfZero records that the last tile's lanes reconstructed, and so
	// cleared, every coefficient of the scratch's cf arena.
	cfZero bool
}

type frameWorkReconWavefrontProgress struct {
	mu      sync.Mutex
	cond    sync.Cond
	aborted atomic.Bool
	// waiters counts goroutines parked on cond. A publisher stores its
	// progress first and takes the lock to wake only when someone waits; a
	// waiter registers under the lock before rechecking, so with sequentially
	// consistent atomics one of the two always sees the other.
	waiters atomic.Int32
}

func (wf *frameWorkReconWavefront) appendRowStart(sbCount int) error {
	idx, ok := frameWorkReconIndex(sbCount)
	if !ok {
		return ErrInvalidBatch
	}
	wf.rowStart = append(wf.rowStart, idx)
	return nil
}

// ensureDone grows the per-row done-counter backing to rowCount entries,
// reusing the previously allocated array when it is already large enough.
// atomic.Int32 must not be copied once used, so the backing array is allocated
// once and resliced; entries are reset by the caller before each wavefront.
func (wf *frameWorkReconWavefront) ensureDone(rowCount int) {
	if cap(wf.doneAlloc) < rowCount {
		wf.doneAlloc = make([]atomic.Int32, rowCount)
	}
	wf.done = wf.doneAlloc[:rowCount]
}

func (wf *frameWorkReconWavefront) ensureProgress() *frameWorkReconWavefrontProgress {
	if wf.progress == nil {
		p := &frameWorkReconWavefrontProgress{}
		p.cond.L = &p.mu
		wf.progress = p
	}
	return wf.progress
}

func (wf *frameWorkReconWavefront) resetProgress(rowCount int) {
	p := wf.ensureProgress()
	p.mu.Lock()
	p.aborted.Store(false)
	for r := range rowCount {
		wf.done[r].Store(0)
	}
	wf.pub = frameWorkReconView{}
	wf.published = 0
	p.mu.Unlock()
}

func (wf *frameWorkReconWavefront) abortProgress() {
	p := wf.ensureProgress()
	p.mu.Lock()
	p.aborted.Store(true)
	p.cond.Broadcast()
	p.mu.Unlock()
}

func (wf *frameWorkReconWavefront) waitForRowProgress(done *atomic.Int32, need int32) bool {
	p := wf.progress
	if done.Load() >= need {
		return !p.aborted.Load()
	}
	p.mu.Lock()
	p.waiters.Add(1)
	for done.Load() < need {
		if p.aborted.Load() {
			p.waiters.Add(-1)
			p.mu.Unlock()
			return false
		}
		p.cond.Wait()
	}
	p.waiters.Add(-1)
	ok := !p.aborted.Load()
	p.mu.Unlock()
	return ok
}

func (wf *frameWorkReconWavefront) publishRowProgress(done *atomic.Int32, value int32) {
	p := wf.progress
	done.Store(value)
	if p.waiters.Load() != 0 {
		p.mu.Lock()
		p.cond.Broadcast()
		p.mu.Unlock()
	}
}

// ensureStates lazily allocates wavefrontWorkers per-goroutine reconstruction
// states with their own prediction/inter/CFL/residual scratch and a private
// geometry cache, retaining them across frames. Each state is rebound to the
// current job's batch, index, scratch, and request every frame; only the heap
// scratch is reused.
func (wf *frameWorkReconWavefront) ensureStates(workers int, c *frameWorkTileResidualLoopController) error {
	int32Len := len(c.req.Int32Scratch)
	residualLen := len(c.req.ResidualScratch)
	if int32Len > int(^uint16(0)) || residualLen > int(^uint16(0)) {
		return ErrInvalidBatch
	}
	if cap(wf.states) < workers {
		wf.states = make([]frameWorkReconState, workers)
		wf.predict = make([]FrameWorkPredictionScratch, workers)
		wf.inter = make([]FrameWorkInterPredictionScratch, workers)
		wf.cfl = make([]FrameWorkCFLPredictionScratch, workers)
	}
	wf.states = wf.states[:workers]
	wf.int32Stride = uint16(int32Len)
	wf.residualStride = uint16(residualLen)
	int32Total := workers * int32Len
	residualTotal := workers * residualLen
	if cap(wf.int32Arena) < int32Total {
		wf.int32Arena = make([]int32, int32Total)
	}
	wf.int32Arena = wf.int32Arena[:int32Total]
	if cap(wf.residualArena) < residualTotal {
		wf.residualArena = make([]int16, residualTotal)
	}
	wf.residualArena = wf.residualArena[:residualTotal]
	for w := range workers {
		wf.predict[w].Inter = &wf.inter[w]
		// Each goroutine reconstructs through its own batch copy whose geometry
		// cache is private, so the per-block JobRegion/JobOutputPlane memoization
		// never races. The cache is per-job; reset it before use.
		st := &wf.states[w]
		st.batch = c.batch
		st.index = c.index
		st.geom.reset()
		st.batch.geomCache = &st.geom
		_ = frameWorkPrimeLumaReconGeometry(&st.batch, c.index)
		st.bindPrimedLumaReconGeometry()
		st.localStats = FrameWorkTileResidualStats{}
		st.stats = &st.localStats
		st.predictionScratch = &wf.predict[w]
		st.cflScratch = &wf.cfl[w]
		st.int32Scratch = wf.workerInt32(w)
		st.residualScratch = wf.workerResidual(w)
		st.pendingCFLPrediction = false
		st.cflPredictionDone = false
		st.cflVisit = tile.BlockLoopVisit{}
	}
	return nil
}

func (wf *frameWorkReconWavefront) workerInt32(worker int) []int32 {
	stride := int(wf.int32Stride)
	off := worker * stride
	return wf.int32Arena[off : off+stride]
}

func (wf *frameWorkReconWavefront) workerResidual(worker int) []int16 {
	stride := int(wf.residualStride)
	off := worker * stride
	return wf.residualArena[off : off+stride]
}

// reserveCFArena sizes the deferred cf arena once for the job instead of letting
// it grow geometrically through its first frames. Each TXB keeps at most its
// pixel area of coefficients (a scan area never exceeds the transform area,
// and transforms past the visible block are not coded), so the job's
// superblock-aligned luma area plus its chroma planes bounds what one job
// appends; the extra 1024 is the slot arenaSlot reserves ahead of a commit.
func (s *FrameWorkTileResidualScratch) reserveCFArena(walk tile.BlockWalkRequest, color parser.ColorConfig) {
	if walk.MIColEnd <= walk.MIColStart || walk.MIRowEnd <= walk.MIRowStart {
		return
	}
	w := (int(walk.MIColEnd-walk.MIColStart)*4 + 63) &^ 63
	h := (int(walk.MIRowEnd-walk.MIRowStart)*4 + 63) &^ 63
	luma := w * h
	need := luma + 1024
	if !color.MonoChrome {
		shift := 0
		if color.SubsamplingX {
			shift++
		}
		if color.SubsamplingY {
			shift++
		}
		need += 2 * (luma >> shift)
	}
	if cap(s.cfArena) < need {
		arena := make([]int32, len(s.cfArena), need)
		copy(arena, s.cfArena)
		s.cfArena = arena
		s.cfDirty = 0
	}
}

// resetDeferredReconBuffers slices the deferred reconstruction buffers back to
// zero length while retaining their capacity, so a worker reusing this scratch
// across jobs never replays a previous job's events or coefficients.
func (s *FrameWorkTileResidualScratch) resetDeferredReconBuffers() {
	s.reconVisits = s.reconVisits[:0]
	s.reconCBI = s.reconCBI[:0]
	s.cfDirty = max(s.cfDirty, min(len(s.cfArena)+1024, cap(s.cfArena)))
	s.cfArena = s.cfArena[:0]
	s.cfOverflow = false
	s.paletteArena = s.paletteArena[:0]
	s.paletteBindings = s.paletteBindings[:0]
	s.extrasArena = s.extrasArena[:0]
	s.subChromaArena = s.subChromaArena[:0]
}

// nextPaletteScratch returns the value-arena index for a per-block palette map
// backing. Callers must store the index, not a pointer, while event collection
// is still appending: slice growth may move value elements.
func (s *FrameWorkTileResidualScratch) nextPaletteScratch() int {
	n := len(s.paletteArena)
	if n < cap(s.paletteArena) {
		s.paletteArena = s.paletteArena[:n+1]
		return n
	}
	s.paletteArena = append(s.paletteArena, tile.PaletteModeScratch{})
	return n
}

// bufferDeferredVisit appends a copy of visit to reconVisits for deferred
// replay and returns its index. The visit is copied once, straight into its
// slot (a BlockLoopVisit is close to 1 KiB, and this runs for every block of a
// deferred tile). The decoded visit's palette colour maps (YMap/UVMap) point
// into a single per-tile PaletteModeScratch that subsequent palette-coded
// blocks overwrite during pass 1; the fused path reads them immediately, but
// the deferred path replays prediction later, by which time the shared scratch
// holds a later block's map. Deep-copy the referenced map(s) into the scratch
// palette arena and return an index binding so pass 2 can install stable
// pointers after event collection. Non-palette visits (the overwhelming
// majority) pay no palette copy.
func (s *FrameWorkTileResidualScratch) bufferDeferredVisit(visit *tile.BlockLoopVisit) (int, frameWorkDeferredSlots) {
	n := len(s.reconVisits)
	if n < cap(s.reconVisits) {
		s.reconVisits = s.reconVisits[:n+1]
		s.reconVisits[n] = *visit
	} else {
		s.reconVisits = append(s.reconVisits, *visit)
	}
	p := &s.reconVisits[n].Prediction
	slots := frameWorkDeferredSlots{palette: -1, extras: -1, subChroma: -1}
	if p.Extras != nil {
		slots.extras = len(s.extrasArena)
		s.extrasArena = append(s.extrasArena, *p.Extras)
		p.Extras = nil
		slots.mask |= frameWorkReconMotionExtras
	}
	if p.SubChromaInter != nil {
		slots.subChroma = len(s.subChromaArena)
		s.subChromaArena = append(s.subChromaArena, *p.SubChromaInter)
		p.SubChromaInter = nil
		slots.mask |= frameWorkReconSubChroma
	}
	pal := &p.Palette
	if pal.YMap == nil && pal.UVMap == nil {
		return n, slots
	}
	slots.palette = s.nextPaletteScratch()
	store := &s.paletteArena[slots.palette]
	if pal.YMap != nil {
		store.Y = *pal.YMap
		pal.YMap = nil
		slots.mask |= frameWorkReconPaletteY
	}
	if pal.UVMap != nil {
		store.UV = *pal.UVMap
		pal.UVMap = nil
		slots.mask |= frameWorkReconPaletteUV
	}
	return n, slots
}

func (s *FrameWorkTileResidualScratch) rememberDeferredPalette(visitIndex int, slots frameWorkDeferredSlots) error {
	mask := slots.mask
	if mask == 0 {
		return nil
	}
	visitIndex32, ok := frameWorkReconIndex(visitIndex)
	if !ok {
		return ErrInvalidBatch
	}
	binding := frameWorkReconPaletteBinding{visitIndex: visitIndex32, paletteIndex: -1, extrasIndex: -1, subChromaIndex: -1, mask: mask}
	if mask&(frameWorkReconPaletteY|frameWorkReconPaletteUV) != 0 {
		if binding.paletteIndex, ok = frameWorkReconIndex(slots.palette); !ok {
			return ErrInvalidBatch
		}
	}
	if mask&frameWorkReconMotionExtras != 0 {
		if binding.extrasIndex, ok = frameWorkReconIndex(slots.extras); !ok {
			return ErrInvalidBatch
		}
	}
	if mask&frameWorkReconSubChroma != 0 {
		if binding.subChromaIndex, ok = frameWorkReconIndex(slots.subChroma); !ok {
			return ErrInvalidBatch
		}
	}
	s.paletteBindings = append(s.paletteBindings, binding)
	return nil
}

func (s *FrameWorkTileResidualScratch) finalizeDeferredPalettes() error {
	_, err := s.installDeferredPalettes(0)
	return err
}

// installDeferredPalettes installs the colour-map pointers of the palette
// bindings recorded since from and returns the new cursor. The arena slots are
// fully written before their binding is recorded and never written again, so
// a pointer stays valid after a later append moves the arena.
func (s *FrameWorkTileResidualScratch) installDeferredPalettes(from int) (int, error) {
	if from < 0 || from > len(s.paletteBindings) {
		return from, ErrInvalidBatch
	}
	for i := from; i < len(s.paletteBindings); i++ {
		binding := &s.paletteBindings[i]
		visitIndex := int(binding.visitIndex)
		if binding.visitIndex < 0 || visitIndex >= len(s.reconVisits) {
			return i, ErrInvalidBatch
		}
		visit := &s.reconVisits[visitIndex]
		if binding.mask&frameWorkReconMotionExtras != 0 {
			extrasIndex := int(binding.extrasIndex)
			if binding.extrasIndex < 0 || extrasIndex >= len(s.extrasArena) {
				return i, ErrInvalidBatch
			}
			visit.Prediction.Extras = &s.extrasArena[extrasIndex]
		}
		if binding.mask&frameWorkReconSubChroma != 0 {
			subChromaIndex := int(binding.subChromaIndex)
			if binding.subChromaIndex < 0 || subChromaIndex >= len(s.subChromaArena) {
				return i, ErrInvalidBatch
			}
			visit.Prediction.SubChromaInter = &s.subChromaArena[subChromaIndex]
		}
		if binding.mask&(frameWorkReconPaletteY|frameWorkReconPaletteUV) == 0 {
			continue
		}
		paletteIndex := int(binding.paletteIndex)
		if binding.paletteIndex < 0 || paletteIndex >= len(s.paletteArena) {
			return i, ErrInvalidBatch
		}
		store := &s.paletteArena[paletteIndex]
		pal := &visit.Prediction.Palette
		if binding.mask&frameWorkReconPaletteY != 0 {
			pal.YMap = &store.Y
		}
		if binding.mask&frameWorkReconPaletteUV != 0 {
			pal.UVMap = &store.UV
		}
	}
	return len(s.paletteBindings), nil
}

// FrameWorkTileRestorationRequest carries frame-wide restoration unit storage
// for one residual decode pass. References are copied into the worker-local
// controller so Wiener/SGR ref-subexp state advances without mutating the
// caller's request while the job runs.
type FrameWorkTileRestorationRequest struct {
	Buffers    FrameWorkRestorationFrameBuffers
	References [3]tile.RestorationReferences
}

// InitReferences seeds the per-plane restoration reference state used by AV1's
// restoration unit syntax.
func (r *FrameWorkTileRestorationRequest) InitReferences() error {
	if r == nil {
		return ErrInvalidBatch
	}
	for plane := range r.References {
		r.References[plane] = tile.DefaultRestorationReferences()
	}
	return nil
}

// FrameWorkTileResidualRequest describes one tile job residual decode pass.
type FrameWorkTileResidualRequest struct {
	Loop          tile.BlockLoopRequest
	TransformMode parser.TransformMode

	PredictionScratch *FrameWorkPredictionScratch
	CDEFIndexMap      *FrameWorkCDEFIndexMap
	LoopFilterMap     *FrameWorkLoopFilterMap
	LoopFilterMasks   *FrameWorkLoopFilterMasks
	Restoration       *FrameWorkTileRestorationRequest
	AfterBlock        tile.BlockLoopVisitor

	Int32Scratch    []int32
	ResidualScratch []int16
}

// FrameWorkTileResidualDefaultRequest carries the caller-owned buffers and
// optional maps used by DecodeAndReconstructJobResidualsDefault. The block-loop
// request and transform selection are derived from the batch's frame context.
type FrameWorkTileResidualDefaultRequest struct {
	CurrentSegmentMap  []uint8
	PreviousSegmentMap []uint8
	SegmentMapStride   uint16

	PredictionScratch *FrameWorkPredictionScratch
	Restoration       *FrameWorkTileRestorationRequest
	AfterBlock        tile.BlockLoopVisitor

	Int32Scratch    []int32
	ResidualScratch []int16
}

// FrameWorkTileResidualRunnerWorker owns the reusable hot-path state for one
// worker lane in FrameWorkTileResidualRunner.
type FrameWorkTileResidualRunnerWorker struct {
	State                  tile.DecodeState
	CDFStorage             FrameWorkTileResidualCDFStorage
	Scratch                FrameWorkTileResidualScratch
	PredictionScratch      FrameWorkPredictionScratch
	InterPredictionScratch FrameWorkInterPredictionScratch
	Int32Scratch           []int32
	ResidualScratch        []int16
	Stats                  FrameWorkTileResidualStats
}

// FrameWorkTileResidualRunner decodes and reconstructs every job assigned to a
// frame-work worker batch using only caller-owned per-worker storage.
type FrameWorkTileResidualRunner struct {
	Workers []FrameWorkTileResidualRunnerWorker
	Request FrameWorkTileResidualDefaultRequest
	Predict bool
}

// FrameWorkTileResidualStats summarizes the composed block-loop/coefficient
// decode and reconstruction work for one tile job.
type FrameWorkTileResidualStats struct {
	Loop tile.BlockLoopStats

	CoefficientBlocks int32
	SkippedBlocks     int32

	TXBs             int32
	NonZero          int32
	AllZero          int32
	EOBTotal         int32
	Residuals        int32
	Predictions      int32
	RestorationUnits int32
}

// Run implements FrameWorkBatchRunner for the default residual decode path.
func (r *FrameWorkTileResidualRunner) Run(b FrameWorkBatch) error {
	return r.RunPtr(&b)
}

// RunPtr is the hot no-copy runner used by Pool when available.
func (r *FrameWorkTileResidualRunner) RunPtr(b *FrameWorkBatch) error {
	if r == nil || b == nil || int(b.Batch.Worker) >= len(r.Workers) {
		return ErrInvalidBatch
	}
	worker := &r.Workers[b.Batch.Worker]
	worker.Stats = FrameWorkTileResidualStats{}
	for i := 0; i < len(b.Jobs); i++ {
		if err := b.JobDecodeState(i, &worker.State); err != nil {
			return err
		}
		if err := b.InitTileResidualCDFStorage(&worker.CDFStorage); err != nil {
			return err
		}
		req := r.Request
		req.Int32Scratch = worker.Int32Scratch
		req.ResidualScratch = worker.ResidualScratch
		var predictionScratch *FrameWorkPredictionScratch
		if r.Predict {
			worker.PredictionScratch.Inter = &worker.InterPredictionScratch
			predictionScratch = &worker.PredictionScratch
		}
		stats, err := b.DecodeAndReconstructJobResidualsDefault(i, &worker.State, worker.CDFStorage.CDFs(), &worker.Scratch, req.withPredictionScratch(predictionScratch))
		worker.Stats.Add(stats)
		if err != nil {
			return err
		}
		if err := b.RetainTileResidualCDFStorage(i, &worker.State, &worker.CDFStorage); err != nil {
			return err
		}
	}
	return nil
}

func (req FrameWorkTileResidualDefaultRequest) withPredictionScratch(scratch *FrameWorkPredictionScratch) FrameWorkTileResidualDefaultRequest {
	req.PredictionScratch = scratch
	return req
}

// Add accumulates another residual stats snapshot into s.
func (s *FrameWorkTileResidualStats) Add(other FrameWorkTileResidualStats) {
	if s == nil {
		return
	}
	s.Loop.PartitionReads += other.Loop.PartitionReads
	s.Loop.Blocks += other.Loop.Blocks
	s.Loop.CoefficientBlocks += other.Loop.CoefficientBlocks
	s.CoefficientBlocks += other.CoefficientBlocks
	s.SkippedBlocks += other.SkippedBlocks
	s.TXBs += other.TXBs
	s.NonZero += other.NonZero
	s.AllZero += other.AllZero
	s.EOBTotal += other.EOBTotal
	s.Residuals += other.Residuals
	s.Predictions += other.Predictions
	s.RestorationUnits += other.RestorationUnits
}

// JobBlockLoopRequest derives the block-loop request for Jobs[index] from the
// frame context carried by this batch.
func (b *FrameWorkBatch) JobBlockLoopRequest(index int, currentSegmentMap []uint8, previousSegmentMap []uint8, segmentMapStride uint16) (tile.BlockLoopRequest, error) {
	if segmentMapStride == 0 && currentSegmentMap == nil && previousSegmentMap == nil && b.SegmentMaps != nil {
		// No caller override: use the frame's own segmentation maps.
		currentSegmentMap, previousSegmentMap, segmentMapStride = b.SegmentMaps.Current, b.SegmentMaps.Previous, b.SegmentMaps.Stride
	}
	region, err := b.JobRegion(index)
	if err != nil {
		return tile.BlockLoopRequest{}, err
	}
	globalMVs, globalTypes := frameWorkBlockLoopGlobalMotion(b.GlobalMotion, b.TileInfo.AllowHighPrecisionMV, b.FrameHeader.ForceIntegerMV)
	refSignBias, err := frameWorkBlockLoopRefSignBias(b.Sequence.EnableOrderHint, b.Sequence.OrderHintBits, b.FrameHeader.OrderHint, b.ReferenceOrderHints)
	if err != nil {
		return tile.BlockLoopRequest{}, err
	}
	refFrameSide, err := frameWorkBlockLoopRefFrameSide(b.Sequence.EnableOrderHint, b.Sequence.OrderHintBits, b.FrameHeader.OrderHint, b.ReferenceOrderHints)
	if err != nil {
		return tile.BlockLoopRequest{}, err
	}
	scaledReferences := b.frameWorkBlockLoopScaledReferences()
	// Frame MI extent for ref-MV clamping (clamp_mv_ref). On overflow the
	// extent stays zero, which disables the clamp rather than misbounding it.
	frameMIRows, _ := frameWorkMIExtent(b.FrameSize.Height)
	frameMICols, _ := frameWorkMIExtent(b.FrameSize.CodedWidth)
	if frameMIRows > uint32(^uint16(0)) || frameMICols > uint32(^uint16(0)) {
		return tile.BlockLoopRequest{}, ErrInvalidBatch
	}
	return tile.BlockLoopRequest{
		Walk: tile.BlockWalkRequest{
			Root:       tile.RootBlockLevel(b.Sequence.Use128x128Superblock),
			MIColStart: region.MIColStart,
			MIRowStart: region.MIRowStart,
			MIColEnd:   region.MIColEnd,
			MIRowEnd:   region.MIRowEnd,
		},
		SkipMode:                    b.SkipMode,
		CDEF:                        b.CDEF,
		Segmentation:                b.Segmentation,
		Delta:                       b.Delta,
		SBSizeMIB:                   b.Sequence.SBSizeMIB,
		Monochrome:                  b.Sequence.ColorConfig.MonoChrome,
		Color:                       b.Sequence.ColorConfig,
		Lossless:                    b.Segmentation.AllLossless,
		CurrentSegmentMap:           currentSegmentMap,
		PreviousSegmentMap:          previousSegmentMap,
		SegmentMapStride:            segmentMapStride,
		FrameType:                   b.FrameHeader.FrameType,
		AllowIntrabc:                b.FrameSize.AllowIntrabc,
		ReferenceMode:               b.TransformRef.ReferenceMode,
		GlobalMVs:                   globalMVs,
		GlobalMotion:                b.GlobalMotion.Ref,
		GlobalMotionTypes:           globalTypes,
		RefSignBias:                 refSignBias,
		RefFrameSide:                refFrameSide,
		ScaledReferences:            scaledReferences,
		ReferenceOrderHints:         b.ReferenceOrderHints,
		InterpolationFilter:         b.TileInfo.InterpolationFilter,
		EnableDualFilter:            b.Sequence.EnableDualFilter,
		EnableFilterIntra:           b.Sequence.EnableFilterIntra,
		AllowScreenContentTools:     b.FrameHeader.AllowScreenContentTools,
		AllowHighPrecisionMV:        b.TileInfo.AllowHighPrecisionMV,
		ForceIntegerMV:              b.FrameHeader.ForceIntegerMV,
		EnableInterIntraCompound:    b.Sequence.EnableInterIntraCompound,
		SwitchableMotionMode:        b.TileInfo.SwitchableMotionMode,
		AllowWarpedMotion:           b.FrameMode.AllowWarpedMotion,
		EnableMaskedCompound:        b.Sequence.EnableMaskedCompound,
		EnableDistWtdCompound:       b.Sequence.EnableJNTComp,
		EnableOrderHint:             b.Sequence.EnableOrderHint,
		UseRefFrameMVS:              b.TileInfo.UseRefFrameMVS,
		TemporalMVSampleUnavailable: b.TileInfo.UseRefFrameMVS,
		OrderHintBits:               b.Sequence.OrderHintBits,
		CurrentOrderHint:            b.FrameHeader.OrderHint,
		TemporalMVs:                 b.TemporalMVs,
		TemporalReferences:          b.deferredTemporalReferences(),
		CurrentMVFrame:              b.CurrentMVFrame,
		FrameMIRows:                 uint16(frameMIRows),
		FrameMICols:                 uint16(frameMICols),
		MotionProgress:              b.frameThreadMotionProgress(index),
		SkipModeRefs: [2]tile.ReferenceFrame{
			tile.ReferenceFrame(b.SkipMode.RefFrameIdx[0]),
			tile.ReferenceFrame(b.SkipMode.RefFrameIdx[1]),
		},
	}, nil
}

func frameWorkBlockLoopGlobalMotion(params parser.GlobalMotionParams, allowHighPrecisionMV bool, forceIntegerMV bool) ([parser.InterRefsPerFrame]motion.Vector, [parser.InterRefsPerFrame]parser.GlobalMotionType) {
	var mvs [parser.InterRefsPerFrame]motion.Vector
	var types [parser.InterRefsPerFrame]parser.GlobalMotionType
	for i, ref := range params.Ref {
		types[i] = ref.Type
		if ref.Type == parser.GlobalMotionTranslation {
			row := ref.Matrix[0] >> 13
			col := ref.Matrix[1] >> 13
			if forceIntegerMV {
				row = (row >> 3) << 3
				col = (col >> 3) << 3
			} else if !allowHighPrecisionMV {
				row = (row >> 1) << 1
				col = (col >> 1) << 1
			}
			if mv, ok := frameWorkMotionVectorFromInt32(row, col); ok {
				mvs[i] = mv
			}
		}
	}
	return mvs, types
}

func frameWorkMotionVectorFromInt32(row int32, col int32) (motion.Vector, bool) {
	if row <= tile.MVLower || row >= tile.MVUpper || col <= tile.MVLower || col >= tile.MVUpper {
		return motion.Vector{}, false
	}
	return motion.Vector{Row: int16(row), Col: int16(col)}, true
}

func frameWorkBlockLoopRefSignBias(enabled bool, bits uint8, current uint8, refs [parser.InterRefsPerFrame]uint8) ([parser.InterRefsPerFrame]bool, error) {
	var bias [parser.InterRefsPerFrame]bool
	if !enabled {
		return bias, nil
	}
	for i, ref := range refs {
		distance, err := frameWorkRelativeOrderHint(bits, ref, current)
		if err != nil {
			return bias, err
		}
		bias[i] = distance > 0
	}
	return bias, nil
}

func frameWorkBlockLoopRefFrameSide(enabled bool, bits uint8, current uint8, refs [parser.InterRefsPerFrame]uint8) ([parser.InterRefsPerFrame]int8, error) {
	var side [parser.InterRefsPerFrame]int8
	if !enabled {
		return side, nil
	}
	for i, ref := range refs {
		distance, err := frameWorkRelativeOrderHint(bits, ref, current)
		if err != nil {
			return side, err
		}
		if distance > 0 {
			side[i] = 1
		} else if ref == current {
			side[i] = -1
		}
	}
	return side, nil
}

// refNoScaleFP is libaom's REF_NO_SCALE: the Q14 (1 << REF_SCALE_SHIFT) value a
// per-axis scale factor takes when the reference and current frame share that
// axis's dimension. av1_is_scaled() reports a scaled reference when either axis
// differs from this.
const refNoScaleFP uint16 = 1 << motion.RefScaleShift

// frameWorkBlockLoopScaledReferences mirrors libaom's per-frame
// av1_setup_scale_factors_for_frame + av1_is_scaled() result for each inter
// reference slot. The decoder needs this during block syntax (not just
// prediction): motion_mode_allowed() reads the 2-symbol OBMC CDF instead of the
// 3-symbol WARP CDF when av1_is_scaled(block_ref_scale_factors[0]) is true (see
// av1/common/blockd.h motion_mode_allowed). A missing scaled flag selects the
// wrong CDF and desyncs the range decoder even though the decoded symbol value
// matches, as observed on the SVC L2T1/L2T2 spatial=1 enhancement layer whose
// references are the 2x-upscaled base layer.
//
// Dimensions follow libaom: the reference's y_crop_width/height versus the
// current frame's coded width/height (cm->width / cm->height).
func (b *FrameWorkBatch) frameWorkBlockLoopScaledReferences() [parser.InterRefsPerFrame]bool {
	var scaled [parser.InterRefsPerFrame]bool
	curWidth := int(b.FrameSize.CodedWidth)
	curHeight := int(b.FrameSize.Height)
	if curWidth <= 0 || curHeight <= 0 {
		return scaled
	}
	for i := 0; i < parser.InterRefsPerFrame && i < len(b.References); i++ {
		ref := b.References[i]
		if ref == nil {
			continue
		}
		refWidth := ref.Format.Width
		refHeight := ref.Format.Height
		if refWidth <= 0 || refHeight <= 0 {
			continue
		}
		if refWidth == curWidth && refHeight == curHeight {
			continue
		}
		sf, err := motion.NewScaleFactors(refWidth, refHeight, curWidth, curHeight)
		if err != nil {
			// An invalid scale ratio cannot drive the warped-motion path;
			// libaom would reject the frame earlier. Leave the slot unscaled.
			continue
		}
		scaled[i] = sf.XScaleFP != refNoScaleFP || sf.YScaleFP != refNoScaleFP
	}
	return scaled
}

// JobBlockLoopContextRootColumns returns the number of caller-owned above-edge
// carrier slots needed by Jobs[index]'s block-loop request.
func (b *FrameWorkBatch) JobBlockLoopContextRootColumns(index int) (int, error) {
	region, err := b.JobRegion(index)
	if err != nil {
		return 0, err
	}
	root := tile.RootBlockLevel(b.Sequence.Use128x128Superblock)
	rootSize := uint32(root.Size4x4())
	miColStart := uint32(region.MIColStart)
	miColEnd := uint32(region.MIColEnd)
	if rootSize == 0 || miColEnd <= miColStart {
		return 0, ErrInvalidBatch
	}
	return int((miColEnd - miColStart + rootSize - 1) / rootSize), nil
}

// DecodeAndReconstructJobResiduals walks one tile job's blocks, invokes the
// caller's prediction hook, decodes residual TXBs, and reconstructs each decoded
// TXB into the batch output frame.
func (b FrameWorkBatch) DecodeAndReconstructJobResiduals(index int, state *tile.DecodeState, cdfs FrameWorkTileResidualCDFs, scratch *FrameWorkTileResidualScratch, req FrameWorkTileResidualRequest) (FrameWorkTileResidualStats, error) {
	return decodeAndReconstructJobResidualsPtr(&b, index, state, cdfs, scratch, req, nil)
}

// DecodeAndReconstructJobResidualsPtr is the pointer-core form used by typed
// residual runners that already own a stack-local batch context.
func (b *FrameWorkBatch) DecodeAndReconstructJobResidualsPtr(index int, state *tile.DecodeState, cdfs FrameWorkTileResidualCDFs, scratch *FrameWorkTileResidualScratch, req FrameWorkTileResidualRequest) (FrameWorkTileResidualStats, error) {
	return decodeAndReconstructJobResidualsPtr(b, index, state, cdfs, scratch, req, nil)
}

// DecodeAndReconstructJobResidualsPtrOwned is the internal high-level
// decoder entry. Generic batch APIs always pass nil compact capability, even
// when handed a copied FrameWorkBatch that originated in an owned decode.
func DecodeAndReconstructJobResidualsPtrOwned(b *FrameWorkBatch, index int, state *tile.DecodeState, cdfs FrameWorkTileResidualCDFs, scratch *FrameWorkTileResidualScratch, req FrameWorkTileResidualRequest, compact *FrameWorkLoopFilterCompactState) (FrameWorkTileResidualStats, error) {
	if b == nil || compact == nil || !FrameWorkLoopFilterCompactStateActive(compact) ||
		b.LoopFilterMasks == nil || !FrameWorkLoopFilterCompactMatchesMasks(b.LoopFilterMasks, compact) {
		return FrameWorkTileResidualStats{}, ErrInvalidBatch
	}
	return decodeAndReconstructJobResidualsPtr(b, index, state, cdfs, scratch, req, compact)
}

func decodeAndReconstructJobResidualsPtr(b *FrameWorkBatch, index int, state *tile.DecodeState, cdfs FrameWorkTileResidualCDFs, scratch *FrameWorkTileResidualScratch, req FrameWorkTileResidualRequest, compact *FrameWorkLoopFilterCompactState) (FrameWorkTileResidualStats, error) {
	if state == nil || scratch == nil {
		return FrameWorkTileResidualStats{}, ErrInvalidBatch
	}
	if b.FrameThread != nil && scratch.wavefront != nil {
		// A frame-threaded batch decoding several tiles reuses this scratch:
		// the previous tile's lanes must finish reading its buffered events.
		if err := b.FrameThread.joinWavefront(scratch.wavefront); err != nil {
			return FrameWorkTileResidualStats{}, err
		}
	}
	scratch.stats = FrameWorkTileResidualStats{}
	if req.CDEFIndexMap == nil {
		req.CDEFIndexMap = b.CDEFIndexMap
	}
	if req.LoopFilterMap == nil {
		req.LoopFilterMap = b.LoopFilterMap
	}
	if req.LoopFilterMasks == nil {
		req.LoopFilterMasks = b.LoopFilterMasks
	}
	compactActive := FrameWorkLoopFilterCompactStateActive(compact)
	if compactActive {
		if req.LoopFilterMasks == nil {
			return FrameWorkTileResidualStats{}, ErrInvalidBatch
		}
		cols, rows, _, err := b.LoopFilterMapShape()
		if err != nil || !FrameWorkLoopFilterCompactActiveShape(req.LoopFilterMasks, compact, cols, rows) {
			return FrameWorkTileResidualStats{}, ErrInvalidBatch
		}
	}

	loopCDFs := cdfs.Loop
	if loopCDFs.Transform == nil {
		loopCDFs.Transform = cdfs.Coeff.Transform
	}
	if loopCDFs.Coeff == nil {
		loopCDFs.Coeff = cdfs.Coeff.Coeff
	}
	loopReq := req.Loop
	if loopReq.ContextCarrier == nil && scratch.LoopContext.Above != nil {
		loopReq.ContextCarrier = &scratch.LoopContext
	}
	loopReq.DecodeCoefficients = true
	ownedIntraContexts := compactActive && frameWorkOwnedIntraContexts(b, scratch, req, loopReq)
	var restoration FrameWorkTileRestorationRequest
	readRestoration := false
	if req.Restoration != nil {
		restoration = *req.Restoration
		readRestoration = true
	}
	// Install a per-job geometry cache so the job-constant JobRegion and
	// per-plane JobOutputPlane windows are computed once and reused across the
	// thousands of transform-block predict/reconstruct calls this job makes
	// (all sharing this index). The cache lives in caller-owned scratch and is
	// reset here so a worker reusing scratch across jobs never serves stale
	// geometry. It rides along on the batch value copied into the controller,
	// so every c.batch.* call below shares it.
	scratch.geomCache.reset()
	scratch.resetDeferredReconBuffers()
	if req.LoopFilterMasks.Valid() {
		scratch.lfMask.reset(req.LoopFilterMasks)
		if scratch.lfMask.edges != nil {
			region, err := b.JobRegion(index)
			if err != nil {
				return FrameWorkTileResidualStats{}, err
			}
			scratch.lfMask.setTile(region)
		}
	}
	b.geomCache = &scratch.geomCache
	if err := frameWorkPrimeLumaReconGeometry(b, index); err != nil {
		return FrameWorkTileResidualStats{}, err
	}
	// The deferred wavefront engages only when the pool offered idle lanes
	// (single-tile frame on a multi-worker pool) and the tile clears the size
	// threshold. When it does not engage, the fused single-thread path runs
	// verbatim at zero added cost.
	wavefrontWorkers := frameWorkWavefrontLanes(*b, index)
	// A frame-threaded tile always reconstructs through the lanes, even when the
	// tile is too small to fill two of them: its parse must never predict, since
	// the reference pixels may still be in flight.
	streamRecon := wavefrontWorkers > 1
	if b.FrameThread != nil {
		wavefrontWorkers = max(wavefrontWorkers, 1)
		streamRecon = true
	}
	if streamRecon {
		scratch.reserveCFArena(loopReq.Walk, b.Sequence.ColorConfig)
	}
	batchValue := *b
	scratch.controller = frameWorkTileResidualLoopController{
		batch:                batchValue,
		index:                index,
		state:                state,
		cdfs:                 cdfs,
		scratch:              scratch,
		req:                  req,
		stats:                &scratch.stats,
		userBeforeSuperblock: loopReq.BeforeSuperblock,
		restoration:          restoration,
		readRestoration:      readRestoration,
		deferReconstruction:  streamRecon,
		wavefrontWorkers:     wavefrontWorkers,
	}
	// Wire the fused / serial-deferred reconstruction state to the controller's
	// request scratch and shared geometry cache. The fused single-thread path
	// pays no extra copy: it dereferences exactly the buffers the previous
	// inline calls used.
	scratch.controller.recon = frameWorkReconState{
		batch:             batchValue,
		index:             index,
		stats:             &scratch.stats,
		predictionScratch: req.PredictionScratch,
		cflScratch:        &scratch.CFL,
		int32Scratch:      req.Int32Scratch,
		residualScratch:   req.ResidualScratch,
	}
	scratch.controller.recon.bindPrimedLumaReconGeometry()
	scratch.controller.reconHot = &scratch.controller.recon
	if loopReq.BeforeSuperblock != nil || readRestoration || streamRecon {
		if scratch.beforeSuperblock == nil {
			scratch.beforeSuperblock = scratch.controller.BeforeSuperblock
		}
		loopReq.BeforeSuperblock = scratch.beforeSuperblock
	}

	loopReq.TransformMode = req.TransformMode
	loopReq.ReducedTxSet = b.FrameMode.ReducedTxSet
	loopReq.QuantDeltasZero = frameWorkQuantDeltasZero(b.Quantization)
	loopCDFs.TransformType = cdfs.TransformType
	if streamRecon {
		if err := scratch.controller.startReconStream(); err != nil {
			return scratch.stats, err
		}
	}
	visitBlock := func(visit *tile.BlockLoopVisit) error {
		if !visit.CoefficientsValid {
			return ErrInvalidBatch
		}
		if req.LoopFilterMasks.Valid() {
			sbLog2 := 4
			if b.Sequence.Use128x128Superblock {
				sbLog2 = 5
			}
			if compactActive {
				if err := req.LoopFilterMasks.buildCompact(compact, &scratch.lfMask, visit, state, sbLog2); err != nil {
					return fmt.Errorf("build compact loop filter block=%+v prediction=%+v: %w", visit.Block, visit.Prediction, err)
				}
			} else {
				// dav1d keeps no per-MI loop-filter record: once the masks carry
				// levels, noskip and tile-edge backups, nothing reads the map.
				if req.LoopFilterMap != nil && !req.LoopFilterMasks.MapFree() {
					if err := req.LoopFilterMap.MarkBlockPtr(visit, state); err != nil {
						return fmt.Errorf("mark loop filter block=%+v prediction=%+v: %w", visit.Block, visit.Prediction, err)
					}
				}
				intra := !visit.Prediction.Valid || visit.Prediction.Intra
				if err := req.LoopFilterMasks.build(&scratch.lfMask, visit, state, sbLog2, intra); err != nil {
					return fmt.Errorf("build loop filter masks block=%+v prediction=%+v: %w", visit.Block, visit.Prediction, err)
				}
			}
		} else if req.LoopFilterMap != nil {
			if err := req.LoopFilterMap.MarkBlockPtr(visit, state); err != nil {
				return fmt.Errorf("mark loop filter block=%+v prediction=%+v: %w", visit.Block, visit.Prediction, err)
			}
		}
		if req.AfterBlock != nil {
			return req.AfterBlock(*visit)
		}
		return nil
	}
	var loopStats tile.BlockLoopStats
	var err error
	if ownedIntraContexts {
		loopStats, err = tile.DecodeBlockLoopWithCoeffControllerPtrOwnedIntra(state, loopCDFs, &scratch.Loop, loopReq, &scratch.controller, visitBlock)
	} else {
		loopStats, err = tile.DecodeBlockLoopWithCoeffControllerPtr(state, loopCDFs, &scratch.Loop, loopReq, &scratch.controller, visitBlock)
	}
	scratch.stats.Loop = loopStats
	if err == nil && req.LoopFilterMasks.Valid() {
		// Back up the tile's edges before its last rows reach reconstruction:
		// a frame-threaded post-filter fixes each region's tile boundaries
		// once the region is reconstructed.
		scratch.lfMask.finishTileEdges()
	}
	if scratch.controller.streaming {
		err = scratch.controller.finishReconStream(err)
	}
	if err != nil {
		return scratch.stats, err
	}
	if ft := b.FrameThread; !streamRecon && ft != nil && ft.Rows != nil && index < len(b.Jobs) {
		// A tile reconstructed without lanes finished every row here.
		ft.Rows.finishRows(int(b.Jobs[index].Tile), FrameProgressDone)
	}
	return scratch.stats, nil
}

// frameWorkOwnedIntraContexts is used only after the compact decoder ownership
// capability has been checked. Caller callbacks or a different carrier retain
// the complete public scratch semantics. The default controller is installed
// after this check, so its internal restoration callback is not a caller hook.
func frameWorkOwnedIntraContexts(b *FrameWorkBatch, scratch *FrameWorkTileResidualScratch, req FrameWorkTileResidualRequest, loop tile.BlockLoopRequest) bool {
	return (b.FrameHeader.FrameType == parser.FrameTypeKey || b.FrameHeader.FrameType == parser.FrameTypeIntraOnly) &&
		loop.FrameType == b.FrameHeader.FrameType && !b.FrameSize.AllowIntrabc && !loop.AllowIntrabc &&
		loop.DecodePredictionModes && !loop.SkipMode.Enabled && loop.ContextCarrier == &scratch.LoopContext &&
		!loop.Walk.UseNeighborBounds && b.WavefrontWorkers == 0 &&
		req.AfterBlock == nil && loop.BeforeSuperblock == nil
}

// frameWorkWavefrontLanes returns how many lanes the deferred SB-row
// wavefront runs for Jobs[index], or 0 to leave reconstruction fused. The
// wavefront helps only when the pool offered idle lanes (set by the pool when a
// single batch is dispatched) and every lane gets enough SB rows for the
// wavefront diagonal to fill before it drains. A tile with fewer rows than the
// offered lanes need runs on fewer lanes rather than none (a 720p frame with
// 128x128 superblocks has six SB rows: three lanes, not a serial fallback as
// soon as the pool exceeds three workers); below two lanes the fused
// single-thread path is the better choice (no buffering, no goroutines).
func frameWorkWavefrontLanes(b FrameWorkBatch, index int) uint16 {
	workers := int(b.WavefrontWorkers)
	if workers <= 1 {
		return 0
	}
	// Intra block copy frames reconstruct blocks by copying from an earlier
	// already-reconstructed region of the SAME frame via a block displacement
	// vector. Unlike intra edge prediction (which reaches at most the upper-right
	// superblock, the halo the wavefront's done[R-1] >= C+2 gate covers), an
	// intrabc DV can point an arbitrary distance up and left, so the referenced
	// samples may still be unreconstructed when a lane reaches the copying block.
	// The SB-row wavefront cannot honour that dependency, so intrabc frames stay
	// on the fused serial reconstruction (byte-exact, worker-count invariant).
	// intrabc frames disable CDEF, loop filter, and loop restoration, so this is
	// the only within-frame parallelism they can take.
	if b.FrameSize.AllowIntrabc {
		return 0
	}
	region, err := b.JobRegion(index)
	if err != nil {
		return 0
	}
	lanes := min(workers, int(region.SBRows)/frameWorkWavefrontMinSBRowsPerWorker)
	if lanes <= 1 {
		return 0
	}
	return uint16(lanes)
}

// DecodeAndReconstructJobResidualsDefault derives the standard block-loop and
// transform requests from the frame-work batch, then decodes and reconstructs
// one tile job using caller-owned scratch buffers.
func (b *FrameWorkBatch) DecodeAndReconstructJobResidualsDefault(index int, state *tile.DecodeState, cdfs FrameWorkTileResidualCDFs, scratch *FrameWorkTileResidualScratch, req FrameWorkTileResidualDefaultRequest) (FrameWorkTileResidualStats, error) {
	loopReq, err := b.JobBlockLoopRequest(index, req.CurrentSegmentMap, req.PreviousSegmentMap, req.SegmentMapStride)
	if err != nil {
		return FrameWorkTileResidualStats{}, err
	}
	return b.DecodeAndReconstructJobResidualsPtr(index, state, cdfs, scratch, FrameWorkTileResidualRequest{
		Loop:              loopReq,
		TransformMode:     b.TransformRef.TransformMode,
		PredictionScratch: req.PredictionScratch,
		Restoration:       req.Restoration,
		AfterBlock:        req.AfterBlock,
		Int32Scratch:      req.Int32Scratch,
		ResidualScratch:   req.ResidualScratch,
	})
}

type frameWorkTileResidualLoopController struct {
	batch   FrameWorkBatch
	index   int
	state   *tile.DecodeState
	cdfs    FrameWorkTileResidualCDFs
	scratch *FrameWorkTileResidualScratch
	req     FrameWorkTileResidualRequest
	stats   *FrameWorkTileResidualStats

	userBeforeSuperblock tile.BlockLoopSuperblockVisitor
	restoration          FrameWorkTileRestorationRequest
	readRestoration      bool

	// deferReconstruction selects dav1d's two-pass frame-thread path: the
	// parse buffers each block's visit, cbi entries and coefficients instead
	// of reconstructing inline, and the wavefront lanes reconstruct them.
	deferReconstruction bool

	// wavefrontWorkers is the number of idle pool goroutines available to fan
	// the deferred replay out as an SB-row wavefront. It is zero on the
	// single-worker and multi-tile paths, where the fused single-thread path
	// runs verbatim.
	wavefrontWorkers uint16

	// streaming is set while scratch.wavefront runs alongside a multi-lane
	// tile's parse (startReconStream).
	streaming bool

	// recon holds the reconstruction state for the fused single-thread path and
	// the serial deferred replay. The multi-worker wavefront uses its own
	// per-goroutine states instead. It is wired to the controller's request
	// scratch so the fused path pays no extra copy.
	recon    frameWorkReconState
	reconHot *frameWorkReconState

	// dq is the dav1d ts->dq[seg_id] (with cf_max and f->qm) ReconB resolved
	// for the (qindex, segment) key dqKey; dqOK is false when that key has no
	// valid quantizer.
	dq    [3]tile.CoefDequant
	dqKey uint16
	dqOK  bool
}

// frameWorkReconState is the per-goroutine reconstruction state: the CFL
// luma-first deferral state machine plus the prediction/residual scratch and a
// private geometry cache. Splitting it out of the controller lets the
// multi-worker wavefront give each goroutine its own isolated state (no shared
// mutable reconstruction state, no shared geomCache => no data race), while the
// fused single-thread path and the serial deferred replay keep using the single
// instance the controller owns. The batch carried here owns its own geomCache
// pointer; reconstruction reads only that goroutine's scratch and that
// goroutine's CFL fields.
type frameWorkReconState struct {
	batch             FrameWorkBatch
	index             int
	stats             *FrameWorkTileResidualStats
	predictionScratch *FrameWorkPredictionScratch
	cflScratch        *FrameWorkCFLPredictionScratch
	int32Scratch      []int32
	residualScratch   []int16

	// geom is this goroutine's private job-geometry cache; batch.geomCache
	// points at it on the wavefront path so the per-block memoization never
	// races. localStats accumulates the goroutine's reconstruction counters,
	// merged into the controller stats after the wavefront joins.
	geom        frameWorkJobGeometryCache
	localStats  FrameWorkTileResidualStats
	lumaReady   bool
	chromaReady bool

	pendingCFLPrediction bool
	cflPredictionDone    bool
	cflVisit             tile.BlockLoopVisit
}

func (s *frameWorkReconState) bindPrimedLumaReconGeometry() {
	cacheIndex, ok := frameWorkJobCacheIndex(s.index)
	c := s.batch.geomCache
	s.lumaReady = false
	s.chromaReady = false
	if c == nil || !ok ||
		c.validMask&frameWorkJobGeometryRegionValid == 0 ||
		c.regionIndex != cacheIndex {
		return
	}
	if c.validMask&frameWorkJobGeometryPlaneYValid == 0 ||
		c.planeIndex[FrameWorkPlaneY] != cacheIndex {
		s.lumaReady = false
		return
	}
	s.lumaReady = true
	if s.batch.Sequence.ColorConfig.MonoChrome {
		return
	}
	if c.validMask&(frameWorkJobGeometryPlaneUValid|frameWorkJobGeometryPlaneVValid) != frameWorkJobGeometryPlaneUValid|frameWorkJobGeometryPlaneVValid ||
		c.planeIndex[FrameWorkPlaneU] != cacheIndex || c.planeIndex[FrameWorkPlaneV] != cacheIndex {
		return
	}
	s.chromaReady = true
}

func (c *frameWorkTileResidualLoopController) BeforeSuperblock(visit tile.BlockLoopSuperblockVisit) error {
	if c.userBeforeSuperblock != nil {
		if err := c.userBeforeSuperblock(visit); err != nil {
			return err
		}
	}
	if c.deferReconstruction {
		if err := c.streamSuperblock(visit); err != nil {
			return err
		}
	}
	if !c.readRestoration {
		return nil
	}
	n, err := c.batch.ReadRestorationUnitsForSuperblock(c.index, c.state, c.cdfs.Restoration, &c.restoration, visit)
	if err != nil {
		return err
	}
	c.stats.RestorationUnits += int32(n)
	return nil
}

// reconBBegin is the block-level part of recon_b_*: the CDEF index mark and
// block counters, then the block's prediction, or its deferred-replay record.
func (c *frameWorkTileResidualLoopController) reconBBegin(visit *tile.BlockLoopVisit) error {
	if visit == nil {
		return ErrInvalidBatch
	}
	if c.req.CDEFIndexMap != nil {
		if err := c.req.CDEFIndexMap.MarkBlockPtr(c.batch.CDEF.Bits, visit); err != nil {
			return err
		}
	}
	if visit.Prefix.SkipTransform {
		c.stats.SkippedBlocks++
	} else {
		c.stats.CoefficientBlocks++
	}
	if c.deferReconstruction {
		return c.scratch.rememberDeferredPalette(c.scratch.bufferDeferredVisit(visit))
	}
	recon := c.reconHot
	if recon == nil {
		recon = c.fusedReconState()
	}
	return recon.predictBlockBegin(visit)
}

// fusedReconState returns the controller's reconstruction state for the fused
// single-thread path, binding it from the controller's request/scratch the
// first time it is needed. DecodeAndReconstructJobResiduals binds it up front
// on the production path; this lazy bind keeps direct-construction unit tests
// (which build the controller as a struct literal) working without each test
// wiring the state by hand. The bind is a single nil check per block on the hot
// path.
func (c *frameWorkTileResidualLoopController) fusedReconState() *frameWorkReconState {
	if c.reconHot != nil {
		return c.reconHot
	}
	if c.recon.stats == nil {
		c.recon = frameWorkReconState{
			batch:             c.batch,
			index:             c.index,
			stats:             c.stats,
			predictionScratch: c.req.PredictionScratch,
			int32Scratch:      c.req.Int32Scratch,
			residualScratch:   c.req.ResidualScratch,
		}
		if c.scratch != nil {
			c.recon.cflScratch = &c.scratch.CFL
		}
		c.recon.bindPrimedLumaReconGeometry()
	}
	c.reconHot = &c.recon
	return c.reconHot
}

// predictBlockBegin runs the predict portion of a block-loop visit: it resets
// the CFL deferral state machine, predicts luma/non-CfL chroma (or arms the
// pending-CfL state), and for skip-transform blocks flushes the deferred CfL
// chroma immediately (there are no TXBs to trigger it later). It is shared by
// the fused single-thread path (called inline from BeforeBlockCoefficients),
// the serial deferred replay, and each goroutine of the multi-worker wavefront,
// each operating on its own frameWorkReconState.
func (s *frameWorkReconState) predictBlockBegin(visit *tile.BlockLoopVisit) error {
	s.pendingCFLPrediction = false
	s.cflPredictionDone = false
	// dav1d's decode_b calls recon_b_inter directly: a bound translational
	// inter block goes straight to the per-plane mc() loop.
	if visit != nil && s.predictionScratch != nil && s.predictionScratch.Inter != nil {
		if l := s.batch.boundLeanInter(s.index); l != nil {
			if handled, err := s.batch.reconBInter(s.index, l, visit, s.predictionScratch.Inter, nil); handled {
				if err != nil {
					return frameWorkPredictBlockError(visit, err)
				}
				s.stats.Predictions++
				return nil
			}
		}
	}
	if err := s.predictBeforeCoefficientsPtr(visit); err != nil {
		return err
	}
	if visit.Prefix.SkipTransform && !visit.Prediction.Intra {
		if err := s.predictDeferredCFLChroma(); err != nil {
			return err
		}
	}
	return nil
}

func (s *frameWorkReconState) predictBeforeCoefficientsPtr(visit *tile.BlockLoopVisit) error {
	if visit == nil {
		return ErrInvalidBatch
	}
	if s.predictionScratch == nil {
		return nil
	}
	if !visit.Prediction.Valid {
		return nil
	}
	// Intra blocks, skip or not, are predicted transform block by transform
	// block from the TXB visits (dav1d recon_b_intra); skip intra blocks get
	// all-zero TXB visits from the coefficient walk.
	if visit.Prediction.Intra {
		if frameWorkVisitUsesCFLPtr(visit) {
			s.pendingCFLPrediction = true
			s.cflVisit = *visit
		}
		return nil
	}
	if frameWorkVisitUsesCFLPtr(visit) {
		if err := s.batch.PredictBlockLumaIntra(s.index, *visit, &s.predictionScratch.Intra); err != nil {
			return fmt.Errorf("predict cfl luma block=%+v: %w", visit.Block, err)
		}
		s.pendingCFLPrediction = true
		s.cflVisit = *visit
		s.stats.Predictions++
		return nil
	}
	if err := s.batch.predictBlockPtr(s.index, visit, s.predictionScratch); err != nil {
		return frameWorkPredictBlockError(visit, err)
	}
	s.stats.Predictions++
	return nil
}

// frameWorkPredictIntraTXBError and frameWorkReconstructTXBError wrap a
// transform block's prediction or reconstruction error, out of line.
//
//go:noinline
func frameWorkPredictIntraTXBError(visit *tile.BlockLoopVisit, block *tile.BlockCoeffBlock, err error) error {
	return fmt.Errorf("predict intra txb plane=%d visit=%+v block=%+v luma_mode=%d angle_delta=%d: %w", block.Plane, visit.Block, block.Block, visit.Prediction.LumaMode, visit.Prediction.LumaAngleDelta, err)
}

//go:noinline
func frameWorkReconstructTXBError(block *tile.BlockCoeffBlock, err error) error {
	return fmt.Errorf("reconstruct plane=%d block=%+v tx=%d: %w", block.Plane, block.Block, block.Transform, err)
}

// frameWorkPredictBlockError wraps a block prediction error with the block,
// kept out of line so the prediction paths carry no formatting frame.
//
//go:noinline
func frameWorkPredictBlockError(visit *tile.BlockLoopVisit, err error) error {
	return fmt.Errorf("predict block=%+v prediction=%+v prefix=%+v: %w", visit.Block, visit.Prediction, visit.Prefix, err)
}

func (s *frameWorkReconState) predictDeferredCFLChroma() error {
	if !s.pendingCFLPrediction || s.cflPredictionDone {
		return nil
	}
	if err := s.batch.PredictBlockChromaCFL(s.index, s.cflVisit, s.cflScratch); err != nil {
		return fmt.Errorf("predict cfl chroma block=%+v: %w", s.cflVisit.Block, err)
	}
	s.cflPredictionDone = true
	return nil
}

// frameWorkReconSB is one superblock of the frame-thread buffers: its blocks
// [start, end) in reconVisits, where its cbi entries and coefficients begin
// (dav1d's per-superblock ts->frame_thread[0].cbi / .cf cursors), and its SB
// column within its row. A superblock's blocks, cbi entries and coefficients
// are contiguous because the block loop decodes superblocks in raster order.
type frameWorkReconSB struct {
	start int32
	end   int32
	col   int32
	cbi   int32
	cf    int32
}

// wavefrontError boxes a worker error so atomic.Pointer can carry the first one
// observed across the wavefront goroutines.
type wavefrontError struct{ err error }

// reconstructRow reconstructs every superblock of one SB row in left-to-right
// order, gating each column on the row above via the done counters.
func (wf *frameWorkReconWavefront) reconstructRow(st *frameWorkReconState, view *frameWorkReconView, row int, rowCount int) error {
	// rowStart holds rowCount+1 entries and done holds rowCount; validating the
	// row index against both once lets the prove pass discharge the per-SB
	// rowStart[row], rowStart[row+1] and done[row] accesses below (and, under the
	// row>0 guard, the row-1 reads) without a check per superblock.
	rowStart := view.rowStart
	if row < 0 || row >= rowCount || row+1 >= len(rowStart) || row >= len(wf.done) {
		return ErrInvalidBatch
	}
	sbs := view.sbs
	lo32 := rowStart[row]
	hi32 := rowStart[row+1]
	if lo32 < 0 || hi32 < lo32 {
		return ErrInvalidBatch
	}
	lo := int(lo32)
	hi := int(hi32)
	if hi > len(sbs) {
		return ErrInvalidBatch
	}
	sbs = sbs[lo:hi]
	// The row-above SB count and the upper/this-row done counters depend only on
	// row, so resolve them once per row instead of per superblock. Pinning the
	// counter pointers also lifts their bounds checks out of the per-SB loop
	// (the function calls in the spin loop below otherwise force a re-check each
	// iteration). doneCur/donePrev alias wf.done entries proven in range above.
	doneCur := &wf.done[row]
	var donePrev *atomic.Int32
	aboveCount := 0
	if row > 0 {
		prevStart := rowStart[row-1]
		curStart := rowStart[row]
		if prevStart < 0 || curStart < prevStart {
			return ErrInvalidBatch
		}
		aboveCount = int(curStart - prevStart)
		donePrev = &wf.done[row-1]
	}
	visitCount, ok := frameWorkReconIndex(len(view.visits))
	if !ok {
		return ErrInvalidBatch
	}
	for k := range sbs {
		sb := sbs[k]
		if row > 0 {
			// SB C of row R reads its top neighbor (R-1,C) and, when it exists,
			// its top-right neighbor (R-1,C+1) — the furthest an intra top-right
			// edge read reaches. Gate on done[R-1] >= C+2 so both are
			// reconstructed, but clamp to the number of superblocks actually in
			// row R-1: if (R-1,C+1) does not exist (right edge), done[R-1] tops
			// out at the row-above SB count and waiting for C+2 would deadlock.
			need := sb.col + 2
			if int(need) > aboveCount {
				need = int32(aboveCount)
			}
			if !wf.waitForRowProgress(donePrev, need) {
				return nil
			}
		}
		if sb.start < 0 || sb.end < sb.start || sb.end > visitCount {
			return ErrInvalidBatch
		}
		visits := view.visits[sb.start:sb.end]
		if ft := wf.frameThread; ft != nil && !ft.waitSuperblockReferences(&st.batch, visits) {
			return ErrFrameThreadReference
		}
		cur := frameWorkReconCursor{cbi: view.cbi, cf: view.cfArena, cbiOff: int(sb.cbi), cfOff: int(sb.cf)}
		for i := range visits {
			if err := st.reconBDeferred(&visits[i], &cur); err != nil {
				return err
			}
		}
		wf.publishRowProgress(doneCur, sb.col+1)
	}
	return nil
}

// reconstructTXBWithDequantized is the prediction and inverse transform add of
// one TXB of recon_b_intra / recon_b_inter: the intra (or deferred CfL)
// prediction, then the residual of the coefficients decode_coefs dequantized
// into cf. An all-zero TXB is predicted only.
func (s *frameWorkReconState) reconstructTXBWithDequantized(visit *tile.BlockLoopVisit, block *tile.BlockCoeffBlock, cf []int32, activeRows int, activeCols int, dcOnly bool) error {
	if s.predictionScratch != nil && visit.Prediction.Valid && visit.Prediction.Intra {
		if block.Plane != 0 && frameWorkVisitUsesCFLPtr(visit) {
			if err := s.predictDeferredCFLChroma(); err != nil {
				return err
			}
		} else {
			if err := s.batch.predictBlockIntraCoeffPtr(s.index, visit, block, &s.predictionScratch.Intra); err != nil {
				return frameWorkPredictIntraTXBError(visit, block, err)
			}
			s.stats.Predictions++
		}
	} else if block.Plane != 0 {
		if err := s.predictDeferredCFLChroma(); err != nil {
			return err
		}
	}
	if block.Result.AllZero || block.Result.EOB == 0 {
		s.stats.Residuals++
		return nil
	}

	if meta, ok := frameWorkBlockCoeffTransformMeta(block.Block.Size); ok {
		done, err := s.addResidualInWindow(&visit.Block, block.Plane, block.Block.X4, block.Block.Y4, meta, block.Transform, block.Result.EOB, cf, activeRows, activeCols, dcOnly)
		if done || err != nil {
			return err
		}
	}
	var geom frameWorkBlockCoeffGeometry
	var err error
	if block.Plane == 0 && s.lumaReady {
		c := s.batch.geomCache
		geom, err = s.batch.blockCoeffGeometryLumaKnown(c.region, c.plane[FrameWorkPlaneY], visit.Block, block)
	} else if block.Plane != 0 && s.chromaReady {
		plane := FrameWorkPlaneU
		if block.Plane == 2 {
			plane = FrameWorkPlaneV
		} else if block.Plane != 1 {
			return frameWorkReconstructTXBError(block, ErrInvalidBatch)
		}
		c := s.batch.geomCache
		geom, err = s.batch.blockCoeffGeometryChromaKnown(c.region, c.plane[plane], plane, visit.Block, block)
	} else {
		geom, err = s.batch.blockCoeffGeometry(s.index, visit.Block, block)
	}
	if err != nil {
		return frameWorkReconstructTXBError(block, err)
	}
	if err := s.batch.reconstructBlockCoeffCoreWithGeometryDequantized(geom, block, activeRows, activeCols, dcOnly, cf, s.int32Scratch, s.residualScratch); err != nil {
		return frameWorkReconstructTXBError(block, err)
	}
	s.stats.Residuals++
	return nil
}

// addResidualInWindow is the inverse transform add of recon_b_intra /
// recon_b_inter: the destination is offset straight from the job's bound plane
// window (f->cur.data[pl] + 4 * (by4 * stride + bx4)) and the TXB at
// superblock-local x4/y4 goes to the inverse transform and add. It reports
// false, writing nothing, for a TXB not wholly inside the window's writable
// extent, which takes the validated geometry path.
func (s *frameWorkReconState) addResidualInWindow(vb *tile.BlockVisit, plane, x4, y4 uint8, meta frameWorkTransformMeta, txtp transform.Type, eob uint16, cf []int32, activeRows, activeCols int, dcOnly bool) (bool, error) {
	var window *FrameWorkPlaneRegion
	var x, y uint32
	if plane == 0 {
		if !s.lumaReady {
			return false, nil
		}
		window = &s.batch.geomCache.plane[FrameWorkPlaneY]
		x = (uint32(vb.MICol) + uint32(x4) - uint32(vb.X4)) << 2
		y = (uint32(vb.MIRow) + uint32(y4) - uint32(vb.Y4)) << 2
	} else {
		if plane > 2 || !s.chromaReady || vb.MICol < uint16(vb.X4) || vb.MIRow < uint16(vb.Y4) {
			return false, nil
		}
		window = &s.batch.geomCache.plane[plane]
		color := &s.batch.Sequence.ColorConfig
		ssX := frameWorkSubsampleShift(color.SubsamplingX)
		ssY := frameWorkSubsampleShift(color.SubsamplingY)
		x = ((uint32(vb.MICol)-uint32(vb.X4))>>ssX + uint32(x4)) << 2
		y = ((uint32(vb.MIRow)-uint32(vb.Y4))>>ssY + uint32(y4)) << 2
	}
	w, h := uint32(meta.size.Width), uint32(meta.size.Height)
	clipW, clipH := window.ClipWidth, window.ClipHeight
	if clipW == 0 {
		clipW = window.Width
	}
	if clipH == 0 {
		clipH = window.Height
	}
	if x < window.X || y < window.Y || uint64(x-window.X)+uint64(w) > uint64(clipW) || uint64(y-window.Y)+uint64(h) > uint64(clipH) {
		return false, nil
	}
	bps := int(window.BytesPerSample)
	stride := int(window.Stride)
	off := int(y-window.Y)*stride + int(x-window.X)*bps
	n := (int(h)-1)*stride + int(w)*bps
	dst := window.Pix[off : off+n : off+n]
	if bps == 1 && s.batch.Sequence.ColorConfig.BitDepth == 8 && txtp != transform.TypeIDTX && txtp != transform.TypeWHTWHT &&
		activeRows > 0 && activeCols > 0 {
		// dav1d's dsp->itx.itxfm_add[tx][txtp](dst, stride, cf, eob): the
		// 8-bit TXB goes straight to its kernel; a kernel that declines writes
		// nothing and the TXB takes the staged path below.
		if dcOnly {
			dsp.AddConstantResidualPlaneBlockTrusted(dst, stride, 1, 0xff, int(w), int(h), transform.InverseDCTDCOnlySample(cf[0], meta.size, 8))
			s.stats.Residuals++
			return true, nil
		}
		if itxfmAdd8(dst, stride, cf, meta.tx, txtp, int(meta.scanSize.Height), activeRows, activeCols) {
			s.stats.Residuals++
			return true, nil
		}
	}
	if reconstruct.ReconstructPlaneBlockVisibleTrustedAtFromDequantized(dst, stride, bps, s.batch.Sequence.ColorConfig.BitDepth,
		int(w), int(h), meta.scanSize, activeRows, activeCols, dcOnly, cf, s.int32Scratch, s.residualScratch,
		reconstruct.Block{Size: meta.size, Transform: txtp, EOB: int16(eob)}) != nil {
		return true, ErrInvalidBatch
	}
	s.stats.Residuals++
	return true, nil
}

func frameWorkVisitUsesCFLPtr(visit *tile.BlockLoopVisit) bool {
	if visit == nil {
		return false
	}
	return visit.Prediction.Valid &&
		visit.Prediction.Intra &&
		visit.Prediction.ChromaModeValid &&
		visit.Prediction.ChromaMode == tile.ChromaIntraModeCFL
}

// frameWorkReconCBI is dav1d's frame_thread.cbi entry of one TXB of a coded
// block: its eob (zero when the TXB is all zero) and transform type, plus the
// extent of the nonzero coefficients the inverse transform reads. A TXB with a
// nonzero eob owns the next scan area of the cf arena.
type frameWorkReconCBI struct {
	eob        uint16
	txtp       transform.Type
	activeRows uint8
	activeCols uint8
}

// frameWorkReconCursor walks the cbi entries and cf arena of the blocks being
// reconstructed, dav1d's ts->frame_thread[0].cbi / .cf in pass 2.
type frameWorkReconCursor struct {
	cbi    []frameWorkReconCBI
	cf     []int32
	cbiOff int
	cfOff  int
}
