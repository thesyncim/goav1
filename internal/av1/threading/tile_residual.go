package threading

import (
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// frameWorkDeferReconstruction keeps the serial deferred two-pass
// reconstruction harness frozen at its production default. The real wavefront
// path still enables deferred reconstruction when worker lanes are available.
const frameWorkDeferReconstruction = false

// frameWorkWavefrontMinSBRowsPerWorker is the SB-row count, per available
// worker, a tile must clear before the deferred SB-row wavefront engages.
// Below it the fused single-thread path wins (no buffering, no goroutines), so
// the wavefront stays off. It defaults to 2 (the diagonal must fill before it
// drains); focused tests override it through
// OverrideWavefrontMinSBRowsPerWorkerForTest.
var frameWorkWavefrontMinSBRowsPerWorker = 2

// frameWorkReconEventKind distinguishes the two deferred reconstruction events:
// a per-block predict "begin" marker and a per-TXB reconstruction.
type frameWorkReconEventKind uint8

const (
	frameWorkReconEventBlockBegin frameWorkReconEventKind = iota
	frameWorkReconEventTXB
)

// frameWorkReconEvent records one deferred reconstruction step captured during
// pass 1 (entropy decode) and replayed in pass 2 (predict+reconstruct) in
// decode order. blockBegin events index reconVisits so pass 2 runs the predict
// portion of BeforeBlockCoefficients; txb events index reconBlocks, whose
// Coeffs have been deep-copied into the scratch arena.
type frameWorkReconEvent struct {
	index         int32
	kind          frameWorkReconEventKind
	currentQIndex uint8
}

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
)

// frameWorkReconPaletteBinding records which buffered reconstruction event
// should be rebound to which value-arena palette maps once event collection is
// complete. Keeping the binding out of frameWorkReconEvent avoids growing every
// non-palette event for a rare screen-content path.
type frameWorkReconPaletteBinding struct {
	visitIndex   int32
	paletteIndex int32
	mask         uint8
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
		req.Predict == nil && req.AfterBlock == nil && req.Transforms == nil && req.UseDefaultTransforms &&
		req.Loop.BeforeSuperblock == nil && req.Loop.BeforeCoefficients == nil && req.Loop.CoeffVisitor == nil &&
		req.Loop.CoeffController == nil && req.Loop.CoeffRequest == nil && req.Loop.TransformSelect == nil
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
	IntraTX      tile.IntraCoeffTransformSelector
	InterTX      tile.InterCoeffTransformSelector
	CFL          FrameWorkCFLPredictionScratch

	controller       frameWorkTileResidualLoopController
	beforeSuperblock tile.BlockLoopSuperblockVisitor
	stats            FrameWorkTileResidualStats
	geomCache        frameWorkJobGeometryCache

	// lfMask carries the reusable decode-time loop-filter mask build state
	// (dav1d decomp_tx scratch + carried above/left tx-context). It is reset
	// once per job and holds capacity across jobs for zero steady-state allocs.
	lfMask frameWorkLoopFilterMaskScratch

	// reconEvents, reconVisits, reconBlocks, and coeffArena back the deferred
	// two-pass reconstruction path (frameWorkDeferReconstruction). reconEvents
	// records the in-order predict/reconstruct op stream; reconVisits holds the
	// block-loop visits referenced by block-begin events; reconBlocks holds the
	// transform blocks referenced by TXB events; coeffArena holds the deep-copied
	// coefficients each TXB block references plus, for non-zero TXBs, the compact
	// dirty-position tail carried in the buffered coefficient slice capacity.
	// The live decode reuses one scratch coefficient slice per TXB, so buffered
	// coefficients must not alias it.
	// paletteArena holds deep-copied palette colour maps for buffered visits
	// whose prediction is palette-coded: the decoded BlockLoopVisit.Prediction
	// carries YMap/UVMap pointers into a single per-tile PaletteModeScratch that
	// later blocks overwrite, so a buffered visit must own its own copy. Palette
	// events store indexes while the arena is still appendable; pointers into the
	// final value arena are installed immediately before replay. All buffers
	// retain capacity across tiles and are sliced to zero length per job by
	// resetDeferredReconBuffers.
	reconEvents []frameWorkReconEvent
	reconVisits []tile.BlockLoopVisit
	reconBlocks []tile.BlockCoeffBlock
	// reconCF parallels reconBlocks: the dequantized coefficients of a TXB the
	// dav1d block walk decoded into cfArena (dav1d's frame-thread cf), or a
	// nil slice when the TXB carries quantized coefficients instead.
	reconCF []frameWorkReconCF
	cfArena []int32
	// coefBuf is the fused path's dav1d t->cf coefficient buffer.
	coefBuf         tile.CoefBuffer
	coeffArena      []int16
	paletteArena    []tile.PaletteModeScratch
	paletteBindings []frameWorkReconPaletteBinding

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
// wavefront scheduling state. rowStart partitions the flat reconEvents list
// into per-SB-row spans ([rowStart[r], rowStart[r+1])); done[r] counts the SBs
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
		st.predict = c.req.Predict
		st.predictionScratch = &wf.predict[w]
		st.cflScratch = &wf.cfl[w]
		st.int32Scratch = wf.workerInt32(w)
		st.residualScratch = wf.workerResidual(w)
		st.quant = frameWorkReconQuantCache{}
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
	}
}

// resetDeferredReconBuffers slices the deferred reconstruction buffers back to
// zero length while retaining their capacity, so a worker reusing this scratch
// across jobs never replays a previous job's events or coefficients.
func (s *FrameWorkTileResidualScratch) resetDeferredReconBuffers() {
	s.reconEvents = s.reconEvents[:0]
	s.reconVisits = s.reconVisits[:0]
	s.reconBlocks = s.reconBlocks[:0]
	s.reconCF = s.reconCF[:0]
	s.cfArena = s.cfArena[:0]
	s.coeffArena = s.coeffArena[:0]
	s.paletteArena = s.paletteArena[:0]
	s.paletteBindings = s.paletteBindings[:0]
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
func (s *FrameWorkTileResidualScratch) bufferDeferredVisit(visit *tile.BlockLoopVisit) (int, int, uint8) {
	n := len(s.reconVisits)
	if n < cap(s.reconVisits) {
		s.reconVisits = s.reconVisits[:n+1]
		s.reconVisits[n] = *visit
	} else {
		s.reconVisits = append(s.reconVisits, *visit)
	}
	pal := &s.reconVisits[n].Prediction.Palette
	if pal.YMap == nil && pal.UVMap == nil {
		return n, -1, 0
	}
	paletteIndex := s.nextPaletteScratch()
	store := &s.paletteArena[paletteIndex]
	mask := uint8(0)
	if pal.YMap != nil {
		store.Y = *pal.YMap
		pal.YMap = nil
		mask |= frameWorkReconPaletteY
	}
	if pal.UVMap != nil {
		store.UV = *pal.UVMap
		pal.UVMap = nil
		mask |= frameWorkReconPaletteUV
	}
	return n, paletteIndex, mask
}

func (s *FrameWorkTileResidualScratch) rememberDeferredPalette(visitIndex int, paletteIndex int, mask uint8) error {
	if mask == 0 {
		return nil
	}
	visitIndex32, ok := frameWorkReconIndex(visitIndex)
	if !ok {
		return ErrInvalidBatch
	}
	paletteIndex32, ok := frameWorkReconIndex(paletteIndex)
	if !ok {
		return ErrInvalidBatch
	}
	s.paletteBindings = append(s.paletteBindings, frameWorkReconPaletteBinding{
		visitIndex:   visitIndex32,
		paletteIndex: paletteIndex32,
		mask:         mask,
	})
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
		paletteIndex := int(binding.paletteIndex)
		if binding.visitIndex < 0 || visitIndex >= len(s.reconVisits) || binding.paletteIndex < 0 || paletteIndex >= len(s.paletteArena) {
			return i, ErrInvalidBatch
		}
		visit := &s.reconVisits[visitIndex]
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

// FrameWorkBlockTransforms carries the transform policy already determined by
// the block-mode layer. The residual driver only consumes it; it does not guess
// mode-dependent transform types.
type FrameWorkBlockTransforms struct {
	Inter           bool
	Luma            transform.Type
	Chroma          [2]transform.Type
	TransformSelect tile.CoeffTransformSelector
	ReadIntraTX     bool
	ReadInterTX     bool
	EOBMultiContext [3]uint8
}

// FrameWorkBlockTransformSelector returns transform syntax decisions for one
// block-loop visit.
type FrameWorkBlockTransformSelector func(tile.BlockLoopVisit) (FrameWorkBlockTransforms, error)

// FrameWorkBlockPredictor prepares prediction pixels for one block before the
// residual, if any, is added to the output frame.
type FrameWorkBlockPredictor func(tile.BlockLoopVisit) error

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

	Predict           FrameWorkBlockPredictor
	PredictionScratch *FrameWorkPredictionScratch
	CDEFIndexMap      *FrameWorkCDEFIndexMap
	LoopFilterMap     *FrameWorkLoopFilterMap
	LoopFilterMasks   *FrameWorkLoopFilterMasks
	Restoration       *FrameWorkTileRestorationRequest
	// UseDefaultTransforms derives transform decisions from the frame-work
	// block mode state when Transforms is nil.
	UseDefaultTransforms bool
	Transforms           FrameWorkBlockTransformSelector
	AfterBlock           tile.BlockLoopVisitor

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

// ExecuteFrameWorkTileResidualRunner runs residual batches through the concrete
// no-copy runner path. Keeping this concrete lets escape analysis inspect RunPtr
// and keep the per-batch context on the stack.
func ExecuteFrameWorkTileResidualRunner(p *Pool, batches []Batch, jobs []tile.Job, base FrameWorkBatch, runner *FrameWorkTileResidualRunner) error {
	if p == nil || len(p.workers) == 0 {
		return ErrInvalidWorkerCount
	}
	if runner == nil {
		return ErrInvalidCallback
	}
	if len(batches) == 0 {
		return nil
	}
	if len(batches) > len(p.workers) {
		return ErrInvalidBatch
	}
	if err := validateBatches(batches, jobs, len(p.workers)); err != nil {
		return err
	}

	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return ErrPoolClosed
	}
	p.mu.Unlock()

	var firstErr error
	for i := range batches {
		batch := batches[i]
		batchJobs, err := batch.jobSlice(jobs)
		if err != nil {
			return err
		}
		ctx := base
		ctx.Batch = batch
		ctx.Jobs = batchJobs
		if err := runner.RunPtr(&ctx); firstErr == nil && err != nil {
			firstErr = err
		}
	}
	return firstErr
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
	s.Loop.SegmentPredictions += other.Loop.SegmentPredictions
	s.Loop.SegmentIDs += other.Loop.SegmentIDs
	s.Loop.Prefixes += other.Loop.Prefixes
	s.Loop.PredictionModes += other.Loop.PredictionModes
	s.Loop.IntraModes += other.Loop.IntraModes
	s.Loop.InterEntries += other.Loop.InterEntries
	s.Loop.InterReferences += other.Loop.InterReferences
	s.Loop.InterModes += other.Loop.InterModes
	s.Loop.RefMVStacks += other.Loop.RefMVStacks
	s.Loop.DRLIndices += other.Loop.DRLIndices
	s.Loop.InterMVReferences += other.Loop.InterMVReferences
	s.Loop.MotionVectors += other.Loop.MotionVectors
	s.Loop.MVResiduals += other.Loop.MVResiduals
	s.Loop.InterpFilters += other.Loop.InterpFilters
	s.Loop.InterIntras += other.Loop.InterIntras
	s.Loop.MotionModes += other.Loop.MotionModes
	s.Loop.CompoundBlends += other.Loop.CompoundBlends
	s.Loop.CoefficientBlocks += other.Loop.CoefficientBlocks
	s.Loop.CoefficientTXBs += other.Loop.CoefficientTXBs
	s.Loop.CoefficientNonZero += other.Loop.CoefficientNonZero
	s.Loop.CoefficientAllZero += other.Loop.CoefficientAllZero
	s.Loop.CoefficientEOBTotal += other.Loop.CoefficientEOBTotal
	s.Loop.DeltaReads += other.Loop.DeltaReads
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
	if state == nil || scratch == nil || (req.Transforms == nil && !req.UseDefaultTransforms) {
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
	wavefrontWorkers := uint16(0)
	if req.Predict == nil {
		wavefrontWorkers = frameWorkWavefrontLanes(*b, index)
	}
	// A frame-threaded tile always reconstructs through the lanes, even when the
	// tile is too small to fill two of them: its parse must never predict, since
	// the reference pixels may still be in flight.
	streamRecon := wavefrontWorkers > 1
	if b.FrameThread != nil && req.Predict == nil {
		wavefrontWorkers = max(wavefrontWorkers, 1)
		streamRecon = true
	}
	deferReconstruction := frameWorkDeferReconstruction || streamRecon
	if deferReconstruction {
		scratch.reserveCFArena(loopReq.Walk, b.Sequence.ColorConfig)
	}
	batchValue := *b
	scratch.controller = frameWorkTileResidualLoopController{
		batch:                  batchValue,
		index:                  index,
		state:                  state,
		cdfs:                   cdfs,
		scratch:                scratch,
		req:                    req,
		stats:                  &scratch.stats,
		userBeforeSuperblock:   loopReq.BeforeSuperblock,
		userBeforeCoefficients: loopReq.BeforeCoefficients,
		userCoeffVisitor:       loopReq.CoeffVisitor,
		restoration:            restoration,
		readRestoration:        readRestoration,
		deferReconstruction:    deferReconstruction,
		wavefrontWorkers:       wavefrontWorkers,
		defaultCoeffPath: req.Transforms == nil && req.UseDefaultTransforms && loopReq.CoeffVisitor == nil &&
			req.Predict == nil && !b.Quantization.UsingQMatrix,
	}
	// Wire the fused / serial-deferred reconstruction state to the controller's
	// request scratch and shared geometry cache. The fused single-thread path
	// pays no extra copy: it dereferences exactly the buffers the previous
	// inline calls used.
	scratch.controller.recon = frameWorkReconState{
		batch:             batchValue,
		index:             index,
		stats:             &scratch.stats,
		predict:           req.Predict,
		predictionScratch: req.PredictionScratch,
		cflScratch:        &scratch.CFL,
		int32Scratch:      req.Int32Scratch,
		residualScratch:   req.ResidualScratch,
	}
	scratch.controller.recon.bindPrimedLumaReconGeometry()
	scratch.controller.reconHot = &scratch.controller.recon
	if loopReq.BeforeSuperblock != nil || readRestoration {
		if scratch.beforeSuperblock == nil {
			scratch.beforeSuperblock = scratch.controller.BeforeSuperblock
		}
		loopReq.BeforeSuperblock = scratch.beforeSuperblock
	}

	if streamRecon {
		if err := scratch.controller.startReconStream(); err != nil {
			return scratch.stats, err
		}
	}
	visitBlock := func(visit *tile.BlockLoopVisit) error {
		if !visit.CoefficientsValid {
			return ErrInvalidBatch
		}
		frameWorkAccumulateResidualStats(&scratch.stats, visit.Coefficients.TotalStats())
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
	if scratch.controller.deferReconstruction && !streamRecon {
		if err := scratch.finalizeDeferredPalettes(); err != nil {
			return scratch.stats, err
		}
		if err := scratch.controller.replayDeferredReconstruction(); err != nil {
			return scratch.stats, err
		}
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
		req.Predict == nil && req.AfterBlock == nil && req.Transforms == nil && req.UseDefaultTransforms &&
		loop.BeforeSuperblock == nil && loop.BeforeCoefficients == nil && loop.CoeffVisitor == nil &&
		loop.CoeffController == nil && loop.CoeffRequest == nil && loop.TransformSelect == nil
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
		Loop:                 loopReq,
		TransformMode:        b.TransformRef.TransformMode,
		PredictionScratch:    req.PredictionScratch,
		Restoration:          req.Restoration,
		UseDefaultTransforms: true,
		AfterBlock:           req.AfterBlock,
		Int32Scratch:         req.Int32Scratch,
		ResidualScratch:      req.ResidualScratch,
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

	userBeforeCoefficients tile.BlockLoopVisitor
	userCoeffVisitor       tile.BlockLoopCoeffVisitor
	userBeforeSuperblock   tile.BlockLoopSuperblockVisitor
	restoration            FrameWorkTileRestorationRequest
	readRestoration        bool

	// deferReconstruction selects the buffered two-pass path: pass 1 buffers
	// predict/reconstruct events instead of running them inline, and
	// replayDeferredReconstruction runs them after the whole tile's entropy
	// decode completes.
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

	// coeffReq is the storage SelectBlockCoeffRequestPtr returns: the block
	// walk reads the request in place (dav1d's decode_b hands b to
	// recon_b_* by pointer), so it is not copied out and back in per block.
	coeffReq tile.BlockCoeffRequest

	// dequantTarget is a stable per-controller target shared with the tracked
	// coefficient reader. The controller is replaced wholesale for each tile
	// job, so prepared factors cannot leak across frame quantization parameters.
	dequantTarget        tile.CoeffDequantTarget
	dequantKey           uint16
	dequantKeyValid      bool
	dequantCurrentQIndex uint8
	dequantSegmentID     uint8

	// defaultCoeffPath is bound once per tile job: the production
	// configuration (default transform selectors, no user coefficient visitor
	// or predict callback, no quantizer matrices) that dav1d's decode_b /
	// recon_b always run. SelectBlockCoeffRequestPtr then builds the request
	// directly instead of re-testing that configuration for every block.
	defaultCoeffPath bool
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
	predict           FrameWorkBlockPredictor
	predictionScratch *FrameWorkPredictionScratch
	cflScratch        *FrameWorkCFLPredictionScratch
	int32Scratch      []int32
	residualScratch   []int16
	quant             frameWorkReconQuantCache

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

// BeginBlockCoefficients is the block loop's per-block hook (dav1d's
// decode_b handing the block to recon_b): the block's prediction or deferred
// reconstruction bookkeeping, then its coefficient request.
func (c *frameWorkTileResidualLoopController) BeginBlockCoefficients(visit *tile.BlockLoopVisit) (*tile.BlockCoeffRequest, error) {
	if err := c.BeforeBlockCoefficientsPtr(visit); err != nil {
		return nil, err
	}
	return c.SelectBlockCoeffRequestPtr(visit)
}

func (c *frameWorkTileResidualLoopController) BeforeBlockCoefficientsPtr(visit *tile.BlockLoopVisit) error {
	if visit == nil {
		return ErrInvalidBatch
	}
	if c.userBeforeCoefficients != nil {
		if err := c.userBeforeCoefficients(*visit); err != nil {
			return err
		}
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
		if c.streaming {
			if err := c.streamBlockBegin(visit); err != nil {
				return err
			}
		}
		visitIndex, paletteIndex, paletteMask := c.scratch.bufferDeferredVisit(visit)
		visitIndex32, ok := frameWorkReconIndex(visitIndex)
		if !ok {
			return ErrInvalidBatch
		}
		c.scratch.reconEvents = append(c.scratch.reconEvents, frameWorkReconEvent{
			kind:  frameWorkReconEventBlockBegin,
			index: visitIndex32,
		})
		if err := c.scratch.rememberDeferredPalette(visitIndex, paletteIndex, paletteMask); err != nil {
			return err
		}
		return nil
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
			predict:           c.req.Predict,
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

func (s *frameWorkReconState) predictBeforeCoefficients(visit tile.BlockLoopVisit) error {
	return s.predictBeforeCoefficientsPtr(&visit)
}

func (s *frameWorkReconState) predictBeforeCoefficientsPtr(visit *tile.BlockLoopVisit) error {
	if visit == nil {
		return ErrInvalidBatch
	}
	if s.predict != nil {
		if err := s.predict(*visit); err != nil {
			return fmt.Errorf("predict callback block=%+v: %w", visit.Block, err)
		}
		s.stats.Predictions++
		return nil
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
		return fmt.Errorf("predict block=%+v prediction=%+v prefix=%+v: %w", visit.Block, visit.Prediction, visit.Prefix, err)
	}
	s.stats.Predictions++
	return nil
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

func (c *frameWorkTileResidualLoopController) SelectBlockCoeffRequestPtr(visit *tile.BlockLoopVisit) (*tile.BlockCoeffRequest, error) {
	// A previous call can have returned a ready target that was never consumed
	// because its visitor failed. Do not let it leak into this block.
	c.dequantTarget.Ready = false
	c.dequantTarget.ActiveRows = 0
	c.dequantTarget.ActiveCols = 0
	c.dequantTarget.TxScale = 0
	if visit == nil {
		return nil, ErrInvalidBatch
	}
	if c.defaultCoeffPath {
		return c.selectDefaultBlockCoeffRequest(visit)
	}
	var transforms FrameWorkBlockTransforms
	var segmentQIndex uint8
	var lossless bool
	if c.req.Transforms != nil {
		var err error
		transforms, err = c.req.Transforms(*visit)
		if err != nil {
			return nil, err
		}
		// libaom uses xd->qindex[segment_id] (the segment-adjusted qindex) —
		// not the frame base — when gating tx-type syntax on the
		// qindex==0/lossless shortcut. Custom transform selectors do not carry
		// that cached value, so resolve it here.
		segmentQIndex, lossless, err = c.batch.BlockQIndex(c.state.CurrentBaseQIdx, visit.SegmentID)
		if err != nil {
			return nil, err
		}
	} else if c.req.UseDefaultTransforms {
		var err error
		transforms, segmentQIndex, lossless, err = c.batch.readBlockTransformsAndQIndexPtr(c.state, visit)
		if err != nil {
			return nil, err
		}
	} else {
		return nil, ErrInvalidBatch
	}
	transformSelect := transforms.TransformSelect
	if transforms.ReadIntraTX {
		if !visit.Prediction.Valid || !visit.Prediction.Intra {
			return nil, ErrInvalidBatch
		}
		chromaMode := visit.Prediction.ChromaMode
		if !chromaMode.Valid() {
			chromaMode = tile.ChromaIntraModeDC
		}
		lumaMode := frameWorkIntraTransformMode(&visit.Prediction)
		c.scratch.IntraTX.Reset(c.state, c.cdfs.TransformType, c.batch.FrameMode.ReducedTxSet, visit.Prefix.SkipTransform, lossless, segmentQIndex, lumaMode, chromaMode)
		transformSelect = &c.scratch.IntraTX
	} else if transforms.ReadInterTX {
		c.scratch.InterTX.ResetForColor(c.state, c.cdfs.TransformType, c.batch.FrameMode.ReducedTxSet, visit.Prefix.SkipTransform, lossless, segmentQIndex, c.batch.Sequence.ColorConfig)
		transformSelect = &c.scratch.InterTX
	}
	// The request is filled field by field in the controller's storage, as
	// dav1d's decode_b fills t and b: a composite literal is built in a
	// temporary and block-copied, and the coefficient walk's reads of the
	// just-written bytes then miss store forwarding.
	request := &c.coeffReq
	tr := &request.Transform
	block := &visit.Block
	tr.Size = block.Size
	tr.X4, tr.Y4 = block.X4, block.Y4
	tr.VisibleW4, tr.VisibleH4 = block.VisibleW4, block.VisibleH4
	tr.HaveTop, tr.HaveLeft = block.HaveTop, block.HaveLeft
	tr.Color = c.batch.Sequence.ColorConfig
	tr.TransformMode = c.req.TransformMode
	tr.Inter = transforms.Inter
	tr.SkipTransform = visit.Prefix.SkipTransform
	tr.Lossless = lossless
	request.LumaType = transforms.Luma
	request.ChromaType = transforms.Chroma
	request.TransformSelect = transformSelect
	request.EOBMultiContext = transforms.EOBMultiContext
	request.SkipAllZeroCoeffClear = c.userCoeffVisitor == nil
	request.Dequant = c.bindCoeffDequantTarget(visit, lossless)
	return request, nil
}

func (c *frameWorkTileResidualLoopController) bindCoeffDequantTarget(visit *tile.BlockLoopVisit, lossless bool) *tile.CoeffDequantTarget {
	if visit == nil || c.userCoeffVisitor != nil ||
		c.req.Predict != nil || c.req.Transforms != nil || !c.req.UseDefaultTransforms ||
		visit.Prefix.SkipTransform || lossless || c.batch.Quantization.UsingQMatrix {
		return nil
	}
	key := frameWorkReconQuantKey(c.state.CurrentBaseQIdx, visit.SegmentID)
	if !c.dequantKeyValid || c.dequantKey != key {
		c.dequantTarget.Factors = [3][2]int32{}
		c.dequantTarget.PreparedPlanes = 0
		c.dequantKey = key
		c.dequantKeyValid = true
		c.dequantCurrentQIndex = c.state.CurrentBaseQIdx
		c.dequantSegmentID = visit.SegmentID
	}
	recon := c.reconHot
	if recon == nil {
		return nil
	}
	if c.dequantTarget.Provider == nil {
		if c.deferReconstruction {
			// Deferred reconstruction consumes the coefficients after the parse,
			// so they go to the frame-thread cf arena instead of int32Scratch.
			c.dequantTarget.Arena = &c.scratch.cfArena
		} else {
			// dav1d's t->cf: a dedicated buffer kept zero between TXBs.
			c.dequantTarget.Coef = &c.scratch.coefBuf
		}
		// The controller and its reconstruction state are job-owned and stable;
		// this exact slice is consumed by the synchronous reconstruction path.
		c.dequantTarget.BitDepth = c.batch.Sequence.ColorConfig.BitDepth
		c.dequantTarget.Provider = c
	}
	return &c.dequantTarget
}

// PrepareCoeffDequantPlane lazily resolves one plane's dequant factors only
// after the tracked 2D reader has found a nonzero EOB. Returning false leaves
// the reader on its ordinary quantized fallback and does not move quantizer
// errors ahead of the original reconstruction point.
func (c *frameWorkTileResidualLoopController) PrepareCoeffDequantPlane(plane uint8) bool {
	if c == nil || plane > uint8(quantize.PlaneV) || !c.dequantKeyValid {
		return false
	}
	bit := uint8(1 << plane)
	if c.dequantTarget.PreparedPlanes&bit != 0 {
		return true
	}
	recon := c.reconHot
	if recon == nil {
		return false
	}
	q, lossless, err := c.batch.cachedBlockQuantizerTrusted(&recon.quant, c.dequantCurrentQIndex, c.dequantSegmentID, quantize.Plane(plane))
	if err != nil || lossless || q.DC <= 0 || q.AC <= 0 {
		return false
	}
	c.dequantTarget.Factors[plane] = [2]int32{q.DC, q.AC}
	c.dequantTarget.PreparedPlanes |= bit
	return true
}

func (c *frameWorkTileResidualLoopController) selectBlockTransforms(visit tile.BlockLoopVisit) (FrameWorkBlockTransforms, error) {
	return c.selectBlockTransformsPtr(&visit)
}

func (c *frameWorkTileResidualLoopController) selectBlockTransformsPtr(visit *tile.BlockLoopVisit) (FrameWorkBlockTransforms, error) {
	if visit == nil {
		return FrameWorkBlockTransforms{}, ErrInvalidBatch
	}
	if c.req.Transforms != nil {
		return c.req.Transforms(*visit)
	}
	if c.req.UseDefaultTransforms {
		return c.batch.readBlockTransformsPtr(c.state, visit)
	}
	return FrameWorkBlockTransforms{}, ErrInvalidBatch
}

func (c *frameWorkTileResidualLoopController) VisitBlockCoeffPtr(visit *tile.BlockLoopVisit, block *tile.BlockCoeffBlock) error {
	dequantReady := c.dequantTarget.Ready
	activeRows := c.dequantTarget.ActiveRows
	activeCols := c.dequantTarget.ActiveCols
	// The tile target only becomes ready for EOB-1 DCT_DCT when its DC-only
	// constant-add shortcut is valid. The transform type in this callback is
	// the selected per-TXB type, so ADST EOB-1 remains on the full clear path.
	dcOnly := block != nil && block.Transform == transform.TypeDCTDCT && block.Result.EOB == 1
	c.dequantTarget.Ready = false
	c.dequantTarget.ActiveRows = 0
	c.dequantTarget.ActiveCols = 0
	if visit == nil || block == nil {
		return ErrInvalidBatch
	}
	if dequantReady && !c.deferReconstruction && c.userCoeffVisitor == nil && c.req.Predict == nil {
		recon := c.reconHot
		if recon == nil {
			recon = c.fusedReconState()
		}
		if err := recon.reconstructTXBWithDequantized(visit, block, c.dequantTarget.Dst, activeRows, activeCols, dcOnly); err != nil {
			return err
		}
	} else if c.deferReconstruction {
		var cf frameWorkReconCF
		if dequantReady {
			cf = frameWorkReconCF{cf: c.dequantTarget.Dst, activeRows: int32(activeRows), activeCols: int32(activeCols), dcOnly: dcOnly}
		}
		if err := c.bufferReconTXB(visit, block, cf, c.state.CurrentBaseQIdx); err != nil {
			return err
		}
	} else {
		recon := c.reconHot
		if recon == nil {
			recon = c.fusedReconState()
		}
		// BlockLoopScratch owns the coefficient decode scratch used by DecodeBlockResidualInto.
		coeffScratch := &c.scratch.Loop.Coeff
		dirty := coeffScratch.CoeffDirtyPositions()
		if err := recon.reconstructTXBWithNonZero(visit, block, dirty, c.state.CurrentBaseQIdx); err != nil {
			return err
		}
	}
	if c.userCoeffVisitor != nil {
		return c.userCoeffVisitor(*visit, *block)
	}
	return nil
}

// bufferReconTXB appends a deferred TXB reconstruction event, deep-copying the
// decoded coefficients into the scratch arena. The live decode reuses one
// coefficient scratch slice per transform block (block.Coeffs aliases it), so
// the buffered slice is repointed into the arena with a three-index slice that
// caps its capacity, preventing a later append from clobbering a neighbor
// event's coefficients. Scan is kept by reference to the immutable scan table
// so deferred reconstruction can use the same sparse dequant path as the fused
// path without copying per-TXB scan data.
func (c *frameWorkTileResidualLoopController) bufferReconTXB(visit *tile.BlockLoopVisit, block *tile.BlockCoeffBlock, cf frameWorkReconCF, currentQIndex uint8) error {
	blockIndex := len(c.scratch.reconBlocks)
	blockIndex32, ok := frameWorkReconIndex(blockIndex)
	if !ok {
		return ErrInvalidBatch
	}
	buffered := *block
	if block.Result.AllZero || cf.cf != nil {
		buffered.Coeffs = nil
	} else if n := len(block.Coeffs); n > 0 {
		dirty := c.scratch.Loop.Coeff.CoeffDirtyPositions()
		dirtyLen := len(dirty)
		if dirtyLen > int(^uint16(0)) {
			return ErrInvalidBatch
		}
		off := len(c.scratch.coeffArena)
		c.scratch.coeffArena = append(c.scratch.coeffArena, block.Coeffs...)
		c.scratch.coeffArena = append(c.scratch.coeffArena, dirty...)
		buffered.Coeffs = c.scratch.coeffArena[off : off+n : off+n+dirtyLen]
	} else {
		buffered.Coeffs = nil
	}
	c.scratch.reconBlocks = append(c.scratch.reconBlocks, buffered)
	c.scratch.reconCF = append(c.scratch.reconCF, cf)
	c.scratch.reconEvents = append(c.scratch.reconEvents, frameWorkReconEvent{
		kind:          frameWorkReconEventTXB,
		index:         blockIndex32,
		currentQIndex: currentQIndex,
	})
	return nil
}

// replayDeferredReconstruction runs the buffered predict and reconstruct events
// in decode order after the whole tile's entropy decode has completed. Because
// the events are replayed in the same order the fused path interleaved them,
// the CfL luma-first state machine and intra neighbor reads observe identical
// inputs, producing byte-identical output.
func (c *frameWorkTileResidualLoopController) replayDeferredReconstruction() error {
	return frameWorkReplayReconEvents(&c.recon, c.scratch.reconEvents, c.scratch.reconVisits, c.scratch.reconBlocks, c.scratch.reconCF)
}

// frameWorkReplayReconEvents runs a span of buffered reconstruction events in
// decode order on one reconstruction state. It is the shared inner loop of both
// the serial deferred replay and each wavefront goroutine's per-SB-row work.
func frameWorkReplayReconEvents(s *frameWorkReconState, events []frameWorkReconEvent, visits []tile.BlockLoopVisit, blocks []tile.BlockCoeffBlock, cfs []frameWorkReconCF) error {
	var visit *tile.BlockLoopVisit
	for i := range events {
		ev := &events[i]
		switch ev.kind {
		case frameWorkReconEventBlockBegin:
			idx := int(ev.index)
			if ev.index < 0 || idx >= len(visits) {
				return ErrInvalidBatch
			}
			visit = &visits[idx]
			if err := s.predictBlockBegin(visit); err != nil {
				return err
			}
		case frameWorkReconEventTXB:
			idx := int(ev.index)
			if visit == nil || ev.index < 0 || idx >= len(blocks) {
				return ErrInvalidBatch
			}
			if idx < len(cfs) && cfs[idx].cf != nil {
				cf := &cfs[idx]
				if err := s.reconstructTXBWithDequantized(visit, &blocks[idx], cf.cf, int(cf.activeRows), int(cf.activeCols), cf.dcOnly); err != nil {
					return err
				}
				continue
			}
			dirty := bufferedCoeffDirtyPositions(&blocks[idx])
			if err := s.reconstructTXBWithNonZero(visit, &blocks[idx], dirty, ev.currentQIndex); err != nil {
				return err
			}
		default:
			return ErrInvalidBatch
		}
	}
	return nil
}

func bufferedCoeffDirtyPositions(block *tile.BlockCoeffBlock) []int16 {
	// Deferred reconstruction stores the copied dirty-position list directly
	// after the coefficient payload and keeps Coeffs length at the real
	// coefficient count. The spare capacity is an internal side channel, not a
	// public slice growth promise.
	if block == nil || len(block.Coeffs) == cap(block.Coeffs) {
		return nil
	}
	n := len(block.Coeffs)
	return block.Coeffs[n:cap(block.Coeffs):cap(block.Coeffs)]
}

// frameWorkReconSB records one superblock's span of buffered reconstruction
// events: the half-open [start, end) range into the flat reconEvents list and
// the SB column within its row. Events for one SB are contiguous because the
// block loop decodes superblocks in raster order.
type frameWorkReconSB struct {
	start int32
	end   int32
	col   int32
}

// replayDeferredReconstructionWavefront replays the buffered reconstruction
// events as an SB-row wavefront across wavefrontWorkers goroutines. Worker w
// statically owns SB rows w, w+W, w+2W, ... and reconstructs each owned row's
// superblocks left-to-right. Before reconstructing SB column C of row R it
// waits until done[R-1] >= C+2 so the upper and upper-right neighbor
// superblocks (R-1,C) and (R-1,C+1) — the furthest an intra top-right edge
// read reaches — are fully reconstructed; after finishing SB column C it
// publishes done[R] = C+1. Row 0 is ungated. Each goroutine uses its own
// reconstruction state (CFL machine, prediction/residual scratch, geometry
// cache), and the buffered events/arenas are read-only during the replay, so
// the only cross-goroutine communication is the atomic done counters. The
// decode path streams the same wavefront while the tile parses
// (startReconStream); this after-parse form replays a fully buffered tile.
func (c *frameWorkTileResidualLoopController) replayDeferredReconstructionWavefront() error {
	region, err := c.batch.JobRegion(c.index)
	if err != nil {
		return err
	}
	sbSizeMIB := uint32(c.batch.Sequence.SBSizeMIB)
	if sbSizeMIB == 0 {
		return ErrInvalidBatch
	}
	rowCount := int(region.SBRows)
	if rowCount <= 0 {
		return ErrInvalidBatch
	}
	workers := min(int(c.wavefrontWorkers), rowCount)

	if c.scratch.wavefront == nil {
		c.scratch.wavefront = &frameWorkReconWavefront{}
	}
	wf := c.scratch.wavefront
	// Partition the flat event list into per-SB spans and SB rows; a new SB
	// begins at a blockBegin event whose (sbRow, sbCol) differs from the
	// previous block's.
	wf.beginSpans(region, int(sbSizeMIB))
	for i := range c.scratch.reconEvents {
		ev := &c.scratch.reconEvents[i]
		if ev.kind != frameWorkReconEventBlockBegin {
			continue
		}
		idx := int(ev.index)
		if ev.index < 0 || idx >= len(c.scratch.reconVisits) {
			return ErrInvalidBatch
		}
		visit := &c.scratch.reconVisits[idx]
		if _, err := wf.noteBlockBegin(i, int(visit.Block.MIRow), int(visit.Block.MICol)); err != nil {
			return err
		}
	}
	if err := wf.closeSpans(len(c.scratch.reconEvents)); err != nil {
		return err
	}
	wf.ensureDone(rowCount)
	wf.resetProgress(rowCount)
	if err := wf.ensureStates(workers, c); err != nil {
		return err
	}
	wf.startLanes(c.batch.wavefrontLanes, workers, rowCount)
	wf.publishRows(rowCount, frameWorkReconView{
		rowStart: wf.rowStart,
		sbs:      wf.sbSpans,
		events:   c.scratch.reconEvents,
		visits:   c.scratch.reconVisits,
		blocks:   c.scratch.reconBlocks,
		cf:       c.scratch.reconCF,
	})
	return wf.finishLanes(c.stats)
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
	eventCount, ok := frameWorkReconIndex(len(view.events))
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
		if sb.start < 0 || sb.end < sb.start || sb.end > eventCount {
			return ErrInvalidBatch
		}
		if ft := wf.frameThread; ft != nil && !ft.waitSuperblockReferences(&st.batch, view.events[sb.start:sb.end], view.visits) {
			return ErrFrameThreadReference
		}
		if err := frameWorkReplayReconEvents(st, view.events[sb.start:sb.end], view.visits, view.blocks, view.cf); err != nil {
			return err
		}
		wf.publishRowProgress(doneCur, sb.col+1)
	}
	return nil
}

// reconstructTXB performs the predict-then-reconstruct work for one decoded
// transform block. It is the single reconstruction seam shared by the inline
// single-thread path (VisitBlockCoeff, which calls it immediately) and the
// deferred parallel reconstruction path. visit and block are taken by pointer
// so the single-thread path pays no extra struct copy; the values dereferenced
// into the predict/reconstruct callees match the previous inline calls exactly.
// currentQIndex is snapshotted by the caller because the live delta-q state
// advances past this block before a deferred pass would run.
func (s *frameWorkReconState) reconstructTXB(visit *tile.BlockLoopVisit, block *tile.BlockCoeffBlock, currentQIndex uint8) error {
	return s.reconstructTXBWithNonZero(visit, block, nil, currentQIndex)
}

func (s *frameWorkReconState) reconstructTXBWithNonZero(visit *tile.BlockLoopVisit, block *tile.BlockCoeffBlock, nonzero []int16, currentQIndex uint8) error {
	if s.predict == nil && s.predictionScratch != nil && visit.Prediction.Valid && visit.Prediction.Intra {
		if block.Plane != 0 && frameWorkVisitUsesCFLPtr(visit) {
			if err := s.predictDeferredCFLChroma(); err != nil {
				return err
			}
		} else {
			if err := s.batch.predictBlockIntraCoeffPtr(s.index, visit, block, &s.predictionScratch.Intra); err != nil {
				return fmt.Errorf("predict intra txb plane=%d visit=%+v block=%+v luma_mode=%d angle_delta=%d: %w", block.Plane, visit.Block, block.Block, visit.Prediction.LumaMode, visit.Prediction.LumaAngleDelta, err)
			}
			s.stats.Predictions++
		}
	} else if block.Plane != 0 {
		if err := s.predictDeferredCFLChroma(); err != nil {
			return err
		}
	}
	if block.Result.AllZero {
		s.stats.Residuals++
		return nil
	}
	// Hand the live decode block straight to the by-pointer reconstruction core
	// instead of materializing a FrameWorkBlockCoeffReconstruction per TXB (which
	// would deep-copy the 120-byte BlockCoeffBlock). Byte-identical: the core
	// reads exactly the fields the struct path forwarded.
	if block.Plane == 0 && s.lumaReady {
		c := s.batch.geomCache
		geom, err := s.batch.blockCoeffGeometryLumaKnown(c.region, c.plane[FrameWorkPlaneY], visit.Block, block)
		if err != nil {
			return fmt.Errorf("reconstruct plane=%d block=%+v tx=%d: %w", block.Plane, block.Block, block.Transform, err)
		}
		if err := s.batch.reconstructBlockCoeffCoreWithGeometryAndNonZero(geom, block, nonzero, block.Transform, currentQIndex, visit.SegmentID, s.int32Scratch, s.residualScratch, &s.quant); err != nil {
			return fmt.Errorf("reconstruct plane=%d block=%+v tx=%d: %w", block.Plane, block.Block, block.Transform, err)
		}
		s.stats.Residuals++
		return nil
	}
	if block.Plane != 0 && s.chromaReady {
		plane := FrameWorkPlaneU
		if block.Plane == 2 {
			plane = FrameWorkPlaneV
		} else if block.Plane != 1 {
			return fmt.Errorf("reconstruct plane=%d block=%+v tx=%d: %w", block.Plane, block.Block, block.Transform, ErrInvalidBatch)
		}
		c := s.batch.geomCache
		geom, err := s.batch.blockCoeffGeometryChromaKnown(c.region, c.plane[plane], plane, visit.Block, block)
		if err != nil {
			return fmt.Errorf("reconstruct plane=%d block=%+v tx=%d: %w", block.Plane, block.Block, block.Transform, err)
		}
		if err := s.batch.reconstructBlockCoeffCoreWithGeometryAndNonZero(geom, block, nonzero, block.Transform, currentQIndex, visit.SegmentID, s.int32Scratch, s.residualScratch, &s.quant); err != nil {
			return fmt.Errorf("reconstruct plane=%d block=%+v tx=%d: %w", block.Plane, block.Block, block.Transform, err)
		}
		s.stats.Residuals++
		return nil
	}
	if ok, err := s.batch.reconstructBlockCoeffLumaPrimed(s.index, visit.Block, block, nonzero, block.Transform, currentQIndex, visit.SegmentID, s.int32Scratch, s.residualScratch, &s.quant); err != nil {
		return fmt.Errorf("reconstruct plane=%d block=%+v tx=%d: %w", block.Plane, block.Block, block.Transform, err)
	} else if ok {
		s.stats.Residuals++
		return nil
	}
	if err := s.batch.reconstructBlockCoeffCoreTrustedWithNonZero(s.index, visit.Block, block, nonzero, block.Transform, currentQIndex, visit.SegmentID, s.int32Scratch, s.residualScratch, &s.quant); err != nil {
		return fmt.Errorf("reconstruct plane=%d block=%+v tx=%d: %w", block.Plane, block.Block, block.Transform, err)
	}
	s.stats.Residuals++
	return nil
}

// reconstructTXBWithDequantized is used only by the synchronous, private
// tracked-2D route. It deliberately keeps the existing quantized path intact;
// the predictor/CfL ordering is mirrored here and the narrow reconstruction
// tail consumes the dequantized scratch before the next TXB reuses it.
func (s *frameWorkReconState) reconstructTXBWithDequantized(visit *tile.BlockLoopVisit, block *tile.BlockCoeffBlock, cf []int32, activeRows int, activeCols int, dcOnly bool) error {
	if s.predict == nil && s.predictionScratch != nil && visit.Prediction.Valid && visit.Prediction.Intra {
		if block.Plane != 0 && frameWorkVisitUsesCFLPtr(visit) {
			if err := s.predictDeferredCFLChroma(); err != nil {
				return err
			}
		} else {
			if err := s.batch.predictBlockIntraCoeffPtr(s.index, visit, block, &s.predictionScratch.Intra); err != nil {
				return fmt.Errorf("predict intra txb plane=%d visit=%+v block=%+v luma_mode=%d angle_delta=%d: %w", block.Plane, visit.Block, block.Block, visit.Prediction.LumaMode, visit.Prediction.LumaAngleDelta, err)
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

	if done, err := s.reconstructTXBBound(visit, block, cf, activeRows, activeCols, dcOnly); done {
		if err == nil {
			s.stats.Residuals++
		}
		return err
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
			return fmt.Errorf("reconstruct plane=%d block=%+v tx=%d: %w", block.Plane, block.Block, block.Transform, ErrInvalidBatch)
		}
		c := s.batch.geomCache
		geom, err = s.batch.blockCoeffGeometryChromaKnown(c.region, c.plane[plane], plane, visit.Block, block)
	} else {
		geom, err = s.batch.blockCoeffGeometry(s.index, visit.Block, block)
	}
	if err != nil {
		return fmt.Errorf("reconstruct plane=%d block=%+v tx=%d: %w", block.Plane, block.Block, block.Transform, err)
	}
	if err := s.batch.reconstructBlockCoeffCoreWithGeometryDequantized(geom, block, activeRows, activeCols, dcOnly, cf, s.int32Scratch, s.residualScratch); err != nil {
		return fmt.Errorf("reconstruct plane=%d block=%+v tx=%d: %w", block.Plane, block.Block, block.Transform, err)
	}
	s.stats.Residuals++
	return nil
}

func frameWorkIntraTransformMode(pred *tile.BlockPredictionModeResult) tile.IntraMode {
	if !pred.FilterIntraValid {
		return pred.LumaMode
	}
	switch pred.FilterIntraMode {
	case tile.FilterIntraModeVertical:
		return tile.IntraModeVertical
	case tile.FilterIntraModeHorizontal:
		return tile.IntraModeHorizontal
	case tile.FilterIntraModeD157:
		return tile.IntraModeD157
	default:
		return tile.IntraModeDC
	}
}

func frameWorkVisitUsesCFL(visit tile.BlockLoopVisit) bool {
	return frameWorkVisitUsesCFLPtr(&visit)
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

func (b *FrameWorkBatch) ReadBlockTransforms(state *tile.DecodeState, visit tile.BlockLoopVisit) (FrameWorkBlockTransforms, error) {
	return b.readBlockTransformsPtr(state, &visit)
}

func (b *FrameWorkBatch) readBlockTransformsPtr(state *tile.DecodeState, visit *tile.BlockLoopVisit) (FrameWorkBlockTransforms, error) {
	transforms, _, _, err := b.readBlockTransformsAndQIndexPtr(state, visit)
	return transforms, err
}

func (b *FrameWorkBatch) readBlockTransformsAndQIndexPtr(state *tile.DecodeState, visit *tile.BlockLoopVisit) (FrameWorkBlockTransforms, uint8, bool, error) {
	if visit == nil {
		return FrameWorkBlockTransforms{}, 0, false, ErrInvalidBatch
	}
	if visit.Prediction.Valid && visit.Prediction.Intra {
		return b.readIntraBlockTransformsAndQIndexPtr(state, visit)
	}
	return b.readInterBlockTransformsAndQIndexPtr(state, visit)
}

func (b *FrameWorkBatch) ReadIntraBlockTransforms(state *tile.DecodeState, visit tile.BlockLoopVisit) (FrameWorkBlockTransforms, error) {
	return b.readIntraBlockTransformsPtr(state, &visit)
}

func (b *FrameWorkBatch) readIntraBlockTransformsPtr(state *tile.DecodeState, visit *tile.BlockLoopVisit) (FrameWorkBlockTransforms, error) {
	transforms, _, _, err := b.readIntraBlockTransformsAndQIndexPtr(state, visit)
	return transforms, err
}

func (b *FrameWorkBatch) readIntraBlockTransformsAndQIndexPtr(state *tile.DecodeState, visit *tile.BlockLoopVisit) (FrameWorkBlockTransforms, uint8, bool, error) {
	if state == nil || visit == nil || !visit.Prediction.Valid || !visit.Prediction.Intra {
		return FrameWorkBlockTransforms{}, 0, false, ErrInvalidBatch
	}
	qIndex, lossless, err := b.BlockQIndex(state.CurrentBaseQIdx, visit.SegmentID)
	if err != nil {
		return FrameWorkBlockTransforms{}, 0, false, err
	}
	return FrameWorkBlockTransforms{
		Inter:       false,
		Luma:        transform.TypeDCTDCT,
		Chroma:      [2]transform.Type{transform.TypeDCTDCT, transform.TypeDCTDCT},
		ReadIntraTX: true,
	}, qIndex, lossless, nil
}

func (b *FrameWorkBatch) ReadInterBlockTransforms(state *tile.DecodeState, visit tile.BlockLoopVisit) (FrameWorkBlockTransforms, error) {
	return b.readInterBlockTransformsPtr(state, &visit)
}

func (b *FrameWorkBatch) readInterBlockTransformsPtr(state *tile.DecodeState, visit *tile.BlockLoopVisit) (FrameWorkBlockTransforms, error) {
	transforms, _, _, err := b.readInterBlockTransformsAndQIndexPtr(state, visit)
	return transforms, err
}

func (b *FrameWorkBatch) readInterBlockTransformsAndQIndexPtr(state *tile.DecodeState, visit *tile.BlockLoopVisit) (FrameWorkBlockTransforms, uint8, bool, error) {
	if state == nil || visit == nil {
		return FrameWorkBlockTransforms{}, 0, false, ErrInvalidBatch
	}
	qIndex, lossless, err := b.BlockQIndex(state.CurrentBaseQIdx, visit.SegmentID)
	if err != nil {
		return FrameWorkBlockTransforms{}, 0, false, err
	}
	return FrameWorkBlockTransforms{
		Inter:       true,
		Luma:        transform.TypeDCTDCT,
		Chroma:      [2]transform.Type{transform.TypeDCTDCT, transform.TypeDCTDCT},
		ReadInterTX: true,
	}, qIndex, lossless, nil
}

func frameWorkAccumulateResidualStats(stats *FrameWorkTileResidualStats, coeff tile.LumaCoeffStats) {
	stats.TXBs += int32(coeff.TXBs)
	stats.NonZero += int32(coeff.NonZero)
	stats.AllZero += int32(coeff.AllZero)
	stats.EOBTotal += int32(coeff.EOBTotal)
}

// frameWorkReconCF is one deferred TXB's slot in the frame-thread cf arena
// (dav1d ts->frame_thread[p].cf) with what itxfm_add needs to consume it.
type frameWorkReconCF struct {
	cf         []int32
	activeRows int32
	activeCols int32
	dcOnly     bool
}

// selectDefaultBlockCoeffRequest is SelectBlockCoeffRequestPtr for the
// job-bound production configuration (defaultCoeffPath): the block's
// transform-type selector and coefficient request are built straight from the
// visit, as dav1d's decode_b hands b to recon_b_*, with the same
// segment-adjusted qindex and lossless gating as the generic path.
func (c *frameWorkTileResidualLoopController) selectDefaultBlockCoeffRequest(visit *tile.BlockLoopVisit) (*tile.BlockCoeffRequest, error) {
	segmentQIndex, lossless, err := c.batch.BlockQIndex(c.state.CurrentBaseQIdx, visit.SegmentID)
	if err != nil {
		return nil, err
	}
	pred := &visit.Prediction
	skip := visit.Prefix.SkipTransform
	intra := pred.Valid && pred.Intra
	var transformSelect tile.CoeffTransformSelector
	if intra {
		chromaMode := pred.ChromaMode
		if !chromaMode.Valid() {
			chromaMode = tile.ChromaIntraModeDC
		}
		c.scratch.IntraTX.Reset(c.state, c.cdfs.TransformType, c.batch.FrameMode.ReducedTxSet, skip, lossless, segmentQIndex, frameWorkIntraTransformMode(pred), chromaMode)
		transformSelect = &c.scratch.IntraTX
	} else {
		c.scratch.InterTX.ResetForColor(c.state, c.cdfs.TransformType, c.batch.FrameMode.ReducedTxSet, skip, lossless, segmentQIndex, c.batch.Sequence.ColorConfig)
		transformSelect = &c.scratch.InterTX
	}
	// Filled field by field in place; see SelectBlockCoeffRequestPtr.
	request := &c.coeffReq
	tr := &request.Transform
	block := &visit.Block
	tr.Size = block.Size
	tr.X4, tr.Y4 = block.X4, block.Y4
	tr.VisibleW4, tr.VisibleH4 = block.VisibleW4, block.VisibleH4
	tr.HaveTop, tr.HaveLeft = block.HaveTop, block.HaveLeft
	tr.Color = c.batch.Sequence.ColorConfig
	tr.TransformMode = c.req.TransformMode
	tr.Inter = !intra
	tr.SkipTransform = skip
	tr.Lossless = lossless
	request.LumaType = transform.TypeDCTDCT
	request.ChromaType = [2]transform.Type{transform.TypeDCTDCT, transform.TypeDCTDCT}
	request.TransformSelect = transformSelect
	request.EOBMultiContext = [3]uint8{}
	request.SkipAllZeroCoeffClear = true
	request.Dequant = nil
	if !skip && !lossless {
		request.Dequant = c.bindCoeffDequantTarget(visit, lossless)
	}
	return request, nil
}
