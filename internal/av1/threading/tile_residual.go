package threading

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// frameWorkDeferReconstruction gates a deferred two-pass reconstruction path
// that splits entropy decode (pass 1) from predict+reconstruct (pass 2). It is
// read once at package init and serves only as a test harness to prove the
// deferred path is byte-identical to the fused single-thread path before the
// wavefront concurrency increment wires real selection. Default off: when off,
// DecodeAndReconstructJobResiduals runs the existing fused path verbatim.
var frameWorkDeferReconstruction = os.Getenv("GOAV1_DEFER_RECON") != ""

// frameWorkWavefrontMinSBRowsPerWorker is the SB-row count, per available
// worker, a tile must clear before the deferred SB-row wavefront engages.
// Below it the fused single-thread path wins (no buffering, no goroutines), so
// the wavefront stays off. It defaults to 2 (the diagonal must fill before it
// drains) and is overridable only as a test hook so focused tests can force the
// wavefront on the small conformance clips.
var frameWorkWavefrontMinSBRowsPerWorker = frameWorkParseMinSBRows()

func frameWorkParseMinSBRows() int {
	if v := os.Getenv("GOAV1_WAVEFRONT_MIN_SBROWS_PER_WORKER"); v != "" {
		n := 0
		for _, ch := range v {
			if ch < '0' || ch > '9' {
				return 2
			}
			n = n*10 + int(ch-'0')
		}
		if n >= 1 {
			return n
		}
	}
	return 2
}

// frameWorkReconEventKind distinguishes the two deferred reconstruction events:
// a per-block predict "begin" marker and a per-TXB reconstruction.
type frameWorkReconEventKind uint8

const (
	frameWorkReconEventBlockBegin frameWorkReconEventKind = iota
	frameWorkReconEventTXB
)

// frameWorkReconEvent records one deferred reconstruction step captured during
// pass 1 (entropy decode) and replayed in pass 2 (predict+reconstruct) in
// decode order. blockBegin events carry the block-loop visit so pass 2 runs the
// predict portion of BeforeBlockCoefficients; txb events carry a decoded
// transform block whose Coeffs have been deep-copied into the scratch arena.
type frameWorkReconEvent struct {
	kind          frameWorkReconEventKind
	visit         tile.BlockLoopVisit
	block         tile.BlockCoeffBlock
	currentQIndex uint8
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
	var next FrameWorkTileResidualCDFStorage
	if err := next.Partition.InitDefault(); err != nil {
		return err
	}
	if err := next.Mode.InitDefault(); err != nil {
		return err
	}
	if err := next.Intra.InitDefault(); err != nil {
		return err
	}
	if err := next.InterRef.InitDefault(); err != nil {
		return err
	}
	if err := next.InterMode.InitDefault(); err != nil {
		return err
	}
	if err := next.MV.InitDefault(); err != nil {
		return err
	}
	if err := next.Interp.InitDefault(); err != nil {
		return err
	}
	if err := next.Motion.InitDefault(); err != nil {
		return err
	}
	if err := next.Blend.InitDefault(); err != nil {
		return err
	}
	if err := next.Transform.InitDefault(); err != nil {
		return err
	}
	if err := next.TransformType.InitDefault(); err != nil {
		return err
	}
	if err := next.Coeff.InitDefault(baseQIndex); err != nil {
		return err
	}
	if err := next.DeltaQ.InitDefaultDelta(); err != nil {
		return err
	}
	if err := next.DeltaLF.InitDefaultDelta(); err != nil {
		return err
	}
	for i := range tile.FrameLoopFilterCount {
		if err := next.DeltaLFMulti[i].InitDefaultDelta(); err != nil {
			return err
		}
	}
	if err := tile.InitDefaultRestorationCDFs(tile.RestorationCDFs{
		Switchable: &next.RestorationSwitchable,
		Wiener:     &next.RestorationWiener,
		SGRProj:    &next.RestorationSGRProj,
	}); err != nil {
		return err
	}
	*s = next
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
		*storage = *b.InitialTileResidualCDFs
		return nil
	}
	return storage.InitDefault(b.Quantization.BaseQIdx)
}

// RetainTileResidualCDFStorage publishes storage as the adapted frame context
// only for the context_update_tile_id job when CDF updates are enabled.
func (b *FrameWorkBatch) RetainTileResidualCDFStorage(index int, state *tile.DecodeState, storage *FrameWorkTileResidualCDFStorage) error {
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
	if err := storage.ResetCDFCounts(); err != nil {
		return err
	}
	*b.RetainedTileResidualCDFs = *storage
	*b.RetainedTileResidualCDFsValid = true
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

	// reconEvents and coeffArena back the deferred two-pass reconstruction path
	// (frameWorkDeferReconstruction). reconEvents records the in-order predict
	// and reconstruct steps captured during entropy decode; coeffArena holds the
	// deep-copied coefficients each TXB event references (the live decode reuses
	// one scratch coefficient slice per TXB, so the buffered slice must not alias
	// it). paletteArena holds deep-copied palette colour maps for buffered visits
	// whose prediction is palette-coded: the decoded BlockLoopVisit.Prediction
	// carries YMap/UVMap pointers into a single per-tile PaletteModeScratch that
	// later blocks overwrite, so a buffered visit must own its own copy. All
	// three retain capacity across tiles and are sliced to zero length per job by
	// resetDeferredReconBuffers.
	reconEvents  []frameWorkReconEvent
	coeffArena   []int16
	paletteArena []*tile.PaletteModeScratch

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
	rowStart       []int
	sbSpans        []frameWorkReconSB
	events         []frameWorkReconEvent
	done           []atomic.Int32
	states         []frameWorkReconState
	predict        []FrameWorkPredictionScratch
	inter          []FrameWorkInterPredictionScratch
	cfl            []FrameWorkCFLPredictionScratch
	int32Arena     []int32
	residualArena  []int16
	int32Stride    int
	residualStride int
	doneAlloc      []atomic.Int32
	aborted        atomic.Bool
}

func (wf *frameWorkReconWavefront) reuseSBs() []frameWorkReconSB {
	return wf.sbSpans[:0]
}

func (wf *frameWorkReconWavefront) commitSBs(sbs []frameWorkReconSB) {
	wf.sbSpans = sbs
}

func (wf *frameWorkReconWavefront) sbs() []frameWorkReconSB {
	return wf.sbSpans
}

func (wf *frameWorkReconWavefront) eventSpan(start int, end int) []frameWorkReconEvent {
	return wf.events[start:end]
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

// ensureStates lazily allocates wavefrontWorkers per-goroutine reconstruction
// states with their own prediction/inter/CFL/residual scratch and a private
// geometry cache, retaining them across frames. Each state is rebound to the
// current job's batch, index, scratch, and request every frame; only the heap
// scratch is reused.
func (wf *frameWorkReconWavefront) ensureStates(workers int, c *frameWorkTileResidualLoopController) error {
	int32Len := len(c.req.Int32Scratch)
	residualLen := len(c.req.ResidualScratch)
	if cap(wf.states) < workers {
		wf.states = make([]frameWorkReconState, workers)
		wf.predict = make([]FrameWorkPredictionScratch, workers)
		wf.inter = make([]FrameWorkInterPredictionScratch, workers)
		wf.cfl = make([]FrameWorkCFLPredictionScratch, workers)
	}
	wf.states = wf.states[:workers]
	wf.int32Stride = int32Len
	wf.residualStride = residualLen
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
		st.geom.reset()
		st.batch.geomCache = &st.geom
		st.index = c.index
		st.localStats = FrameWorkTileResidualStats{}
		st.stats = &st.localStats
		st.predict = c.req.Predict
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
	off := worker * wf.int32Stride
	return wf.int32Arena[off : off+wf.int32Stride]
}

func (wf *frameWorkReconWavefront) workerResidual(worker int) []int16 {
	off := worker * wf.residualStride
	return wf.residualArena[off : off+wf.residualStride]
}

// resetDeferredReconBuffers slices the deferred reconstruction buffers back to
// zero length while retaining their capacity, so a worker reusing this scratch
// across jobs never replays a previous job's events or coefficients.
func (s *FrameWorkTileResidualScratch) resetDeferredReconBuffers() {
	s.reconEvents = s.reconEvents[:0]
	s.coeffArena = s.coeffArena[:0]
	// Retain the previously allocated palette scratch backings (they live in the
	// slice's capacity beyond the reset length) so a worker reusing this scratch
	// across jobs allocates a palette backing at most once per arena slot.
	s.paletteArena = s.paletteArena[:0]
}

// nextPaletteScratch returns a per-block palette map backing from the arena,
// reusing a previously allocated entry from the slice's retained capacity when
// available and allocating a fresh one otherwise. The returned pointer is
// stable for the lifetime of the buffered events because the arena stores
// pointers (slice growth never moves the pointed-to backings).
func (s *FrameWorkTileResidualScratch) nextPaletteScratch() *tile.PaletteModeScratch {
	n := len(s.paletteArena)
	if n < cap(s.paletteArena) {
		s.paletteArena = s.paletteArena[:n+1]
		if s.paletteArena[n] == nil {
			s.paletteArena[n] = &tile.PaletteModeScratch{}
		}
		return s.paletteArena[n]
	}
	store := &tile.PaletteModeScratch{}
	s.paletteArena = append(s.paletteArena, store)
	return store
}

// captureDeferredVisit returns a copy of visit safe to buffer for deferred
// replay. The decoded visit's palette colour maps (YMap/UVMap) point into a
// single per-tile PaletteModeScratch that subsequent palette-coded blocks
// overwrite during pass 1; the fused path reads them immediately, but the
// deferred path replays prediction after the whole tile has decoded, by which
// time the shared scratch holds a later block's map. Deep-copy the referenced
// map(s) into the scratch palette arena and repoint the buffered visit so pass
// 2 reads this block's own indices. Non-palette visits (the overwhelming
// majority) are returned unchanged and pay no copy.
func (s *FrameWorkTileResidualScratch) captureDeferredVisit(visit tile.BlockLoopVisit) tile.BlockLoopVisit {
	pal := &visit.Prediction.Palette
	if pal.YMap == nil && pal.UVMap == nil {
		return visit
	}
	store := s.nextPaletteScratch()
	if pal.YMap != nil {
		store.Y = *pal.YMap
		pal.YMap = &store.Y
	}
	if pal.UVMap != nil {
		store.UV = *pal.UVMap
		pal.UVMap = &store.UV
	}
	return visit
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
	EOBMultiContext [3]int
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
	SegmentMapStride   int

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

	CoefficientBlocks int
	SkippedBlocks     int

	TXBs             int
	NonZero          int
	AllZero          int
	EOBTotal         int
	Residuals        int
	Predictions      int
	RestorationUnits int
}

// Run implements FrameWorkBatchRunner for the default residual decode path.
func (r *FrameWorkTileResidualRunner) Run(b FrameWorkBatch) error {
	if r == nil || int(b.Batch.Worker) >= len(r.Workers) {
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
func (b *FrameWorkBatch) JobBlockLoopRequest(index int, currentSegmentMap []uint8, previousSegmentMap []uint8, segmentMapStride int) (tile.BlockLoopRequest, error) {
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
		CurrentMVFrame:              b.CurrentMVFrame,
		FrameMIRows:                 frameMIRows,
		FrameMICols:                 frameMICols,
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
			mv := motion.Vector{
				Row: ref.Matrix[0] >> 13,
				Col: ref.Matrix[1] >> 13,
			}
			if forceIntegerMV {
				mv.Row = (mv.Row >> 3) << 3
				mv.Col = (mv.Col >> 3) << 3
			} else if !allowHighPrecisionMV {
				mv.Row = (mv.Row >> 1) << 1
				mv.Col = (mv.Col >> 1) << 1
			}
			mvs[i] = mv
		}
	}
	return mvs, types
}

func frameWorkBlockLoopRefSignBias(enabled bool, bits uint8, current uint32, refs [parser.InterRefsPerFrame]uint32) ([parser.InterRefsPerFrame]bool, error) {
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

func frameWorkBlockLoopRefFrameSide(enabled bool, bits uint8, current uint32, refs [parser.InterRefsPerFrame]uint32) ([parser.InterRefsPerFrame]int8, error) {
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
const refNoScaleFP int32 = 1 << motion.RefScaleShift

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
	if rootSize == 0 || region.MIColEnd <= region.MIColStart {
		return 0, ErrInvalidBatch
	}
	return int((region.MIColEnd - region.MIColStart + rootSize - 1) / rootSize), nil
}

// DecodeAndReconstructJobResiduals walks one tile job's blocks, invokes the
// caller's prediction hook, decodes residual TXBs, and reconstructs each decoded
// TXB into the batch output frame.
func (b FrameWorkBatch) DecodeAndReconstructJobResiduals(index int, state *tile.DecodeState, cdfs FrameWorkTileResidualCDFs, scratch *FrameWorkTileResidualScratch, req FrameWorkTileResidualRequest) (FrameWorkTileResidualStats, error) {
	if state == nil || scratch == nil || (req.Transforms == nil && !req.UseDefaultTransforms) {
		return FrameWorkTileResidualStats{}, ErrInvalidBatch
	}
	scratch.stats = FrameWorkTileResidualStats{}
	if req.CDEFIndexMap == nil {
		req.CDEFIndexMap = b.CDEFIndexMap
	}
	if req.LoopFilterMap == nil {
		req.LoopFilterMap = b.LoopFilterMap
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
	b.geomCache = &scratch.geomCache
	// The deferred wavefront engages only when the pool offered idle lanes
	// (single-tile frame on a multi-worker pool) and the tile clears the size
	// threshold. The GOAV1_DEFER_RECON force-on hook keeps the serial deferred
	// replay available as a byte-identity test harness without spawning
	// goroutines. When neither is set, the fused single-thread path runs
	// verbatim at zero added cost.
	wavefrontWorkers := 0
	if req.Predict == nil && frameWorkWavefrontEligible(b, index) {
		wavefrontWorkers = b.WavefrontWorkers
	}
	deferReconstruction := frameWorkDeferReconstruction || wavefrontWorkers > 1
	scratch.controller = frameWorkTileResidualLoopController{
		batch:                  b,
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
	}
	// Wire the fused / serial-deferred reconstruction state to the controller's
	// request scratch and shared geometry cache. The fused single-thread path
	// pays no extra copy: it dereferences exactly the buffers the previous
	// inline calls used.
	scratch.controller.recon = frameWorkReconState{
		batch:             b,
		index:             index,
		stats:             &scratch.stats,
		predict:           req.Predict,
		predictionScratch: req.PredictionScratch,
		cflScratch:        &scratch.CFL,
		int32Scratch:      req.Int32Scratch,
		residualScratch:   req.ResidualScratch,
	}
	if loopReq.BeforeSuperblock != nil || readRestoration {
		if scratch.beforeSuperblock == nil {
			scratch.beforeSuperblock = scratch.controller.BeforeSuperblock
		}
		loopReq.BeforeSuperblock = scratch.beforeSuperblock
	}

	loopStats, err := tile.DecodeBlockLoopWithCoeffControllerPtr(state, loopCDFs, &scratch.Loop, loopReq, &scratch.controller, func(visit *tile.BlockLoopVisit) error {
		if !visit.CoefficientsValid {
			return ErrInvalidBatch
		}
		frameWorkAccumulateResidualStats(&scratch.stats, visit.Coefficients.TotalStats())
		if req.LoopFilterMap != nil {
			if err := req.LoopFilterMap.MarkBlockPtr(visit, state); err != nil {
				return fmt.Errorf("mark loop filter block=%+v prediction=%+v: %w", visit.Block, visit.Prediction, err)
			}
		}
		if req.AfterBlock != nil {
			return req.AfterBlock(*visit)
		}
		return nil
	})
	scratch.stats.Loop = loopStats
	if err != nil {
		return scratch.stats, err
	}
	if scratch.controller.deferReconstruction {
		if scratch.controller.wavefrontWorkers > 1 {
			if err := scratch.controller.replayDeferredReconstructionWavefront(); err != nil {
				return scratch.stats, err
			}
		} else if err := scratch.controller.replayDeferredReconstruction(); err != nil {
			return scratch.stats, err
		}
	}
	return scratch.stats, nil
}

// frameWorkWavefrontEligible reports whether the deferred SB-row wavefront may
// engage for Jobs[index]. The wavefront helps only when the pool offered idle
// lanes (set by the pool when a single batch is dispatched) and the tile has
// enough SB rows to keep those lanes busy past the wavefront ramp. Below the
// threshold the fused single-thread path is the better choice (no buffering, no
// goroutines), so this returns false and the caller leaves reconstruction
// fused.
func frameWorkWavefrontEligible(b FrameWorkBatch, index int) bool {
	workers := b.WavefrontWorkers
	if workers <= 1 {
		return false
	}
	region, err := b.JobRegion(index)
	if err != nil {
		return false
	}
	sbRows := int(region.SBRows)
	// Require enough SB rows per worker so the wavefront diagonal fills before
	// it drains. Tiny frames stay on the fused path.
	return sbRows >= frameWorkWavefrontMinSBRowsPerWorker*workers
}

// DecodeAndReconstructJobResidualsDefault derives the standard block-loop and
// transform requests from the frame-work batch, then decodes and reconstructs
// one tile job using caller-owned scratch buffers.
func (b *FrameWorkBatch) DecodeAndReconstructJobResidualsDefault(index int, state *tile.DecodeState, cdfs FrameWorkTileResidualCDFs, scratch *FrameWorkTileResidualScratch, req FrameWorkTileResidualDefaultRequest) (FrameWorkTileResidualStats, error) {
	loopReq, err := b.JobBlockLoopRequest(index, req.CurrentSegmentMap, req.PreviousSegmentMap, req.SegmentMapStride)
	if err != nil {
		return FrameWorkTileResidualStats{}, err
	}
	return b.DecodeAndReconstructJobResiduals(index, state, cdfs, scratch, FrameWorkTileResidualRequest{
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
	wavefrontWorkers int

	// recon holds the reconstruction state for the fused single-thread path and
	// the serial deferred replay. The multi-worker wavefront uses its own
	// per-goroutine states instead. It is wired to the controller's request
	// scratch so the fused path pays no extra copy.
	recon frameWorkReconState
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

	// geom is this goroutine's private job-geometry cache; batch.geomCache
	// points at it on the wavefront path so the per-block memoization never
	// races. localStats accumulates the goroutine's reconstruction counters,
	// merged into the controller stats after the wavefront joins.
	geom       frameWorkJobGeometryCache
	localStats FrameWorkTileResidualStats

	pendingCFLPrediction bool
	cflPredictionDone    bool
	cflVisit             tile.BlockLoopVisit
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
	c.stats.RestorationUnits += n
	return nil
}

func (c *frameWorkTileResidualLoopController) BeforeBlockCoefficients(visit tile.BlockLoopVisit) error {
	return c.BeforeBlockCoefficientsPtr(&visit)
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
		if err := c.req.CDEFIndexMap.MarkBlockPtr(c.batch.CDEF, visit); err != nil {
			return err
		}
	}
	if visit.Prefix.SkipTransform {
		c.stats.SkippedBlocks++
	} else {
		c.stats.CoefficientBlocks++
	}
	if c.deferReconstruction {
		c.scratch.reconEvents = append(c.scratch.reconEvents, frameWorkReconEvent{
			kind:  frameWorkReconEventBlockBegin,
			visit: c.scratch.captureDeferredVisit(*visit),
		})
		return nil
	}
	return c.fusedReconState().predictBlockBegin(visit)
}

// fusedReconState returns the controller's reconstruction state for the fused
// single-thread path, binding it from the controller's request/scratch the
// first time it is needed. DecodeAndReconstructJobResiduals binds it up front
// on the production path; this lazy bind keeps direct-construction unit tests
// (which build the controller as a struct literal) working without each test
// wiring the state by hand. The bind is a single nil check per block on the hot
// path.
func (c *frameWorkTileResidualLoopController) fusedReconState() *frameWorkReconState {
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
	}
	return &c.recon
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
	if visit.Prefix.SkipTransform {
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
	if visit.Prediction.Intra && !visit.Prefix.SkipTransform {
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

func (c *frameWorkTileResidualLoopController) SelectBlockCoeffRequest(visit tile.BlockLoopVisit) (tile.BlockCoeffRequest, error) {
	return c.SelectBlockCoeffRequestPtr(&visit)
}

func (c *frameWorkTileResidualLoopController) SelectBlockCoeffRequestPtr(visit *tile.BlockLoopVisit) (tile.BlockCoeffRequest, error) {
	if visit == nil {
		return tile.BlockCoeffRequest{}, ErrInvalidBatch
	}
	var transforms FrameWorkBlockTransforms
	var segmentQIndex uint8
	var lossless bool
	if c.req.Transforms != nil {
		var err error
		transforms, err = c.req.Transforms(*visit)
		if err != nil {
			return tile.BlockCoeffRequest{}, err
		}
		// libaom uses xd->qindex[segment_id] (the segment-adjusted qindex) —
		// not the frame base — when gating tx-type syntax on the
		// qindex==0/lossless shortcut. Custom transform selectors do not carry
		// that cached value, so resolve it here.
		segmentQIndex, lossless, err = c.batch.BlockQIndex(c.state.CurrentBaseQIdx, visit.SegmentID)
		if err != nil {
			return tile.BlockCoeffRequest{}, err
		}
	} else if c.req.UseDefaultTransforms {
		var err error
		transforms, segmentQIndex, lossless, err = c.batch.readBlockTransformsAndQIndexPtr(c.state, visit)
		if err != nil {
			return tile.BlockCoeffRequest{}, err
		}
	} else {
		return tile.BlockCoeffRequest{}, ErrInvalidBatch
	}
	transformSelect := transforms.TransformSelect
	if transforms.ReadIntraTX {
		if !visit.Prediction.Valid || !visit.Prediction.Intra {
			return tile.BlockCoeffRequest{}, ErrInvalidBatch
		}
		chromaMode := visit.Prediction.ChromaMode
		if !chromaMode.Valid() {
			chromaMode = tile.ChromaIntraModeDC
		}
		lumaMode := frameWorkIntraTransformMode(visit.Prediction)
		c.scratch.IntraTX.Reset(c.state, c.cdfs.TransformType, c.batch.FrameMode.ReducedTxSet, visit.Prefix.SkipTransform, lossless, segmentQIndex, lumaMode, chromaMode)
		transformSelect = &c.scratch.IntraTX
	} else if transforms.ReadInterTX {
		c.scratch.InterTX.ResetForColor(c.state, c.cdfs.TransformType, c.batch.FrameMode.ReducedTxSet, visit.Prefix.SkipTransform, lossless, segmentQIndex, c.batch.Sequence.ColorConfig)
		transformSelect = &c.scratch.InterTX
	}
	return tile.BlockCoeffRequest{
		Transform: tile.TransformTreeRequest{
			Size:          visit.Block.Size,
			X4:            visit.Block.X4,
			Y4:            visit.Block.Y4,
			VisibleW4:     visit.Block.VisibleW4,
			VisibleH4:     visit.Block.VisibleH4,
			HaveTop:       visit.Block.HaveTop,
			HaveLeft:      visit.Block.HaveLeft,
			Color:         c.batch.Sequence.ColorConfig,
			TransformMode: c.req.TransformMode,
			Inter:         transforms.Inter,
			SkipTransform: visit.Prefix.SkipTransform,
			Lossless:      lossless,
		},
		LumaType:              transforms.Luma,
		ChromaType:            transforms.Chroma,
		TransformSelect:       transformSelect,
		EOBMultiContext:       transforms.EOBMultiContext,
		SkipAllZeroCoeffClear: c.userCoeffVisitor == nil,
	}, nil
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

func (c *frameWorkTileResidualLoopController) VisitBlockCoeff(visit tile.BlockLoopVisit, block tile.BlockCoeffBlock) error {
	return c.VisitBlockCoeffPtr(&visit, &block)
}

func (c *frameWorkTileResidualLoopController) VisitBlockCoeffPtr(visit *tile.BlockLoopVisit, block *tile.BlockCoeffBlock) error {
	if visit == nil || block == nil {
		return ErrInvalidBatch
	}
	if c.deferReconstruction {
		c.bufferReconTXB(visit, block, c.state.CurrentBaseQIdx)
	} else if err := c.fusedReconState().reconstructTXB(visit, block, c.state.CurrentBaseQIdx); err != nil {
		return err
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
// event's coefficients. Scan is dropped: reconstruction consumes only EOB,
// Coeffs, and geometry-derived scan size.
func (c *frameWorkTileResidualLoopController) bufferReconTXB(visit *tile.BlockLoopVisit, block *tile.BlockCoeffBlock, currentQIndex uint8) {
	buffered := *block
	if block.Result.AllZero {
		buffered.Coeffs = nil
	} else if n := len(block.Coeffs); n > 0 {
		off := len(c.scratch.coeffArena)
		c.scratch.coeffArena = append(c.scratch.coeffArena, block.Coeffs...)
		buffered.Coeffs = c.scratch.coeffArena[off : off+n : off+n]
	} else {
		buffered.Coeffs = nil
	}
	buffered.Scan = nil
	c.scratch.reconEvents = append(c.scratch.reconEvents, frameWorkReconEvent{
		kind:          frameWorkReconEventTXB,
		visit:         c.scratch.captureDeferredVisit(*visit),
		block:         buffered,
		currentQIndex: currentQIndex,
	})
}

// replayDeferredReconstruction runs the buffered predict and reconstruct events
// in decode order after the whole tile's entropy decode has completed. Because
// the events are replayed in the same order the fused path interleaved them,
// the CfL luma-first state machine and intra neighbor reads observe identical
// inputs, producing byte-identical output.
func (c *frameWorkTileResidualLoopController) replayDeferredReconstruction() error {
	return frameWorkReplayReconEvents(&c.recon, c.scratch.reconEvents)
}

// frameWorkReplayReconEvents runs a span of buffered reconstruction events in
// decode order on one reconstruction state. It is the shared inner loop of both
// the serial deferred replay and each wavefront goroutine's per-SB-row work.
func frameWorkReplayReconEvents(s *frameWorkReconState, events []frameWorkReconEvent) error {
	for i := range events {
		ev := &events[i]
		switch ev.kind {
		case frameWorkReconEventBlockBegin:
			if err := s.predictBlockBegin(&ev.visit); err != nil {
				return err
			}
		case frameWorkReconEventTXB:
			if err := s.reconstructTXB(&ev.visit, &ev.block, ev.currentQIndex); err != nil {
				return err
			}
		default:
			return ErrInvalidBatch
		}
	}
	return nil
}

// frameWorkReconSB records one superblock's span of buffered reconstruction
// events: the half-open [start, end) range into the flat reconEvents list and
// the SB column within its row. Events for one SB are contiguous because the
// block loop decodes superblocks in raster order.
type frameWorkReconSB struct {
	start int
	end   int
	col   int
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
// the only cross-goroutine communication is the atomic done counters.
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
	workers := min(c.wavefrontWorkers, rowCount)

	if c.scratch.wavefront == nil {
		c.scratch.wavefront = &frameWorkReconWavefront{}
	}
	wf := c.scratch.wavefront
	// Partition the flat event list into per-SB-row spans and record each
	// superblock's column. rowStart[r] is the first SB of row r in the sbs
	// slice; rowStart[rowCount] is the total SB count. A new SB begins at a
	// blockBegin event whose (sbRow, sbCol) differs from the previous block's.
	wf.rowStart = wf.rowStart[:0]
	sbs := wf.reuseSBs()
	prevRow, prevCol := -1, -1
	for i := range c.scratch.reconEvents {
		ev := &c.scratch.reconEvents[i]
		if ev.kind != frameWorkReconEventBlockBegin {
			continue
		}
		sbRow := int((ev.visit.Block.MIRow - region.MIRowStart) / sbSizeMIB)
		sbCol := int((ev.visit.Block.MICol - region.MIColStart) / sbSizeMIB)
		if sbRow < 0 || sbRow >= rowCount || sbCol < 0 {
			return ErrInvalidBatch
		}
		if sbRow == prevRow && sbCol == prevCol {
			continue
		}
		if sbRow < prevRow || (sbRow == prevRow && sbCol < prevCol) {
			// Events must arrive in SB-raster order for the wavefront gate to be
			// valid; bail to the safe serial replay if they ever do not.
			return ErrInvalidBatch
		}
		// Close the previous SB span.
		if len(sbs) > 0 {
			sbs[len(sbs)-1].end = i
		}
		// Open new row spans up to and including sbRow.
		for len(wf.rowStart) <= sbRow {
			wf.rowStart = append(wf.rowStart, len(sbs))
		}
		sbs = append(sbs, frameWorkReconSB{start: i, end: len(c.scratch.reconEvents), col: sbCol})
		prevRow, prevCol = sbRow, sbCol
	}
	for len(wf.rowStart) <= rowCount {
		wf.rowStart = append(wf.rowStart, len(sbs))
	}
	wf.commitSBs(sbs)

	wf.events = c.scratch.reconEvents
	wf.aborted.Store(false)
	wf.ensureDone(rowCount)
	for r := range rowCount {
		wf.done[r].Store(0)
	}

	if err := wf.ensureStates(workers, c); err != nil {
		return err
	}

	var wg sync.WaitGroup
	var firstErr atomic.Pointer[wavefrontError]
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func(w int) {
			defer wg.Done()
			st := &wf.states[w]
			for row := w; row < rowCount; row += workers {
				if err := wf.reconstructRow(st, row, rowCount); err != nil {
					// Signal abort so any goroutine spinning on this row's (now
					// never-published) done counter stops waiting instead of
					// deadlocking, then record the first error.
					wf.aborted.Store(true)
					firstErr.CompareAndSwap(nil, &wavefrontError{err: err})
					return
				}
			}
		}(w)
	}
	wg.Wait()
	if e := firstErr.Load(); e != nil {
		return e.err
	}
	// Merge the per-goroutine reconstruction counters into the tile stats. The
	// goroutines have all joined, so this read is race-free.
	for w := 0; w < workers; w++ {
		c.stats.Predictions += wf.states[w].localStats.Predictions
		c.stats.Residuals += wf.states[w].localStats.Residuals
	}
	return nil
}

// wavefrontError boxes a worker error so atomic.Pointer can carry the first one
// observed across the wavefront goroutines.
type wavefrontError struct{ err error }

// reconstructRow reconstructs every superblock of one SB row in left-to-right
// order, gating each column on the row above via the done counters.
func (wf *frameWorkReconWavefront) reconstructRow(st *frameWorkReconState, row int, rowCount int) error {
	// rowStart holds rowCount+1 entries and done holds rowCount; validating the
	// row index against both once lets the prove pass discharge the per-SB
	// rowStart[row], rowStart[row+1] and done[row] accesses below (and, under the
	// row>0 guard, the row-1 reads) without a check per superblock.
	if row < 0 || row >= rowCount || row+1 >= len(wf.rowStart) || row >= len(wf.done) {
		return ErrInvalidBatch
	}
	sbs := wf.sbs()
	lo := wf.rowStart[row]
	hi := wf.rowStart[row+1]
	if lo < 0 || hi < lo || hi > len(sbs) {
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
		aboveCount = wf.rowStart[row] - wf.rowStart[row-1]
		donePrev = &wf.done[row-1]
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
			need := int32(sb.col + 2)
			if int(need) > aboveCount {
				need = int32(aboveCount)
			}
			for donePrev.Load() < need {
				if wf.aborted.Load() {
					return nil
				}
				runtime.Gosched()
			}
		}
		if err := frameWorkReplayReconEvents(st, wf.eventSpan(sb.start, sb.end)); err != nil {
			return err
		}
		doneCur.Store(int32(sb.col + 1))
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
	if s.predict == nil && s.predictionScratch != nil && visit.Prediction.Valid && visit.Prediction.Intra && !visit.Prefix.SkipTransform {
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
	if err := s.batch.reconstructBlockCoeffCore(s.index, visit.Block, block, block.Transform, currentQIndex, visit.SegmentID, s.int32Scratch, s.residualScratch); err != nil {
		return fmt.Errorf("reconstruct plane=%d block=%+v tx=%d: %w", block.Plane, block.Block, block.Transform, err)
	}
	s.stats.Residuals++
	return nil
}

func frameWorkIntraTransformMode(pred tile.BlockPredictionModeResult) tile.IntraMode {
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
	stats.TXBs += coeff.TXBs
	stats.NonZero += coeff.NonZero
	stats.AllZero += coeff.AllZero
	stats.EOBTotal += coeff.EOBTotal
}
