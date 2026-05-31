package goav1

import (
	"errors"
	"fmt"
)

// Decoder is an ergonomic, high-level wrapper around the byte-exact public
// residual-stream-runner decode path. It hides the verbose scratch-binding
// sequence a low-level integration must perform (probe stream -> derive frame
// format -> bind frame pool -> size and allocate the residual-stream scratch
// arenas -> bind the stream-plan runner -> drive RunLowOverhead with the
// reusable supported post-filter runner) while delegating decode to exactly
// that conformant path. It does not reimplement any decode logic.
//
// Construction probes the supplied AV1 temporal-unit payloads to size every
// scratch arena once; after construction, decoding is allocation-free for the
// same workloads the underlying stream runner is allocation-free for.
//
// Lifetime and aliasing: the *Frame values returned by DecodeNext, DecodeAll,
// and the one-shot DecodeIVF helper alias a caller-owned arena (the frame pool
// and the runner's output slice) that the Decoder owns and reuses across
// calls. A returned *Frame (and the plane pixel memory it points at) remains
// valid only until the next DecodeNext / DecodeAll call or until Close. The
// returned frame slice is also decoder-owned and reused on the next call. Copy
// the plane bytes out if you need them to outlive the next call. DecodeAll
// returns frames from a single underlying Run, so the batch it returns is
// mutually valid until the following call.
//
// A Decoder is not safe for concurrent use; serialize calls. The worker count
// (see WithWorkers) controls intra-frame tile parallelism inside the runner,
// not concurrent use of the Decoder itself.
type Decoder struct {
	pool       FramePool
	outputPool FramePool
	workerPool *TileWorkerPool

	stream   DecoderStream
	refs     DecoderSurfaceReferences
	state    DecoderFrameWorkState
	stats    DecoderFrameWorkTileResidualStats
	sideData DecoderFrameWorkSideData
	batch    DecoderFrameWorkBatchResidualRunner

	refSurface []int
	refFrames  []*Frame
	releases   []int

	scratch    DecoderFrameWorkResidualStreamScratch
	runner     DecoderFrameWorkResidualStreamRunner
	postFilter DecoderFrameWorkReusableSupportedPostFilterRunner
	external   decoderExternalPostFilterRunner

	payloads [][]byte
	format   FrameFormat

	next        int
	closed      bool
	useExternal bool
	visible     []*Frame
}

// decoderConfig holds resolved construction options.
type decoderConfig struct {
	workers int
}

// Option configures a Decoder at construction time.
type Option func(*decoderConfig)

// WithWorkers sets the number of tile-worker goroutines the underlying runner
// uses for intra-frame tile parallelism. It must be >= 1. The default is 1,
// which reproduces the single-threaded conformance path exactly. Higher worker
// counts do not change decoded output, only how tiles within a frame are split
// across goroutines.
func WithWorkers(n int) Option {
	return func(c *decoderConfig) { c.workers = n }
}

func resolveConfig(opts []Option) decoderConfig {
	cfg := decoderConfig{workers: 1}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	return cfg
}

// NewDecoder builds a Decoder from a complete, ordered set of AV1 temporal-unit
// payloads (one per IVF frame). The payloads are probed up front to size scratch
// and to derive the frame format and superblock alignment from the bound stream
// headers via FrameCodedFormatFromHeaders, so callers never guess the surface
// geometry.
//
// payloads is retained by reference (not copied); the caller must keep the
// backing bytes alive and unmodified for the Decoder's lifetime. Decode frames
// in order with DecodeNext or all at once with DecodeAll. Always Close the
// Decoder to release its worker goroutines.
func NewDecoder(payloads [][]byte, opts ...Option) (*Decoder, error) {
	cfg := resolveConfig(opts)
	if cfg.workers < 1 {
		return nil, fmt.Errorf("goav1: workers must be >= 1, got %d", cfg.workers)
	}
	if len(payloads) == 0 {
		return nil, errors.New("goav1: no payloads to decode")
	}

	workerPool, err := NewTileWorkerPool(cfg.workers)
	if err != nil {
		return nil, fmt.Errorf("goav1: worker pool: %w", err)
	}

	// Probe the stream to learn how much scratch the runner will need and to
	// bind the sequence/frame headers used to derive the pool format.
	var probeStream DecoderStream
	const eventsPerFrame = 16
	probeEventBudget := len(payloads)*eventsPerFrame + 64
	probeEvents := make([]DecoderEvent, probeEventBudget)
	probeSpans := make([]TileSpan, MaxTiles)
	probeJobs := make([]TileJob, MaxTiles)
	probeBatches := make([]TileBatch, MaxTiles)

	plan, err := DecoderFrameWorkResidualLowOverheadStreamsPlan(
		probeStream, payloads, cfg.workers,
		probeEvents, probeSpans, probeJobs, probeBatches,
	)
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: stream plan: %w", err)
	}
	if !plan.HasEvent() {
		workerPool.Close()
		return nil, errors.New("goav1: stream plan did not identify a bind event")
	}

	// Derive the pool format from the bound headers so the superblock alignment
	// (64 vs 128) matches the surface the decoder reconstructs into.
	format, err := FrameCodedFormatFromHeaders(plan.Bind.Sequence, plan.Bind.Event.FrameSize, 64)
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: frame format from stream plan: %w", err)
	}

	const surfaceCount = RefFrames + 1
	pool, err := newDecoderFramePool(format, surfaceCount)
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: frame pool: %w", err)
	}

	useExternal, outputFormat, err := decoderExternalOutputFormat(payloads, 64)
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: super-res output format: %w", err)
	}
	var outputPool FramePool
	if useExternal {
		outputPool, err = newDecoderFramePool(outputFormat, surfaceCount)
		if err != nil {
			workerPool.Close()
			return nil, fmt.Errorf("goav1: output frame pool: %w", err)
		}
	}

	d := &Decoder{
		pool:        pool,
		outputPool:  outputPool,
		workerPool:  workerPool,
		refSurface:  make([]int, InterRefsPerFrame),
		refFrames:   make([]*Frame, InterRefsPerFrame),
		releases:    make([]int, RefFrames),
		payloads:    payloads,
		format:      format,
		useExternal: useExternal,
		visible:     make([]*Frame, 0, plan.Size.Event.Outputs),
	}
	if d.useExternal {
		d.external.outputPool = &d.outputPool
	}
	d.scratch = newDecoderStreamScratch(plan.Size)

	runtime := DecoderFrameWorkResidualEventRuntime{
		State:             &d.state,
		Refs:              &d.refs,
		FramePool:         &d.pool,
		Align:             64,
		ReferenceSurfaces: d.refSurface,
		ReferenceFrames:   d.refFrames,
		Releases:          d.releases,
		WorkerPool:        d.workerPool,
		SideData:          &d.sideData,
		Stats:             &d.stats,
	}
	if d.useExternal {
		provider := decoderExternalSurfaceProvider{coded: &d.pool, output: &d.outputPool}
		runtime.External = DecoderFrameWorkExternalReferenceRuntime{
			Provider:      provider,
			GlobalSurface: func(local int) int { return local },
			Releaser:      provider,
		}
	}
	runner, _, err := BindDecoderFrameWorkResidualStreamPlanRunner(plan, &d.stream,
		runtime, d.scratch, &d.batch)
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: bind runner: %w", err)
	}
	d.runner = runner
	if cap(d.visible) == 0 {
		d.visible = make([]*Frame, 0, 1)
	}
	return d, nil
}

// NewDecoderFromIVF parses an in-memory IVF stream and builds a Decoder over
// every frame it contains. The frame payloads are copied out of the iterator
// (which reuses its backing buffer), so ivf does not need to remain alive after
// this call returns.
func NewDecoderFromIVF(ivf []byte, opts ...Option) (*Decoder, error) {
	payloads, err := ivfPayloads(ivf)
	if err != nil {
		return nil, err
	}
	return NewDecoder(payloads, opts...)
}

// ivfPayloads demuxes an IVF stream into a fresh, caller-owned slice of AV1
// temporal-unit payloads. Each payload is copied because the IVF iterator may
// reuse its payload backing across Next calls.
func ivfPayloads(ivf []byte) ([][]byte, error) {
	it, err := NewIVFIterator(ivf)
	if err != nil {
		return nil, fmt.Errorf("goav1: ivf iterator: %w", err)
	}
	var payloads [][]byte
	for {
		f, ok, err := it.Next()
		if err != nil {
			return nil, fmt.Errorf("goav1: ivf next: %w", err)
		}
		if !ok {
			break
		}
		payloads = append(payloads, append([]byte(nil), f.Payload...))
	}
	if len(payloads) == 0 {
		return nil, errors.New("goav1: ivf stream produced no frames")
	}
	return payloads, nil
}

// DecodeNext decodes the next temporal-unit payload and returns the visible
// frames it produced, in display order. Most payloads yield exactly one frame,
// but a payload may yield zero (e.g. a hidden/no-show frame) or more than one
// (e.g. a show-existing-frame following a coded frame). When all payloads have
// been consumed it returns (nil, false, nil); subsequent calls keep returning
// that until Reset.
//
// The returned *Frame values alias the Decoder's reused output arena and remain
// valid only until the next DecodeNext / DecodeAll call or Close. See the
// Decoder type documentation for full lifetime semantics.
func (d *Decoder) DecodeNext() (frames []*Frame, ok bool, err error) {
	if d.closed {
		return nil, false, errors.New("goav1: decoder closed")
	}
	if d.next >= len(d.payloads) {
		return nil, false, nil
	}
	i := d.next
	d.next++

	var result DecoderFrameWorkResidualStreamResult
	postFilter := DecoderFrameWorkPostFilterRunner(&d.postFilter)
	if d.useExternal {
		postFilter = &d.external
	}
	if err := d.runner.RunLowOverheadIntoWithPostFilterRunner(&result, d.payloads[i], postFilter); err != nil {
		return nil, false, fmt.Errorf("goav1: frame %d: %w", i, err)
	}

	out := d.visible[:0]
	for _, f := range result.Run.Outputs {
		if f != nil {
			out = append(out, f)
		}
	}
	d.visible = out
	return out, true, nil
}

// DecodeAll decodes every remaining payload and returns all visible frames in
// display order as a single batch.
//
// IMPORTANT lifetime caveat: the returned *Frame values all alias the Decoder's
// reused output arena, which the underlying runner overwrites on each payload.
// Across a multi-payload stream the same arena slots are reused, so only the
// frames from the final payload are guaranteed to hold their decoded pixels by
// the time DecodeAll returns. Callers that need every frame's pixels must use
// DecodeNext and copy the bytes out per payload (see DecodeIVF, which returns
// independent copies). DecodeAll is intended for single-frame / last-frame
// inspection and for measuring the decode without retaining pixels.
func (d *Decoder) DecodeAll() ([]*Frame, error) {
	if d.closed {
		return nil, errors.New("goav1: decoder closed")
	}
	var frames []*Frame
	for {
		got, ok, err := d.DecodeNext()
		if err != nil {
			return frames, err
		}
		if !ok {
			break
		}
		frames = append(frames, got...)
	}
	return frames, nil
}

// Reset rewinds the Decoder to its initial state so the same bound payloads can
// be decoded again from the start without reallocating scratch. It re-resets
// the frame pool, reference state, and stream runner exactly as a fresh decode
// would.
func (d *Decoder) Reset() error {
	if d.closed {
		return errors.New("goav1: decoder closed")
	}
	d.pool.Reset()
	if d.useExternal {
		d.outputPool.Reset()
	}
	d.refs.Reset()
	d.state.Reset()
	d.stats = DecoderFrameWorkTileResidualStats{}
	if err := d.runner.Reset(); err != nil {
		return fmt.Errorf("goav1: runner reset: %w", err)
	}
	d.next = 0
	return nil
}

// Close releases the worker goroutine pool owned by the Decoder. After Close,
// the Decoder must not be used. Close is idempotent.
func (d *Decoder) Close() {
	if d.closed {
		return
	}
	d.closed = true
	if d.workerPool != nil {
		d.workerPool.Close()
		d.workerPool = nil
	}
}

// DecodeIVF is a one-shot convenience helper: it demuxes an in-memory IVF
// stream, decodes every frame through the conformant public path, and returns
// each visible frame's planes as an independent, caller-owned copy in display
// order. Unlike the *Frame values returned by Decoder methods, these DecodedFrame
// copies own their pixel memory and remain valid indefinitely, so DecodeIVF is
// the simplest correct entry point when you just want the decoded pixels.
//
// It builds and closes an internal Decoder; for repeated decoding or to avoid
// the per-frame copy, construct a Decoder directly.
func DecodeIVF(ivf []byte, opts ...Option) ([]DecodedFrame, error) {
	dec, err := NewDecoderFromIVF(ivf, opts...)
	if err != nil {
		return nil, err
	}
	defer dec.Close()

	var out []DecodedFrame
	for {
		frames, ok, err := dec.DecodeNext()
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		for _, f := range frames {
			out = append(out, copyDecodedFrame(f))
		}
	}
	return out, nil
}

// DecodedFrame is an independent, caller-owned snapshot of one decoded frame's
// visible samples. Unlike the aliasing *Frame returned by Decoder methods, a
// DecodedFrame owns its plane bytes and stays valid for the lifetime the caller
// keeps it.
//
// Y, U, and V hold the visible (cropped) samples for each plane, row-major with
// no stride padding. For monochrome streams U and V are nil. Samples wider than
// 8 bits are stored little-endian as BytesPerSample bytes each, matching the
// canonical I420/I400 raw-video layout used by aomdec --rawvideo goldens.
type DecodedFrame struct {
	Width          int // luma visible width in samples
	Height         int // luma visible height in samples
	BytesPerSample int // 1 for 8-bit, 2 for 10/12-bit
	ChromaWidth    int // chroma plane visible width (0 if monochrome)
	ChromaHeight   int // chroma plane visible height (0 if monochrome)
	Y              []byte
	U              []byte
	V              []byte
}

// copyDecodedFrame extracts the visible samples of an aliasing *Frame into an
// independent DecodedFrame, stripping per-plane stride padding so the byte
// layout matches the canonical I420/I400 raw video the libaom goldens use.
func copyDecodedFrame(f *Frame) DecodedFrame {
	bps := f.Layout.BytesPerSample
	df := DecodedFrame{
		Width:          f.Y.Width,
		Height:         f.Y.Height,
		BytesPerSample: bps,
		ChromaWidth:    f.U.Width,
		ChromaHeight:   f.U.Height,
		Y:              copyPlane(f.Y, bps),
		U:              copyPlane(f.U, bps),
		V:              copyPlane(f.V, bps),
	}
	return df
}

// copyPlane copies a plane's visible samples into a fresh row-packed slice,
// dropping stride padding. It returns nil for empty planes (e.g. chroma of a
// monochrome frame).
func copyPlane(p FramePlane, bytesPerSample int) []byte {
	if p.Width == 0 || p.Height == 0 || len(p.Pix) == 0 {
		return nil
	}
	rowBytes := p.Width * bytesPerSample
	dst := make([]byte, rowBytes*p.Height)
	for row := 0; row < p.Height; row++ {
		off := row * p.Stride
		copy(dst[row*rowBytes:(row+1)*rowBytes], p.Pix[off:off+rowBytes])
	}
	return dst
}

// newDecoderFramePool allocates and binds a frame pool sized for the given
// format and surface count using the public sizing helper.
func newDecoderFramePool(format FrameFormat, count int) (FramePool, error) {
	_, backingSize, err := FramePoolRequiredSize(format, count)
	if err != nil {
		return FramePool{}, err
	}
	return BindFramePool(make([]byte, backingSize), format,
		make([]Frame, count), make([]int, count), make([]bool, count))
}

// newDecoderStreamScratch allocates every scratch arena the residual stream
// runner needs, sized from the probed plan. It mirrors the binding performed by
// cmd/aom-go-dec and the conformance harness so the convenience path is the
// same byte-exact path.
func newDecoderStreamScratch(size DecoderFrameWorkResidualStreamScratchSize) DecoderFrameWorkResidualStreamScratch {
	return DecoderFrameWorkResidualStreamScratch{
		Events:    make([]DecoderEvent, size.Events),
		Event:     newDecoderEventScratch(size.Event),
		SideData:  newDecoderSideDataScratch(size.Event.SideData),
		Outputs:   make([]*Frame, size.Event.Outputs),
		RTPBuffer: make([]byte, size.RTPBuffer),
		RTPSpans:  make([]RTPObuSpan, size.RTPSpans),
	}
}

func newDecoderEventScratch(size DecoderFrameWorkResidualEventScratchSize) DecoderFrameWorkResidualEventScratch {
	return DecoderFrameWorkResidualEventScratch{
		Runner:   newDecoderBatchRunnerScratch(size.Runner),
		SideData: newDecoderSideDataScratch(size.SideData),
		Spans:    make([]TileSpan, size.Plan.SpanCount),
		Jobs:     make([]TileJob, size.Plan.JobCount),
		Batches:  make([]TileBatch, size.Plan.BatchCount),
	}
}

func newDecoderBatchRunnerScratch(size DecoderFrameWorkBatchResidualRunnerScratchSize) DecoderFrameWorkBatchResidualRunnerScratch {
	return DecoderFrameWorkBatchResidualRunnerScratch{
		States:                  make([]TileDecodeState, size.Workers),
		Storages:                make([]DecoderFrameWorkTileResidualCDFStorage, size.Workers),
		TileScratch:             make([]DecoderFrameWorkTileResidualScratch, size.Workers),
		RestorationRequests:     make([]DecoderFrameWorkTileRestorationRequest, size.RestorationRequests),
		PredictionScratch:       make([]DecoderFrameWorkPredictionScratch, size.Workers),
		InterPredictionScratch:  make([]DecoderFrameWorkInterPredictionScratch, size.Workers),
		Stats:                   make([]DecoderFrameWorkTileResidualStats, size.Workers),
		Int32Scratch:            make([]int32, size.Int32Scratch),
		ResidualScratch:         make([]int16, size.ResidualScratch),
		LoopContextAboveScratch: make([]TileBlockLoopRootAboveContext, size.LoopContextAbove),
	}
}

func newDecoderSideDataScratch(size DecoderFrameWorkSideDataScratchSize) DecoderFrameWorkSideDataScratch {
	return DecoderFrameWorkSideDataScratch{
		CDEFIndexMap:             make([]uint8, size.CDEFIndexMap),
		CDEFReadMap:              make([]bool, size.CDEFReadMap),
		LoopFilterMap:            make([]DecoderFrameWorkLoopFilterBlockRecord, size.LoopFilterMap),
		RestorationRecords:       make([]TileRestorationUnitRecord, size.RestorationRecords),
		RestorationBoundaryAbove: make([]uint16, size.RestorationBoundaryAbove),
		RestorationBoundaryBelow: make([]uint16, size.RestorationBoundaryBelow),
	}
}
