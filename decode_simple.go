package goav1

import (
	"errors"
	"fmt"

	internalcdef "github.com/thesyncim/goav1/internal/av1/cdef"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	internallfmask "github.com/thesyncim/goav1/internal/av1/lfmask"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"

	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
)

// Decoder decodes a complete AV1 stream. The constructor probes the stream,
// derives the frame format, and sizes and binds the frame pool, the
// residual-stream scratch and the post-filter runner once, so decoding itself
// does not allocate.
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
// (see WithWorkers) controls the decoder's internal parallelism (tiles, SB-row
// reconstruction, post-filter bands and, with WithMaxFrameDelay, frame
// threading), not concurrent use of the Decoder itself.
type Decoder struct {
	pool       internalframe.Pool
	outputPool internalframe.Pool
	workerPool *internalthreading.Pool

	stream   internaldecoder.Stream
	refs     internaldecoder.SurfaceReferences
	state    internaldecoder.FrameWorkState
	stats    internalthreading.FrameWorkTileResidualStats
	sideData decoderFrameWorkSideData
	batch    decoderFrameWorkBatchResidualRunner

	compactLF     internalthreading.FrameWorkLoopFilterCompactState
	lfLevels      internalthreading.FrameWorkLoopFilterLevelTable
	lfTileEdges   internalthreading.FrameWorkLoopFilterTileEdges
	lfNoSkip      internalthreading.FrameWorkLoopFilterNoSkip
	compactNoSkip []byte

	refSurface [internalparser.InterRefsPerFrame]int
	refFrames  [internalparser.InterRefsPerFrame]*Frame
	releases   [internalparser.RefFrames]int

	scratch    decoderFrameWorkResidualStreamScratch
	runner     decoderFrameWorkResidualStreamRunner
	postFilter decoderFrameWorkReusableSupportedPostFilterRunner
	external   decoderExternalPostFilterRunner

	// postFilterParallel fans the post-filter chain's independent row bands out
	// across the same worker count as tile decode. It is reused across frames so
	// the parallel path stays allocation-free after warm-up.
	postFilterParallel internaldecoder.FrameWorkPostFilterParallel

	payloadKind   decoderPayloadKind
	payloadSource decoderPayloadSource
	payloadBuf    []byte
	format        FrameFormat

	next        int
	closed      bool
	useExternal bool
	visible     []*Frame
	visibleOne  [1]*Frame

	// shownHeld tracks the pool surfaces behind the last returned batch.
	// Frames that refresh no reference slot are owned by nothing once
	// shown - the slot bookkeeping never sees them - so the next decode
	// call returns them to their pool when no reference still holds them.
	shownHeld    []shownSurface
	shownHeldBuf [2 * (internalparser.RefFrames + 1)]shownSurface

	// fp is the frame-threading pipeline (dav1d n_fc > 1), or nil when frames
	// decode one at a time.
	fp *decoderFramePipeline

	// Error concealment state (inactive and zero-alloc unless WithErrorConcealment(true)).
	errorConcealment bool
	concealFrame     Frame
	concealBuf       []byte
	hasConcealFrame  bool
}

// shownSurface names one tracked output surface and the pool that owns it.
type shownSurface struct {
	pool    *internalframe.Pool
	index   int
	surface int
}

type decoderPayloadKind uint8

const (
	decoderPayloadLowOverhead decoderPayloadKind = iota
	decoderPayloadRTP
)

// decoderMaxPayloadEvents bounds the OBUs one payload may carry. Every OBU is
// parsed into a DecoderEvent of several kilobytes, so without a bound a payload
// of two-byte OBUs would make the decoder allocate thousands of times its size.
const decoderMaxPayloadEvents = 64

// decoderConfig holds resolved construction options.
type decoderConfig struct {
	workers          int
	maxFrameDelay    int
	frameSizeLimit   int
	disableFilmGrain bool
	operatingPoint   int
	onlyHighestLayer bool
	errorConcealment bool
}

// ErrDecoderFrameSizeLimit reports a stream that carries a frame larger than
// the limit set with WithFrameSizeLimit.
var ErrDecoderFrameSizeLimit = errors.New("goav1: frame size exceeds the configured limit")

// Option configures a Decoder at construction time.
type Option func(*decoderConfig)

// WithFilmGrain controls whether AV1 film-grain synthesis is applied to output
// frames that carry active film-grain parameters. The default is true. Setting
// it to false skips film-grain synthesis (matching dav1d --filmgrain 0) without
// affecting reference-frame reconstruction.
func WithFilmGrain(enabled bool) Option {
	return func(c *decoderConfig) { c.disableFilmGrain = !enabled }
}

// WithOperatingPoint selects the AV1 operating point index (0..31) per AV1
// spec section 6.4.1 / 7.5. OBUs outside the chosen operating point's
// operating_point_idc mask are skipped. The default is 0.
func WithOperatingPoint(index int) Option {
	return func(c *decoderConfig) { c.operatingPoint = index }
}

// WithAllLayers controls whether all spatial layers in the selected operating
// point are emitted as display frames (true, the default) or only the highest
// spatial layer in that operating point (false, matching dav1d --alllayers 0).
func WithAllLayers(all bool) Option {
	return func(c *decoderConfig) { c.onlyHighestLayer = !all }
}

// WithErrorConcealment enables decoder error concealment for resilient
// playback over lossy or corrupted bitstreams. When enabled, corrupt OBUs,
// truncated tile payloads, or missing-reference decode errors do not abort the
// stream: any active partial frame is aborted cleanly, RTP fragment state is
// reset if applicable, and the last successfully decoded frame is emitted with
// Concealed=true (or the payload is skipped if no good frame has been decoded
// yet) so decoding resumes on subsequent valid frames. Disabled by default.
func WithErrorConcealment(enabled bool) Option {
	return func(c *decoderConfig) { c.errorConcealment = enabled }
}

// WithWorkers sets the number of tile-worker goroutines the underlying runner
// uses for intra-frame tile parallelism. It must be >= 1. The default is 1,
// which reproduces the single-threaded conformance path exactly. Higher worker
// counts do not change decoded output, only how the work is split across
// goroutines. With more than one worker the low-overhead decoders also run
// dav1d-style frame threading by default, so DecodeNext returns frames up to
// ceil(sqrt(n))-1 calls late; see WithMaxFrameDelay (1 restores one frame per
// call).
func WithWorkers(n int) Option {
	return func(c *decoderConfig) { c.workers = n }
}

// WithFrameSizeLimit makes the decoder reject a stream that carries a frame of
// more than pixels luma samples (upscaled width times height) with
// ErrDecoderFrameSizeLimit, as dav1d's frame_size_limit does. A decoder sizes
// its buffers from the stream's own headers when it is constructed, so set a
// limit whenever the input is not trusted. Zero, the default, means no limit.
func WithFrameSizeLimit(pixels int) Option {
	return func(c *decoderConfig) { c.frameSizeLimit = pixels }
}

// decoderCheckFrameSizeLimit reports ErrDecoderFrameSizeLimit when one of
// events carries a frame size above limit luma samples.
func decoderCheckFrameSizeLimit(limit int, events []internaldecoder.Event) error {
	if limit <= 0 {
		return nil
	}
	for i := range events {
		size := &events[i].FrameSize
		if uint64(max(size.CodedWidth, size.UpscaledWidth))*uint64(size.Height) > uint64(limit) {
			return ErrDecoderFrameSizeLimit
		}
	}
	return nil
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
// headers via frameCodedFormatFromHeaders, so callers never guess the surface
// geometry.
//
// payloads is retained by reference (not copied); the caller must keep the
// backing bytes alive and unmodified for the Decoder's lifetime. Decode frames
// in order with DecodeNext or all at once with DecodeAll. Always Close the
// Decoder to release its worker goroutines.
func NewDecoder(payloads [][]byte, opts ...Option) (*Decoder, error) {
	return newDecoderFromPayloadSource(newSliceDecoderPayloadSource(payloads), opts...)
}

// NewDecoderFromRTPPayloads builds a Decoder from ordered AV1 RTP payload
// bodies, with RTP headers and extensions already removed. DecodeNext consumes
// one RTP payload per call; fragmented frames can return an empty frame slice
// with ok=true until the payload that completes the frame arrives.
//
// payloads is retained by reference (not copied); the caller must keep the
// backing bytes alive and unmodified for the Decoder's lifetime. The same
// decoder can also drive caller-supplied live payloads with DecodeRTPPayload and
// DecodeRTPPayloadAfterLoss; the construction payloads size the reusable arenas,
// so later live payloads must fit those planned RTP/event capacities.
//
// RTP decoders are realtime decoders: like realtime dav1d integrations
// (max_frame_delay = 1) they never frame-thread, whatever WithWorkers and
// WithMaxFrameDelay say, so every frame is returned by the call whose payload
// completes it.
func NewDecoderFromRTPPayloads(payloads [][]byte, opts ...Option) (*Decoder, error) {
	return newDecoderFromPayloadSourceKind(newSliceDecoderPayloadSource(payloads), decoderPayloadRTP, opts...)
}

func newDecoderFromPayloadSource(source decoderPayloadSource, opts ...Option) (*Decoder, error) {
	return newDecoderFromPayloadSourceKind(source, decoderPayloadLowOverhead, opts...)
}

func newDecoderFromPayloadSourceKind(source decoderPayloadSource, kind decoderPayloadKind, opts ...Option) (*Decoder, error) {
	cfg := resolveConfig(opts)
	if cfg.workers < 1 {
		return nil, fmt.Errorf("goav1: workers must be >= 1, got %d", cfg.workers)
	}
	if cfg.maxFrameDelay < 0 {
		return nil, fmt.Errorf("goav1: max frame delay must be >= 0, got %d", cfg.maxFrameDelay)
	}
	if cfg.frameSizeLimit < 0 {
		return nil, fmt.Errorf("goav1: frame size limit must be >= 0, got %d", cfg.frameSizeLimit)
	}
	if cfg.operatingPoint < 0 || cfg.operatingPoint >= 32 {
		return nil, fmt.Errorf("goav1: operating point must be in [0, 31], got %d", cfg.operatingPoint)
	}
	if source.len() == 0 {
		return nil, errors.New("goav1: no payloads to decode")
	}

	workerPool, err := internalthreading.NewPool(cfg.workers)
	if err != nil {
		return nil, fmt.Errorf("goav1: worker pool: %w", err)
	}

	// Probe the stream to learn how much scratch the runner will need and to
	// bind the sequence/frame headers used to derive the pool format.
	var probeStream internaldecoder.Stream
	probeStream.SetDisableFilmGrain(cfg.disableFilmGrain)
	probeStream.SetOperatingPoint(cfg.operatingPoint, !cfg.onlyHighestLayer)
	probeEvents := make([]internaldecoder.Event, decoderMaxPayloadEvents)
	probeSpans := make([]internalparser.TileSpan, internalparser.MaxTiles)
	probeJobs := make([]internaltile.Job, internalparser.MaxTiles)
	probeBatches := make([]internalthreading.Batch, internalparser.MaxTiles)
	probePayload := source.makeProbeBuffer()

	probeSource := source
	if cfg.errorConcealment {
		probeSource, err = filterValidPayloadsForConcealment(source, kind, cfg, probePayload)
		if err != nil {
			workerPool.Close()
			return nil, err
		}
	}

	var plan decoderFrameWorkResidualStreamPlan
	switch kind {
	case decoderPayloadLowOverhead:
		plan, err = decoderFrameWorkResidualLowOverheadStreamsPlanFromSource(
			probeStream, probeSource, cfg.workers, cfg.frameSizeLimit, probePayload,
			probeEvents, probeSpans, probeJobs, probeBatches,
		)
	case decoderPayloadRTP:
		plan, err = decoderFrameWorkResidualRTPPayloadsStreamPlanFromSource(
			probeStream, probeSource, cfg.workers, cfg.frameSizeLimit, probeEvents, probeSpans, probeJobs, probeBatches,
		)
	default:
		err = internaldecoder.ErrInvalidFrameWorkState
	}
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: stream plan: %w", err)
	}
	if !plan.HasEvent() {
		workerPool.Close()
		return nil, errors.New("goav1: stream plan did not identify a bind event")
	}

	// Pool slots fit the largest coded frame layout in the stream; each frame
	// binds a slot at its own size and format (dav1d allocates every picture at
	// its frame size and its sequence's bit depth, chroma layout and superblock
	// size, so a new coded video sequence may change any of them).
	format, err := decoderCodedPoolFormatFromSource(probeSource, kind, 64, probePayload)
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: frame format from stream plan: %w", err)
	}

	// Frame threading keeps up to n_fc frames in flight beyond the reference
	// set, each on its own surface. Error concealment isolates each payload
	// serially (tile workers still run in parallel) so a corrupt frame cannot
	// cascade reference-progress errors into queued future frames.
	frameContexts := 1
	if kind == decoderPayloadLowOverhead && !cfg.errorConcealment {
		frameContexts = decoderFrameContextCount(cfg.workers, cfg.maxFrameDelay)
	}
	surfaceCount := internalparser.RefFrames + 1
	if frameContexts > 1 {
		surfaceCount += 2 * frameContexts
	}
	var useExternal bool
	var outputFormat FrameFormat
	switch kind {
	case decoderPayloadLowOverhead:
		useExternal, outputFormat, err = decoderExternalOutputFormatFromSource(probeSource, 64, probePayload)
	case decoderPayloadRTP:
		useExternal, outputFormat, err = decoderExternalOutputFormatFromRTPSource(probeSource, 64)
	}
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: super-res output format: %w", err)
	}
	var motionSize FrameSize
	switch kind {
	case decoderPayloadLowOverhead:
		motionSize, err = decoderTemporalMotionScratchFrameSizeUpperBoundFromSource(probeSource, probeEvents, probePayload)
	case decoderPayloadRTP:
		motionSize, err = decoderTemporalMotionScratchFrameSizeUpperBoundFromRTPSource(probeSource)
	}
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: temporal motion scratch plan: %w", err)
	}
	var postArena decoderFrameWorkPostFilterRequestScratchSize
	switch kind {
	case decoderPayloadLowOverhead:
		postArena, err = decoderPostFilterScratchArenaUpperBoundFromSource(probeSource, 64, workerPool, probeEvents, probePayload)
	case decoderPayloadRTP:
		postArena, err = decoderPostFilterScratchArenaUpperBoundFromRTPSource(probeSource, 64, workerPool)
	}
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: postfilter scratch plan: %w", err)
	}
	arenaSize, err := decoderArenaSizeFor(format, outputFormat, useExternal, surfaceCount, plan.Size, postArena)
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: arena size: %w", err)
	}
	if err := arenaSize.addBytes(source.payloadBufferLen()); err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: payload buffer size: %w", err)
	}
	arena := newDecoderArena(arenaSize)

	pool, err := newDecoderFramePool(format, surfaceCount, &arena)
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: frame pool: %w", err)
	}
	var outputPool internalframe.Pool
	if useExternal {
		outputPool, err = newDecoderFramePool(outputFormat, surfaceCount, &arena)
		if err != nil {
			workerPool.Close()
			return nil, fmt.Errorf("goav1: output frame pool: %w", err)
		}
	}

	d := &Decoder{
		pool:             pool,
		outputPool:       outputPool,
		workerPool:       workerPool,
		payloadKind:      kind,
		payloadSource:    source,
		payloadBuf:       arena.takeBytes(source.payloadBufferLen()),
		format:           format,
		useExternal:      useExternal,
		errorConcealment: cfg.errorConcealment,
	}
	d.stream.SetDisableFilmGrain(cfg.disableFilmGrain)
	d.stream.SetOperatingPoint(cfg.operatingPoint, !cfg.onlyHighestLayer)
	internaldecoder.EnableOwnedTemporalMotionStorage(&d.state)
	if d.useExternal {
		d.external.outputPool = &d.outputPool
	}
	d.scratch = newDecoderStreamScratch(plan.Size, &arena)
	d.compactNoSkip = arena.takeBytes(decoderCompactNoSkipBytes(plan.Size.Event.SideData.CDEFIndexMap))
	d.lfNoSkip.Preseed(arena.takeBytes(decoderCompactNoSkipBytes(plan.Size.Event.SideData.CDEFIndexMap)))
	d.lfTileEdges.Preseed(arena.takeBytes(decoderLoopFilterTileEdgeBytes(plan.Size.Event.SideData.LoopFilterMasks)))
	if cap(d.scratch.Outputs) > 0 {
		d.visible = d.scratch.Outputs[:0]
	} else {
		d.visible = d.visibleOne[:0]
	}
	if err := d.state.PreallocTemporalMotionScratchFrames(motionSize, frameContexts); err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: temporal motion scratch: %w", err)
	}
	if d.useExternal {
		d.external.size = postArena
		d.external.scratch = decoderPostFilterScratchFromArena(postArena, &arena)
		d.external.supported.size = postArena
		d.external.supported.runner.Scratch = decoderPostFilterScratchFromArena(postArena, &arena)
	} else {
		d.postFilter.size = postArena
		d.postFilter.runner.Scratch = decoderPostFilterScratchFromArena(postArena, &arena)
	}
	// Let the post-filter chain fan its independent row bands across the tile
	// worker count. This is a byte-exact scheduling change (each band reads the
	// previous stage's complete output via boundary snapshots and writes disjoint
	// rows), so it never alters decoded output.
	d.postFilterParallel.Workers = d.workerPool.WorkerCount()
	d.postFilter.Parallel = &d.postFilterParallel
	d.external.supported.Parallel = &d.postFilterParallel

	runtime := decoderFrameWorkResidualEventRuntime{
		State:             &d.state,
		Refs:              &d.refs,
		FramePool:         &d.pool,
		Align:             64,
		ReferenceSurfaces: d.refSurface[:],
		ReferenceFrames:   d.refFrames[:],
		Releases:          d.releases[:],
		WorkerPool:        d.workerPool,
		SideData:          &d.sideData,
		Stats:             &d.stats,
	}
	if d.useExternal {
		runtime.TileListOutputPool = &d.outputPool
		provider := decoderExternalSurfaceProvider{coded: &d.pool, output: &d.outputPool}
		runtime.External = decoderFrameWorkExternalReferenceRuntime{
			Provider:      provider,
			GlobalSurface: func(local int) int { return local },
			Releaser:      provider,
			PostPublisher: &d.external,
		}
	}
	runner, _, err := bindDecoderFrameWorkResidualStreamPlanRunner(plan, &d.stream,
		runtime, d.scratch, &d.batch)
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: bind runner: %w", err)
	}
	d.runner = runner
	d.runner.EventRunner.lfLevels = &d.lfLevels
	d.runner.EventRunner.lfTileEdges = &d.lfTileEdges
	d.runner.EventRunner.lfNoSkip = &d.lfNoSkip
	if !d.useExternal && d.workerPool.WorkerCount() == 1 {
		d.batch.compactState = &d.compactLF
		d.batch.compactOwner = &d.batch
		d.batch.compactSideOwner = &d.sideData
		d.runner.EventRunner.compactState = &d.compactLF
		d.runner.EventRunner.compactNoSkip = d.compactNoSkip
		d.runner.EventRunner.compactPostOwner = &d.postFilter
		d.runner.EventRunner.compactBatchOwner = &d.batch
		d.runner.EventRunner.compactSideOwner = &d.sideData
	}
	if frameContexts > 1 && !d.useExternal {
		fp, err := newDecoderFramePipeline(d, frameContexts, plan, postArena, surfaceCount)
		if err != nil {
			workerPool.Close()
			return nil, fmt.Errorf("goav1: frame threading: %w", err)
		}
		d.fp = fp
		internaldecoder.EnableFrameThreadCDFPool(&d.state)
		d.runner.EventRunner.fp = fp
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
	// Count the frames first (the iterator only walks headers) so the payload
	// list and one shared copy of every payload are each a single allocation.
	count, total := 0, 0
	for {
		f, ok, err := it.Next()
		if err != nil {
			return nil, fmt.Errorf("goav1: ivf next: %w", err)
		}
		if !ok {
			break
		}
		count++
		total += len(f.Payload)
	}
	if count == 0 {
		return nil, errors.New("goav1: ivf stream produced no frames")
	}
	it, err = NewIVFIterator(ivf)
	if err != nil {
		return nil, fmt.Errorf("goav1: ivf iterator: %w", err)
	}
	payloads := make([][]byte, 0, count)
	backing := make([]byte, 0, total)
	for range count {
		f, ok, err := it.Next()
		if err != nil {
			return nil, fmt.Errorf("goav1: ivf next: %w", err)
		}
		if !ok {
			break
		}
		start := len(backing)
		backing = append(backing, f.Payload...)
		payloads = append(payloads, backing[start:len(backing):len(backing)])
	}
	if len(payloads) == 0 {
		return nil, errors.New("goav1: ivf stream produced no frames")
	}
	return payloads, nil
}

// DecodeNext decodes the next payload and returns the visible frames it
// produced, in display order. For NewDecoder/NewDecoderFromIVF this payload is
// one low-overhead temporal unit; for NewDecoderFromRTPPayloads it is one AV1
// RTP payload body. Most low-overhead payloads yield exactly one frame, but a
// payload may yield zero (e.g. a hidden/no-show frame or an RTP fragment before
// the frame completes) or more than one (e.g. a show-existing-frame following a
// coded frame). When all payloads have been consumed it returns
// (nil, false, nil); subsequent calls keep returning that until Reset.
//
// With frame threading (WithWorkers > 1 and a frame delay above 1, the
// default for low-overhead decoders) a call returns the frames that left the
// in-flight ring, so a frame can arrive up to delay-1 calls after the payload
// that carried it, and the final calls before (nil, false, nil) drain the ring.
// Order and pixels are unchanged.
//
// The returned *Frame values alias the Decoder's reused output arena and remain
// valid only until the next DecodeNext / DecodeAll call or Close. See the
// Decoder type documentation for full lifetime semantics.
func (d *Decoder) DecodeNext() (frames []*Frame, ok bool, err error) {
	defer func() {
		if err != nil && d != nil {
			internalthreading.InvalidateFrameWorkLoopFilterCompactState(&d.compactLF)
		}
	}()
	if d == nil || d.payloadSource.len() == 0 {
		return nil, false, errors.New("goav1: decoder is not initialized")
	}
	if d.closed {
		return nil, false, errors.New("goav1: decoder closed")
	}
	d.releaseShownSurfaces()
	if d.fp != nil {
		return d.decodeNextFrameThreaded()
	}
	if d.next >= d.payloadSource.len() {
		return nil, false, nil
	}
	i := d.next
	d.next++
	payload, err := d.payloadSource.payloadAt(i, d.payloadBuf)
	if err != nil {
		return nil, false, fmt.Errorf("goav1: frame %d payload: %w", i, err)
	}

	var result decoderFrameWorkResidualStreamResult
	postFilter := d.postFilterRunner()
	switch d.payloadKind {
	case decoderPayloadLowOverhead:
		err = d.runner.RunLowOverheadIntoWithPostFilterRunner(&result, payload, postFilter)
	case decoderPayloadRTP:
		err = d.runner.RunRTPPayloadIntoWithPostFilterRunner(&result, payload, postFilter)
	default:
		err = internaldecoder.ErrInvalidFrameWorkState
	}
	if err != nil {
		if d.errorConcealment {
			return d.recoverAndConceal(), true, nil
		}
		return nil, false, fmt.Errorf("goav1: payload %d: %w", i, err)
	}

	out := d.visible[:0]
	for _, f := range result.Run.Outputs {
		if f != nil {
			f.Concealed = false
			out = append(out, f)
		}
	}
	if d.errorConcealment && len(out) > 0 {
		d.snapshotConcealFrame(out[len(out)-1])
	}
	d.visible = out
	d.trackShownSurfaces(out)
	return out, true, nil
}

// DecodeRTPPayload decodes one caller-supplied AV1 RTP payload body using an
// RTP decoder constructed by NewDecoderFromRTPPayloads. It is intended for live
// receive loops that use the constructor payloads only to size the reusable
// decode arenas. Fragmented RTP payloads can return an empty frame slice until
// the payload that completes the frame arrives.
//
// Returned frames have the same lifetime as DecodeNext. Do not call this on a
// Decoder constructed from low-overhead or IVF payloads.
func (d *Decoder) DecodeRTPPayload(payload []byte) ([]*Frame, error) {
	return d.decodeExternalRTPPayload(payload, false)
}

// DecodeRTPPacket parses one complete RTP packet and decodes its AV1 RTP
// payload body using a decoder constructed by NewDecoderFromRTPPayloads or
// NewDecoderFromRTPPackets.
func (d *Decoder) DecodeRTPPacket(packet []byte) ([]*Frame, error) {
	rtp, err := ParseRTPPacket(packet)
	if err != nil {
		if d != nil {
			internalthreading.InvalidateFrameWorkLoopFilterCompactState(&d.compactLF)
		}
		return nil, fmt.Errorf("goav1: rtp packet: %w", err)
	}
	return d.decodeExternalRTPPayload(rtp.Payload, false)
}

// DecodeRTPSequencedPacket decodes one packet released by RTPPacketSequencer,
// using the after-loss RTP path when packet.AfterLoss is set.
func (d *Decoder) DecodeRTPSequencedPacket(packet RTPSequencedPacket) ([]*Frame, error) {
	return d.decodeExternalRTPPayload(packet.Packet.Payload, packet.AfterLoss)
}

// DecodeRTPPayloadAfterLoss clears retained RTP fragment bytes, then decodes
// one caller-supplied AV1 RTP payload body. Parser sequence and reference state
// are preserved. Use
// this after the jitter buffer detects a packet gap before feeding the next
// payload that should restart depacketization.
func (d *Decoder) DecodeRTPPayloadAfterLoss(payload []byte) ([]*Frame, error) {
	return d.decodeExternalRTPPayload(payload, true)
}

// DecodeRTPPacketAfterLoss clears retained RTP fragment bytes, then parses one
// complete RTP packet and decodes its AV1 RTP payload body. Parser sequence and
// reference state are preserved.
func (d *Decoder) DecodeRTPPacketAfterLoss(packet []byte) ([]*Frame, error) {
	rtp, err := ParseRTPPacket(packet)
	if err != nil {
		if d != nil {
			internalthreading.InvalidateFrameWorkLoopFilterCompactState(&d.compactLF)
		}
		return nil, fmt.Errorf("goav1: rtp packet: %w", err)
	}
	return d.decodeExternalRTPPayload(rtp.Payload, true)
}

func (d *Decoder) decodeExternalRTPPayload(payload []byte, afterLoss bool) (frames []*Frame, decodeErr error) {
	defer func() {
		if decodeErr != nil && d != nil {
			internalthreading.InvalidateFrameWorkLoopFilterCompactState(&d.compactLF)
		}
	}()
	if d == nil || d.payloadKind != decoderPayloadRTP {
		return nil, errors.New("goav1: decoder is not initialized for RTP payloads")
	}
	if d.closed {
		return nil, errors.New("goav1: decoder closed")
	}
	d.releaseShownSurfaces()
	var result decoderFrameWorkResidualStreamResult
	postFilter := d.postFilterRunner()
	var err error
	if afterLoss {
		err = d.runner.RunRTPPayloadAfterLossIntoWithPostFilterRunner(&result, payload, postFilter)
	} else {
		err = d.runner.RunRTPPayloadIntoWithPostFilterRunner(&result, payload, postFilter)
	}
	if err != nil {
		if d.errorConcealment {
			return d.recoverAndConceal(), nil
		}
		return nil, fmt.Errorf("goav1: rtp payload: %w", err)
	}
	out := d.visible[:0]
	for _, f := range result.Run.Outputs {
		if f != nil {
			f.Concealed = false
			out = append(out, f)
		}
	}
	if d.errorConcealment && len(out) > 0 {
		d.snapshotConcealFrame(out[len(out)-1])
	}
	d.visible = out
	d.trackShownSurfaces(out)
	return out, nil
}

// recoverAndConceal cleans up partial frame-work state after a decode error and
// returns a freeze-frame copy of the last successfully decoded picture marked
// with Concealed=true (or an empty slice if no good picture has been decoded
// yet).
func (d *Decoder) recoverAndConceal() []*Frame {
	internalthreading.InvalidateFrameWorkLoopFilterCompactState(&d.compactLF)
	if d.state.Active() {
		_ = d.state.Abort(&d.pool)
	}
	if d.payloadKind == decoderPayloadRTP {
		_ = d.runner.ResetRTP()
	}
	if !d.hasConcealFrame {
		d.visible = d.visible[:0]
		return d.visible
	}
	d.concealFrame.Concealed = true
	d.visibleOne[0] = &d.concealFrame
	d.visible = d.visibleOne[:1]
	return d.visible
}

// snapshotConcealFrame copies f's visible planes into d.concealFrame, reusing
// d.concealBuf so subsequent good frames do not allocate.
func (d *Decoder) snapshotConcealFrame(f *Frame) {
	if f == nil {
		return
	}
	bps := f.Layout.BytesPerSample
	if bps < 1 {
		bps = 1
	}
	yBytes := f.Y.Width * bps * f.Y.Height
	uBytes := f.U.Width * bps * f.U.Height
	vBytes := f.V.Width * bps * f.V.Height
	total := yBytes + uBytes + vBytes
	if cap(d.concealBuf) < total {
		d.concealBuf = make([]byte, total)
	}
	buf := d.concealBuf[:total]
	copyPackedPlane := func(dst []byte, p FramePlane) FramePlane {
		if p.Width == 0 || p.Height == 0 || len(p.Pix) == 0 {
			return FramePlane{}
		}
		rowBytes := p.Width * bps
		for row := 0; row < p.Height; row++ {
			off := row * p.Stride
			copy(dst[row*rowBytes:(row+1)*rowBytes], p.Pix[off:off+rowBytes])
		}
		return FramePlane{
			Pix:    dst[:rowBytes*p.Height],
			Stride: rowBytes,
			Width:  p.Width,
			Height: p.Height,
		}
	}
	d.concealFrame = Frame{
		Format:    f.Format,
		Layout:    f.Layout,
		Y:         copyPackedPlane(buf[:yBytes], f.Y),
		U:         copyPackedPlane(buf[yBytes:yBytes+uBytes], f.U),
		V:         copyPackedPlane(buf[yBytes+uBytes:total], f.V),
		Color:     f.Color,
		Metadata:  f.Metadata,
		Concealed: false,
	}
	d.hasConcealFrame = true
}

func (d *Decoder) postFilterRunner() internaldecoder.FrameWorkPostFilterRunner {
	if d.useExternal {
		return &d.external
	}
	return &d.postFilter
}

// outputSource maps a returned frame to the pool surface behind it: a
// film-grain output is a detached display picture over its reconstructed
// source.
func (d *Decoder) outputSource(f *Frame) *Frame {
	if s := d.postFilter.postFilterSource(f); s != f {
		return s
	}
	if d.fp != nil {
		return d.fp.outputSource(f)
	}
	return f
}

// trackShownSurfaces records the pool slots behind one returned batch so the
// next call can release the ones no reference slot holds.
func (d *Decoder) trackShownSurfaces(out []*Frame) {
	d.shownHeld = d.shownHeldBuf[:0]
	d.trackShownSurfacesInPool(out, &d.pool, 0)
	if d.useExternal {
		d.trackShownSurfacesInPool(out, &d.outputPool, decoderExternalOutputSurfaceBase)
	}
}

func (d *Decoder) trackShownSurfacesInPool(out []*Frame, pool *internalframe.Pool, surfaceBase int) {
	if pool == nil {
		return
	}
	for _, f := range out {
		// A film-grain output is a detached view; the pool surface behind it
		// is the reconstructed picture, which must be released like any other
		// shown surface.
		f = d.outputSource(f)
		for i := 0; i < pool.Cap(); i++ {
			pf, err := pool.Frame(i)
			if err != nil || pf != f {
				continue
			}
			dup := false
			for _, h := range d.shownHeld {
				if h.pool == pool && h.index == i {
					dup = true
					break
				}
			}
			if !dup && len(d.shownHeld) < cap(d.shownHeldBuf) {
				d.shownHeld = append(d.shownHeld, shownSurface{pool: pool, index: i, surface: surfaceBase + i})
			}
		}
	}
}

// releaseShownSurfaces returns the previous batch's surfaces to their pools
// unless a reference slot still holds them - those are released by the slot
// bookkeeping when they leave the reference set.
func (d *Decoder) releaseShownSurfaces() {
	for _, h := range d.shownHeld {
		held := false
		for r := 0; r < internalparser.RefFrames; r++ {
			if surface, ok := d.refs.ReferenceSlot(r); ok && surface == h.surface {
				held = true
				break
			}
		}
		if !held && d.fp != nil && d.fp.deferredRelease(h.surface) {
			// An in-flight frame dropped it from the reference slots and
			// returns it to the pool when it retires.
			held = true
		}
		if !held {
			// Tolerate already-released slots: a surface freed by the
			// reference bookkeeping between calls is exactly the benign
			// case.
			_ = h.pool.Release(h.index)
		}
	}
	d.shownHeld = d.shownHeld[:0]
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
	if d == nil || d.payloadSource.len() == 0 {
		return nil, errors.New("goav1: decoder is not initialized")
	}
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
	if d == nil || d.payloadSource.len() == 0 {
		return errors.New("goav1: decoder is not initialized")
	}
	if d.closed {
		return errors.New("goav1: decoder closed")
	}
	if d.fp != nil {
		d.fp.reset()
	}
	d.pool.Reset()
	if d.useExternal {
		d.outputPool.Reset()
	}
	d.shownHeld = d.shownHeld[:0]
	d.refs.Reset()
	d.state.Reset()
	internalthreading.DeactivateFrameWorkLoopFilterCompactState(&d.compactLF)
	d.stats = internalthreading.FrameWorkTileResidualStats{}
	if err := d.runner.Reset(); err != nil {
		return fmt.Errorf("goav1: runner reset: %w", err)
	}
	d.hasConcealFrame = false
	d.next = 0
	return nil
}

// Close releases the worker goroutine pool owned by the Decoder. After Close,
// the Decoder must not be used. Close is idempotent.
func (d *Decoder) Close() {
	if d == nil {
		return
	}
	if d.closed {
		return
	}
	d.closed = true
	internalthreading.DeactivateFrameWorkLoopFilterCompactState(&d.compactLF)
	if d.fp != nil {
		d.fp.close()
	}
	if d.workerPool != nil {
		d.workerPool.Close()
		d.workerPool = nil
	}
	d.postFilterParallel.Close()
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
	Color          ColorInfo
	Metadata       FrameMetadata
	Concealed      bool
}

// copyDecodedFrame extracts the visible samples of an aliasing *Frame into an
// independent DecodedFrame, stripping per-plane stride padding so the byte
// layout matches the canonical I420/I400 raw video the libaom goldens use.
func copyDecodedFrame(f *Frame) DecodedFrame {
	bps := f.Layout.BytesPerSample
	meta := f.Metadata
	if len(meta.ITUTT35) > 0 {
		copied := make([]MetadataITUTT35, len(meta.ITUTT35))
		for i, m := range meta.ITUTT35 {
			if len(m.Payload) > 0 {
				m.Payload = append([]byte(nil), m.Payload...)
			}
			copied[i] = m
		}
		meta.ITUTT35 = copied
	}
	df := DecodedFrame{
		Width:          f.Y.Width,
		Height:         f.Y.Height,
		BytesPerSample: bps,
		ChromaWidth:    f.U.Width,
		ChromaHeight:   f.U.Height,
		Y:              copyPlane(f.Y, bps),
		U:              copyPlane(f.U, bps),
		V:              copyPlane(f.V, bps),
		Color:          f.Color,
		Metadata:       meta,
		Concealed:      f.Concealed,
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
func newDecoderFramePool(format FrameFormat, count int, arena *decoderArena) (internalframe.Pool, error) {
	_, backingSize, err := framePoolRequiredSize(format, count)
	if err != nil {
		return internalframe.Pool{}, err
	}
	return internalframe.BindPool(arena.takeBytes(backingSize), format,
		make([]Frame, count), arena.takeInts(count), arena.takeBools(count))
}

// newDecoderStreamScratch allocates every scratch arena the residual stream
// runner needs, sized from the probed plan. It mirrors the binding performed by
// cmd/aom-go-dec and the conformance harness so the convenience path is the
// same byte-exact path.
func newDecoderStreamScratch(size decoderFrameWorkResidualStreamScratchSize, arena *decoderArena) decoderFrameWorkResidualStreamScratch {
	return decoderFrameWorkResidualStreamScratch{
		Events:    make([]internaldecoder.Event, size.Events),
		Event:     newDecoderEventScratch(size.Event, arena),
		SideData:  newDecoderSideDataScratch(size.Event.SideData, arena),
		Outputs:   make([]*Frame, size.Event.Outputs),
		RTPBuffer: arena.takeBytes(size.RTPBuffer),
		RTPSpans:  make([]RTPObuSpan, size.RTPSpans),
	}
}

func newDecoderEventScratch(size decoderFrameWorkResidualEventScratchSize, arena *decoderArena) decoderFrameWorkResidualEventScratch {
	return decoderFrameWorkResidualEventScratch{
		Runner:   newDecoderBatchRunnerScratch(size.Runner, arena),
		SideData: newDecoderSideDataScratch(size.SideData, arena),
		Spans:    make([]internalparser.TileSpan, size.Plan.SpanCount),
		Jobs:     make([]internaltile.Job, size.Plan.JobCount),
		Batches:  make([]internalthreading.Batch, size.Plan.BatchCount),
	}
}

func newDecoderBatchRunnerScratch(size decoderFrameWorkBatchResidualRunnerScratchSize, arena *decoderArena) decoderFrameWorkBatchResidualRunnerScratch {
	return decoderFrameWorkBatchResidualRunnerScratch{
		States:                  make([]internaltile.DecodeState, size.Workers),
		Storages:                make([]internalthreading.FrameWorkTileResidualCDFStorage, size.Workers),
		TileScratch:             make([]internalthreading.FrameWorkTileResidualScratch, size.Workers),
		RestorationRequests:     make([]internalthreading.FrameWorkTileRestorationRequest, size.RestorationRequests),
		PredictionScratch:       make([]internalthreading.FrameWorkPredictionScratch, size.Workers),
		InterPredictionScratch:  make([]internalthreading.FrameWorkInterPredictionScratch, size.Workers),
		Stats:                   make([]internalthreading.FrameWorkTileResidualStats, size.Workers),
		Int32Scratch:            arena.takeInt32s(size.Int32Scratch),
		ResidualScratch:         arena.takeInt16s(size.ResidualScratch),
		LoopContextAboveScratch: make([]internaltile.BlockLoopRootAboveContext, size.LoopContextAbove),
	}
}

func newDecoderSideDataScratch(size decoderFrameWorkSideDataScratchSize, arena *decoderArena) decoderFrameWorkSideDataScratch {
	return decoderFrameWorkSideDataScratch{
		CDEFIndexMap:             arena.takeUint8s(size.CDEFIndexMap),
		CDEFReadMap:              arena.takeBools(size.CDEFReadMap),
		LoopFilterMap:            make([]internalthreading.FrameWorkLoopFilterBlockRecord, size.LoopFilterMap),
		LoopFilterMasks:          make([]internallfmask.FilterMask, size.LoopFilterMasks),
		LoopFilterLevelCache:     make([]uint8, size.LoopFilterLevelCache),
		RestorationRecords:       make([]internaltile.RestorationUnitRecord, size.RestorationRecords),
		RestorationBoundaryAbove: arena.takeUint16s(size.RestorationBoundaryAbove),
		RestorationBoundaryBelow: arena.takeUint16s(size.RestorationBoundaryBelow),
	}
}

type decoderArenaSize struct {
	bytes   int
	uint16s int
	int16s  int
	int32s  int
	ints    int
	bools   int
}

func decoderArenaSizeFor(format FrameFormat, outputFormat FrameFormat, useExternal bool, surfaceCount int, stream decoderFrameWorkResidualStreamScratchSize, post decoderFrameWorkPostFilterRequestScratchSize) (decoderArenaSize, error) {
	var size decoderArenaSize
	if err := size.addFramePool(format, surfaceCount); err != nil {
		return decoderArenaSize{}, err
	}
	if useExternal {
		if err := size.addFramePool(outputFormat, surfaceCount); err != nil {
			return decoderArenaSize{}, err
		}
	}
	if err := size.addStream(stream); err != nil {
		return decoderArenaSize{}, err
	}
	noSkipBytes, ok := decoderArenaMultiply(stream.Event.SideData.CDEFIndexMap, 8)
	if !ok {
		return decoderArenaSize{}, internalframe.ErrShortBuffer
	}
	if err := size.addBytes(noSkipBytes); err != nil {
		return decoderArenaSize{}, err
	}
	// dav1d noskip_mask and tile-edge backups for the non-compact paths,
	// preallocated so a cold first frame stays allocation-free.
	if err := size.addBytes(noSkipBytes); err != nil {
		return decoderArenaSize{}, err
	}
	if err := size.addBytes(decoderLoopFilterTileEdgeBytes(stream.Event.SideData.LoopFilterMasks)); err != nil {
		return decoderArenaSize{}, err
	}
	postCopies := 1
	if useExternal {
		postCopies = 2
	}
	for range postCopies {
		if err := size.addPostFilter(post); err != nil {
			return decoderArenaSize{}, err
		}
	}
	return size, nil
}

// decoderLoopFilterTileEdgeBytes bounds the four tile-edge backup buffers:
// tile columns never exceed two per 128-pixel region column (64-pixel
// superblocks) and a region holds 32 MI rows, so tileCols*rows (and likewise
// tileRows*cols, and the chroma counterparts) is at most 64 per region.
func decoderLoopFilterTileEdgeBytes(regions int) int {
	n, ok := decoderArenaMultiply(regions, 4*64)
	if !ok {
		return 0
	}
	return n
}

func decoderCompactNoSkipBytes(cdefUnits int) int {
	n, ok := decoderArenaMultiply(cdefUnits, 8)
	if !ok {
		return 0
	}
	return n
}

func decoderArenaMultiply(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if b != 0 && a > maxInt/b {
		return 0, false
	}
	return a * b, true
}

func (s *decoderArenaSize) addFramePool(format FrameFormat, count int) error {
	_, backingSize, err := framePoolRequiredSize(format, count)
	if err != nil {
		return err
	}
	if err := s.addBytes(backingSize); err != nil {
		return err
	}
	if err := s.addInts(count); err != nil {
		return err
	}
	return s.addBools(count)
}

func (s *decoderArenaSize) addStream(size decoderFrameWorkResidualStreamScratchSize) error {
	if err := s.addBytes(size.RTPBuffer); err != nil {
		return err
	}
	if err := s.addEvent(size.Event); err != nil {
		return err
	}
	return s.addSideData(size.Event.SideData)
}

func (s *decoderArenaSize) addEvent(size decoderFrameWorkResidualEventScratchSize) error {
	if err := s.addBatchRunner(size.Runner); err != nil {
		return err
	}
	return s.addSideData(size.SideData)
}

func (s *decoderArenaSize) addBatchRunner(size decoderFrameWorkBatchResidualRunnerScratchSize) error {
	if err := s.addInt32s(size.Int32Scratch); err != nil {
		return err
	}
	return s.addInt16s(size.ResidualScratch)
}

func (s *decoderArenaSize) addSideData(size decoderFrameWorkSideDataScratchSize) error {
	if err := s.addBytes(size.CDEFIndexMap); err != nil {
		return err
	}
	if err := s.addBools(size.CDEFReadMap); err != nil {
		return err
	}
	totalBoundary, ok := decoderArenaAdd2(size.RestorationBoundaryAbove, size.RestorationBoundaryBelow)
	if !ok {
		return internalframe.ErrShortBuffer
	}
	return s.addUint16s(totalBoundary)
}

func (s *decoderArenaSize) addPostFilter(size decoderFrameWorkPostFilterRequestScratchSize) error {
	if err := s.addBytes(size.ByteScratch); err != nil {
		return err
	}
	if err := s.addUint16s(size.Uint16Scratch); err != nil {
		return err
	}
	if err := s.addInt16s(size.Int16Scratch); err != nil {
		return err
	}
	return s.addInt32s(size.Int32Scratch)
}

func (s *decoderArenaSize) addBytes(n int) error   { return decoderArenaAdd(&s.bytes, n) }
func (s *decoderArenaSize) addUint16s(n int) error { return decoderArenaAdd(&s.uint16s, n) }
func (s *decoderArenaSize) addInt16s(n int) error  { return decoderArenaAdd(&s.int16s, n) }
func (s *decoderArenaSize) addInt32s(n int) error  { return decoderArenaAdd(&s.int32s, n) }
func (s *decoderArenaSize) addInts(n int) error    { return decoderArenaAdd(&s.ints, n) }
func (s *decoderArenaSize) addBools(n int) error   { return decoderArenaAdd(&s.bools, n) }

func decoderArenaAdd(dst *int, n int) error {
	if n < 0 {
		return internalframe.ErrShortBuffer
	}
	sum, ok := decoderArenaAdd2(*dst, n)
	if !ok {
		return internalframe.ErrShortBuffer
	}
	*dst = sum
	return nil
}

func decoderArenaAdd2(a int, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if a > maxInt-b {
		return 0, false
	}
	return a + b, true
}

type decoderArena struct {
	bytes   []byte
	uint16s []uint16
	int16s  []int16
	int32s  []int32
	ints    []int
	bools   []bool
}

// newDecoderArena backs each typed region with one zeroed allocation per
// element type in use.
func newDecoderArena(size decoderArenaSize) decoderArena {
	var a decoderArena
	if size.ints > 0 {
		a.ints = make([]int, size.ints)
	}
	if size.int32s > 0 {
		a.int32s = make([]int32, size.int32s)
	}
	if size.uint16s > 0 {
		a.uint16s = make([]uint16, size.uint16s)
	}
	if size.int16s > 0 {
		a.int16s = make([]int16, size.int16s)
	}
	if size.bytes > 0 {
		a.bytes = make([]byte, size.bytes)
	}
	if size.bools > 0 {
		a.bools = make([]bool, size.bools)
	}
	return a
}

func (a *decoderArena) takeBytes(n int) []byte {
	if n == 0 {
		return nil
	}
	out := a.bytes[:n]
	a.bytes = a.bytes[n:]
	return out
}

func (a *decoderArena) takeUint8s(n int) []uint8 {
	return a.takeBytes(n)
}

func (a *decoderArena) takeUint16s(n int) []uint16 {
	if n == 0 {
		return nil
	}
	out := a.uint16s[:n]
	a.uint16s = a.uint16s[n:]
	return out
}

func (a *decoderArena) takeInt16s(n int) []int16 {
	if n == 0 {
		return nil
	}
	out := a.int16s[:n]
	a.int16s = a.int16s[n:]
	return out
}

func (a *decoderArena) takeInt32s(n int) []int32 {
	if n == 0 {
		return nil
	}
	out := a.int32s[:n]
	a.int32s = a.int32s[n:]
	return out
}

func (a *decoderArena) takeInts(n int) []int {
	if n == 0 {
		return nil
	}
	out := a.ints[:n]
	a.ints = a.ints[n:]
	return out
}

func (a *decoderArena) takeBools(n int) []bool {
	if n == 0 {
		return nil
	}
	out := a.bools[:n]
	a.bools = a.bools[n:]
	return out
}

func decoderPostFilterScratchFromArena(size decoderFrameWorkPostFilterRequestScratchSize, arena *decoderArena) decoderFrameWorkPostFilterRequestScratch {
	return decoderFrameWorkPostFilterRequestScratch{
		LoopFilterEdges:    make([]internaldecoder.FrameWorkLoopFilterPostFilterEdge, size.LoopFilterEdges),
		LoopFilterSchedule: make([]uint32, size.LoopFilterSchedule),
		CDEFDirectionGrid:  make([]internalcdef.DirectionGrid, size.CDEFDirectionGrid),
		CDEFVarianceGrid:   make([]internalcdef.VarianceGrid, size.CDEFVarianceGrid),
		ByteScratch:        arena.takeBytes(size.ByteScratch),
		Uint16Scratch:      arena.takeUint16s(size.Uint16Scratch),
		Int16Scratch:       arena.takeInt16s(size.Int16Scratch),
		Int32Scratch:       arena.takeInt32s(size.Int32Scratch),
	}
}
