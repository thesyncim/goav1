package goav1

import (
	"errors"
	"fmt"

	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
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

	refSurface [InterRefsPerFrame]int
	refFrames  [InterRefsPerFrame]*Frame
	releases   [RefFrames]int

	scratch    DecoderFrameWorkResidualStreamScratch
	runner     DecoderFrameWorkResidualStreamRunner
	postFilter DecoderFrameWorkReusableSupportedPostFilterRunner
	external   decoderExternalPostFilterRunner

	// postFilterParallel fans the post-filter chain's independent row bands out
	// across the same worker count as tile decode. It is reused across frames so
	// the parallel path stays allocation-free after warm-up.
	postFilterParallel DecoderFrameWorkPostFilterParallel

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
	shownHeldBuf [2 * (RefFrames + 1)]shownSurface
}

// shownSurface names one tracked output surface and the pool that owns it.
type shownSurface struct {
	pool    *FramePool
	index   int
	surface int
}

type decoderPayloadKind uint8

const (
	decoderPayloadLowOverhead decoderPayloadKind = iota
	decoderPayloadRTP
)

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
	if source.len() == 0 {
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
	probeEventBudget := source.len()*eventsPerFrame + 64
	probeEvents := make([]DecoderEvent, probeEventBudget)
	probeSpans := make([]TileSpan, MaxTiles)
	probeJobs := make([]TileJob, MaxTiles)
	probeBatches := make([]TileBatch, MaxTiles)
	probePayload := source.makeProbeBuffer()

	var plan DecoderFrameWorkResidualStreamPlan
	switch kind {
	case decoderPayloadLowOverhead:
		plan, err = decoderFrameWorkResidualLowOverheadStreamsPlanFromSource(
			probeStream, source, cfg.workers, probePayload,
			probeEvents, probeSpans, probeJobs, probeBatches,
		)
	case decoderPayloadRTP:
		plan, err = decoderFrameWorkResidualRTPPayloadsStreamPlanFromSource(
			probeStream, source, cfg.workers, probeEvents, probeSpans, probeJobs, probeBatches,
		)
	default:
		err = ErrDecoderInvalidFrameWorkState
	}
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
	var useExternal bool
	var outputFormat FrameFormat
	switch kind {
	case decoderPayloadLowOverhead:
		useExternal, outputFormat, err = decoderExternalOutputFormatFromSource(source, 64, probePayload)
	case decoderPayloadRTP:
		useExternal, outputFormat, err = decoderExternalOutputFormatFromRTPSource(source, 64)
	}
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: super-res output format: %w", err)
	}
	var motionSize FrameSize
	switch kind {
	case decoderPayloadLowOverhead:
		motionSize, err = decoderTemporalMotionScratchFrameSizeUpperBoundFromSource(source, probeEvents, probePayload)
	case decoderPayloadRTP:
		motionSize, err = decoderTemporalMotionScratchFrameSizeUpperBoundFromRTPSource(source)
	}
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: temporal motion scratch plan: %w", err)
	}
	var postArena DecoderFrameWorkPostFilterRequestScratchSize
	switch kind {
	case decoderPayloadLowOverhead:
		postArena, err = decoderPostFilterScratchArenaUpperBoundFromSource(source, 64, probeEvents, probePayload)
	case decoderPayloadRTP:
		postArena, err = decoderPostFilterScratchArenaUpperBoundFromRTPSource(source, 64)
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
	var outputPool FramePool
	if useExternal {
		outputPool, err = newDecoderFramePool(outputFormat, surfaceCount, &arena)
		if err != nil {
			workerPool.Close()
			return nil, fmt.Errorf("goav1: output frame pool: %w", err)
		}
	}

	d := &Decoder{
		pool:          pool,
		outputPool:    outputPool,
		workerPool:    workerPool,
		payloadKind:   kind,
		payloadSource: source,
		payloadBuf:    arena.takeBytes(source.payloadBufferLen()),
		format:        format,
		useExternal:   useExternal,
	}
	if d.useExternal {
		d.external.outputPool = &d.outputPool
	}
	d.scratch = newDecoderStreamScratch(plan.Size, &arena)
	if cap(d.scratch.Outputs) > 0 {
		d.visible = d.scratch.Outputs[:0]
	} else {
		d.visible = d.visibleOne[:0]
	}
	if err := d.state.PreallocTemporalMotionScratch(motionSize); err != nil {
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
	internaldecoder.BindFrameWorkPostFilterParallelPool(&d.postFilterParallel, d.workerPool)
	d.postFilter.Parallel = &d.postFilterParallel

	runtime := DecoderFrameWorkResidualEventRuntime{
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
		runtime.External = DecoderFrameWorkExternalReferenceRuntime{
			Provider:      provider,
			GlobalSurface: func(local int) int { return local },
			Releaser:      provider,
			PostPublisher: &d.external,
		}
	}
	runner, _, err := BindDecoderFrameWorkResidualStreamPlanRunner(plan, &d.stream,
		runtime, d.scratch, &d.batch)
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: bind runner: %w", err)
	}
	d.runner = runner
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

// DecodeNext decodes the next payload and returns the visible frames it
// produced, in display order. For NewDecoder/NewDecoderFromIVF this payload is
// one low-overhead temporal unit; for NewDecoderFromRTPPayloads it is one AV1
// RTP payload body. Most low-overhead payloads yield exactly one frame, but a
// payload may yield zero (e.g. a hidden/no-show frame or an RTP fragment before
// the frame completes) or more than one (e.g. a show-existing-frame following a
// coded frame). When all payloads have been consumed it returns
// (nil, false, nil); subsequent calls keep returning that until Reset.
//
// The returned *Frame values alias the Decoder's reused output arena and remain
// valid only until the next DecodeNext / DecodeAll call or Close. See the
// Decoder type documentation for full lifetime semantics.
func (d *Decoder) DecodeNext() (frames []*Frame, ok bool, err error) {
	if d == nil || d.payloadSource.len() == 0 {
		return nil, false, errors.New("goav1: decoder is not initialized")
	}
	if d.closed {
		return nil, false, errors.New("goav1: decoder closed")
	}
	d.releaseShownSurfaces()
	if d.next >= d.payloadSource.len() {
		return nil, false, nil
	}
	i := d.next
	d.next++
	payload, err := d.payloadSource.payloadAt(i, d.payloadBuf)
	if err != nil {
		return nil, false, fmt.Errorf("goav1: frame %d payload: %w", i, err)
	}

	var result DecoderFrameWorkResidualStreamResult
	postFilter := d.postFilterRunner()
	switch d.payloadKind {
	case decoderPayloadLowOverhead:
		err = d.runner.RunLowOverheadIntoWithPostFilterRunner(&result, payload, postFilter)
	case decoderPayloadRTP:
		err = d.runner.RunRTPPayloadIntoWithPostFilterRunner(&result, payload, postFilter)
	default:
		err = ErrDecoderInvalidFrameWorkState
	}
	if err != nil {
		return nil, false, fmt.Errorf("goav1: payload %d: %w", i, err)
	}

	out := d.visible[:0]
	for _, f := range result.Run.Outputs {
		if f != nil {
			out = append(out, f)
		}
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
// are preserved, matching the lower-level RunRTPPayloadAfterLoss* methods. Use
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
		return nil, fmt.Errorf("goav1: rtp packet: %w", err)
	}
	return d.decodeExternalRTPPayload(rtp.Payload, true)
}

func (d *Decoder) decodeExternalRTPPayload(payload []byte, afterLoss bool) ([]*Frame, error) {
	if d == nil || d.payloadKind != decoderPayloadRTP {
		return nil, errors.New("goav1: decoder is not initialized for RTP payloads")
	}
	if d.closed {
		return nil, errors.New("goav1: decoder closed")
	}
	d.releaseShownSurfaces()
	var result DecoderFrameWorkResidualStreamResult
	postFilter := d.postFilterRunner()
	var err error
	if afterLoss {
		err = d.runner.RunRTPPayloadAfterLossIntoWithPostFilterRunner(&result, payload, postFilter)
	} else {
		err = d.runner.RunRTPPayloadIntoWithPostFilterRunner(&result, payload, postFilter)
	}
	if err != nil {
		return nil, fmt.Errorf("goav1: rtp payload: %w", err)
	}
	out := d.visible[:0]
	for _, f := range result.Run.Outputs {
		if f != nil {
			out = append(out, f)
		}
	}
	d.visible = out
	d.trackShownSurfaces(out)
	return out, nil
}

func (d *Decoder) postFilterRunner() DecoderFrameWorkPostFilterRunner {
	if d.useExternal {
		return &d.external
	}
	return &d.postFilter
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

func (d *Decoder) trackShownSurfacesInPool(out []*Frame, pool *FramePool, surfaceBase int) {
	if pool == nil {
		return
	}
	for _, f := range out {
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
		for r := 0; r < RefFrames; r++ {
			if surface, ok := d.refs.ReferenceSlot(r); ok && surface == h.surface {
				held = true
				break
			}
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
	d.pool.Reset()
	if d.useExternal {
		d.outputPool.Reset()
	}
	d.shownHeld = d.shownHeld[:0]
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
	if d == nil {
		return
	}
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
func newDecoderFramePool(format FrameFormat, count int, arena *decoderArena) (FramePool, error) {
	_, backingSize, err := FramePoolRequiredSize(format, count)
	if err != nil {
		return FramePool{}, err
	}
	return BindFramePool(arena.takeBytes(backingSize), format,
		make([]Frame, count), arena.takeInts(count), arena.takeBools(count))
}

// newDecoderStreamScratch allocates every scratch arena the residual stream
// runner needs, sized from the probed plan. It mirrors the binding performed by
// cmd/aom-go-dec and the conformance harness so the convenience path is the
// same byte-exact path.
func newDecoderStreamScratch(size DecoderFrameWorkResidualStreamScratchSize, arena *decoderArena) DecoderFrameWorkResidualStreamScratch {
	return DecoderFrameWorkResidualStreamScratch{
		Events:    make([]DecoderEvent, size.Events),
		Event:     newDecoderEventScratch(size.Event, arena),
		SideData:  newDecoderSideDataScratch(size.Event.SideData, arena),
		Outputs:   make([]*Frame, size.Event.Outputs),
		RTPBuffer: arena.takeBytes(size.RTPBuffer),
		RTPSpans:  make([]RTPObuSpan, size.RTPSpans),
	}
}

func newDecoderEventScratch(size DecoderFrameWorkResidualEventScratchSize, arena *decoderArena) DecoderFrameWorkResidualEventScratch {
	return DecoderFrameWorkResidualEventScratch{
		Runner:   newDecoderBatchRunnerScratch(size.Runner, arena),
		SideData: newDecoderSideDataScratch(size.SideData, arena),
		Spans:    make([]TileSpan, size.Plan.SpanCount),
		Jobs:     make([]TileJob, size.Plan.JobCount),
		Batches:  make([]TileBatch, size.Plan.BatchCount),
	}
}

func newDecoderBatchRunnerScratch(size DecoderFrameWorkBatchResidualRunnerScratchSize, arena *decoderArena) DecoderFrameWorkBatchResidualRunnerScratch {
	return DecoderFrameWorkBatchResidualRunnerScratch{
		States:                  make([]TileDecodeState, size.Workers),
		Storages:                make([]DecoderFrameWorkTileResidualCDFStorage, size.Workers),
		TileScratch:             make([]DecoderFrameWorkTileResidualScratch, size.Workers),
		RestorationRequests:     make([]DecoderFrameWorkTileRestorationRequest, size.RestorationRequests),
		PredictionScratch:       make([]DecoderFrameWorkPredictionScratch, size.Workers),
		InterPredictionScratch:  make([]DecoderFrameWorkInterPredictionScratch, size.Workers),
		Stats:                   make([]DecoderFrameWorkTileResidualStats, size.Workers),
		Int32Scratch:            arena.takeInt32s(size.Int32Scratch),
		ResidualScratch:         arena.takeInt16s(size.ResidualScratch),
		LoopContextAboveScratch: make([]TileBlockLoopRootAboveContext, size.LoopContextAbove),
	}
}

func newDecoderSideDataScratch(size DecoderFrameWorkSideDataScratchSize, arena *decoderArena) DecoderFrameWorkSideDataScratch {
	return DecoderFrameWorkSideDataScratch{
		CDEFIndexMap:             arena.takeUint8s(size.CDEFIndexMap),
		CDEFReadMap:              arena.takeBools(size.CDEFReadMap),
		LoopFilterMap:            make([]DecoderFrameWorkLoopFilterBlockRecord, size.LoopFilterMap),
		LoopFilterMasks:          make([]DecoderFrameWorkLoopFilterFilterMask, size.LoopFilterMasks),
		LoopFilterLevelCache:     make([][4]uint8, size.LoopFilterLevelCache),
		RestorationRecords:       make([]TileRestorationUnitRecord, size.RestorationRecords),
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

func decoderArenaSizeFor(format FrameFormat, outputFormat FrameFormat, useExternal bool, surfaceCount int, stream DecoderFrameWorkResidualStreamScratchSize, post DecoderFrameWorkPostFilterRequestScratchSize) (decoderArenaSize, error) {
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

func (s *decoderArenaSize) addFramePool(format FrameFormat, count int) error {
	_, backingSize, err := FramePoolRequiredSize(format, count)
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

func (s *decoderArenaSize) addStream(size DecoderFrameWorkResidualStreamScratchSize) error {
	if err := s.addBytes(size.RTPBuffer); err != nil {
		return err
	}
	if err := s.addEvent(size.Event); err != nil {
		return err
	}
	return s.addSideData(size.Event.SideData)
}

func (s *decoderArenaSize) addEvent(size DecoderFrameWorkResidualEventScratchSize) error {
	if err := s.addBatchRunner(size.Runner); err != nil {
		return err
	}
	return s.addSideData(size.SideData)
}

func (s *decoderArenaSize) addBatchRunner(size DecoderFrameWorkBatchResidualRunnerScratchSize) error {
	if err := s.addInt32s(size.Int32Scratch); err != nil {
		return err
	}
	return s.addInt16s(size.ResidualScratch)
}

func (s *decoderArenaSize) addSideData(size DecoderFrameWorkSideDataScratchSize) error {
	if err := s.addBytes(size.CDEFIndexMap); err != nil {
		return err
	}
	if err := s.addBools(size.CDEFReadMap); err != nil {
		return err
	}
	totalBoundary, ok := decoderArenaAdd2(size.RestorationBoundaryAbove, size.RestorationBoundaryBelow)
	if !ok {
		return ErrFrameShortBuffer
	}
	return s.addUint16s(totalBoundary)
}

func (s *decoderArenaSize) addPostFilter(size DecoderFrameWorkPostFilterRequestScratchSize) error {
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
		return ErrFrameShortBuffer
	}
	sum, ok := decoderArenaAdd2(*dst, n)
	if !ok {
		return ErrFrameShortBuffer
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

func newDecoderArena(size decoderArenaSize) decoderArena {
	return decoderArena{
		bytes:   make([]byte, size.bytes),
		uint16s: make([]uint16, size.uint16s),
		int16s:  make([]int16, size.int16s),
		int32s:  make([]int32, size.int32s),
		ints:    make([]int, size.ints),
		bools:   make([]bool, size.bools),
	}
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

func decoderPostFilterScratchFromArena(size DecoderFrameWorkPostFilterRequestScratchSize, arena *decoderArena) DecoderFrameWorkPostFilterRequestScratch {
	return DecoderFrameWorkPostFilterRequestScratch{
		LoopFilterEdges:    make([]DecoderFrameWorkLoopFilterPostFilterEdge, size.LoopFilterEdges),
		LoopFilterSchedule: make([]uint32, size.LoopFilterSchedule),
		CDEFDirectionGrid:  make([]CDEFDirectionGrid, size.CDEFDirectionGrid),
		CDEFVarianceGrid:   make([]CDEFVarianceGrid, size.CDEFVarianceGrid),
		ByteScratch:        arena.takeBytes(size.ByteScratch),
		Uint16Scratch:      arena.takeUint16s(size.Uint16Scratch),
		Int16Scratch:       arena.takeInt16s(size.Int16Scratch),
		Int32Scratch:       arena.takeInt32s(size.Int32Scratch),
	}
}
