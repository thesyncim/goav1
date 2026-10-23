package encoder

import (
	"fmt"
	"sync"

	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

// video.go is the streaming encoder surface: a VideoEncoder owns the reference
// reconstruction state and turns a sequence of source frames into a decodable
// temporal-unit stream — a keyframe first (or on demand), then
// motion-compensated P-frames chained off the previous frame's reconstruction.

// VideoEncoder encodes a stream of same-sized frames, either at a fixed base
// qindex or under closed-loop CBR rate control.
type VideoEncoder struct {
	width, height int // coded dimensions (multiples of 8)
	// renderWidth/renderHeight are the caller-facing frame dimensions; when
	// they differ from the coded size the source pads by edge replication
	// and the headers signal them through render_size.
	renderWidth, renderHeight int
	color                     SequenceColorConfig
	padded                    SourceFrame420
	keyRecon                  SourceFrame420
	qIndex                    uint8
	recon                     SourceFrame420
	haveKey                   bool

	rcEnabled      bool
	rcTargetBits   int
	rcFramesPerSec int
	rcMinQ, rcMaxQ uint8
	// rcPerFrameBits is avg_frame_bandwidth (target / frame rate).
	rcPerFrameBits int
	// svtRC is SVT-AV1's one-pass CBR rate control shared by every
	// temporal layer (svt_rc.go); svtRef holds the reference slots' RC facts
	// (base_q_idx, temporal layer, intra) for LAST and the L1T3 middle layer,
	// and svtCur / svtCurCR the picture being coded.
	svtRC    svtRateControl
	svtRef   [2]svtRCRef
	svtCur   svtRCFrame
	svtCurQ  uint8
	svtCurCR svtCRStats
	svtCurOn bool
	// cr is the cyclic refresh state of CBR streams.
	cr cyclicRefresh

	pc        pframeCoder
	reconBufs [2]SourceFrame420
	reconIdx  int

	// Tile-column parallelism: tileColsLog2 > 0 splits inter frames into
	// uniform tile columns, each encoded by its own coder (independent CDFs
	// and entropy stream per AV1 tile semantics) on its own goroutine.
	tileColsLog2 uint8
	// keyframeTileColsLog2 is the keyframe tile-column split. Keyframes have
	// no P-frame decision pass, so they cannot use the SB-row wavefront and
	// keep tile-column parallelism even when inter frames run single-tile +
	// wavefront (SetMaxThreads). SetTileColumns sets both.
	keyframeTileColsLog2 uint8
	tilePCs              []pframeCoder

	// forceSplit makes the P-frame coders run the decision/write split even
	// when it cannot parallelize. Test-only: the byte-identity oracle uses it
	// to compare the serial split against the fused walk. In production the
	// split runs only when the SB-row wavefront will (>=2 lanes).
	forceSplit bool

	// wavefront runs the split pipeline's decision pass over SB rows with the
	// top-right dependency (pframe_wavefront.go) on single-tile inter frames;
	// wavefrontLanes is its worker budget (0 disables). Only e.pc — the
	// single-tile coder — is wired to it: multi-tile frames parallelize
	// across tile columns instead.
	wavefront      pframeWavefront
	wavefrontLanes int
	lf             loopFilterApplier
	cdefApp        cdefApplier
	// cdefSearch is the SVT CDEF strength search state; cdefRefLast and
	// cdefRefT1 are the signaled strengths of the LAST and middle-layer
	// reference slots (the reference objects' ref_cdef_strengths).
	cdefSearch  cdefSearcher
	cdefRefLast svtCDEFRefStrengths
	cdefRefT1   svtCDEFRefStrengths
	cdefLogs    svtCDEFLogs
	// refPad caches the padded reference copies (refpad.go).
	refPad refPadCache
	// lazyCDEF is the deferred CDEF pass of a non-reference picture.
	lazyCDEF     lazyCDEFPass
	hme          hmeState
	singleThread bool
	effortLevel  int8

	temporalLayers int
	t2Recon        SourceFrame420
	frameCtxT1     frameCDFs
	haveCtxT1      bool
	frameIndex     int
	lastTemporalID uint8
	t1Recon        SourceFrame420
	lastRecon      SourceFrame420

	// Steady-state scratch: per-frame slices and the temporal-unit buffers
	// persist across frames so the encode path stays allocation-free. The
	// returned temporal unit aliases tuScratch and stays valid until the
	// next encode call.
	payloads      []TilePayload
	tileErrs      []error
	tuGroup       []byte
	tuScratch     []byte
	tuScratch2    []byte
	tileWork      chan tileWorkRange
	tileWait      sync.WaitGroup
	tileWorkerWG  sync.WaitGroup // persistent helpers, separate from per-job completion
	tileWorkers   int
	tileJobParams struct {
		src, refRecon  SourceFrame420
		golden         *SourceFrame420
		out            *SourceFrame420
		effQ           uint8
		prevCtx        *frameCDFs
		referenceMode  parser.ReferenceMode
		forceIntegerMV bool
		allowScreen    bool
		color          parser.ColorConfig
		tile           TileInfo
		miCols         uint16
		lfMap          *threading.FrameWorkLoopFilterMap
		payloads       []TilePayload
		errs           []error
	}

	// The in-loop filters of a finished frame run concurrently with the
	// next frame's source-only stages (padding and the coarse search);
	// filterDone carries the pass result and joinFilter is the barrier
	// every reader of the filtered reconstruction or the applier state
	// crosses first.
	filterDone     chan error
	filterWork     chan struct{}
	filterStarted  bool
	filterWorkerWG sync.WaitGroup // persistent helper, separate from per-frame completion
	filterPending  bool
	filterErr      error
	filterParams   struct {
		out     *SourceFrame420
		lfLevel uint8
		cdef    parser.CDEFParams
		cdefIdx []uint8
		// parallelMasks is snapshotted on the dispatch goroutine (which is
		// serialized with config mutation) so the detached filter worker never
		// reads e.tileColsLog2, which SetConfig may mutate concurrently.
		parallelMasks bool
	}

	// frameCtx chains the adapted symbol contexts across the base layer:
	// each non-droppable frame names its predecessor through
	// primary_ref_frame and starts from the saved state instead of the
	// defaults. haveCtx arms after the first keyframe.
	frameCtx     frameCDFs
	haveCtx      bool
	prevLFDeltas LoopFilterDeltas

	refState parser.ReferenceState

	// Golden anchor: slot 1 holds an older high-quality reference refreshed
	// every goldenEvery base-layer frames (0 disables block-level use). The
	// encoder keeps its own copy for motion search. avgFrameLowMotion is
	// libaom's rc.avg_frame_low_motion (percent of MI units coded as
	// near-zero-motion LAST blocks, EMA-smoothed); low values pull golden
	// refreshes forward per the one-pass realtime golden update.
	golden            SourceFrame420
	goldenEvery       int
	sinceGoldenFresh  int
	avgFrameLowMotion int

	// content is the screen_content_mode; sc is the sc_class of the last
	// keyframe (perform_sc_detection), which every inter frame inherits.
	content ContentHint
	sc      svtSCClass
	scDet   svtSCDetector

	decisionStatsEnabled bool
	decisionStats        EncoderDecisionStats

	// Throughput pipelining (opt-in, default off). See video_pipeline.go: the
	// EncodeThroughput/Drain surface buffers one source so a droppable L1T2 leaf
	// can overlap the following base frame, trading +1 frame of latency for wall
	// throughput. When pipeline is false the encoder is byte- and
	// latency-identical to the historical serial path.
	pipeline    bool
	pipeHeld    bool
	pipeHeldKey bool
	pipeHeldTID uint8
	pipeHeldSrc SourceFrame420
	pipeSrcBuf  [2]SourceFrame420
	pipeSrcIdx  int
	// pipeHeldDone marks the held item as a precomputed base result (its state
	// was already committed at overlap time); emit it and hold the next source.
	pipeHeldDone  bool
	pipeHeldTU    []byte
	pipeHeldRecon SourceFrame420
	pipeOverlaps  int

	// Leaf-isolated overlap state: the buffered droppable leaf runs on this
	// dedicated coder/HME/filter set (never the shared e.pc/e.hme/e.lf) so it
	// can execute concurrently with the following base frame's serial
	// encodePReusing on the caller goroutine.
	leafPC           pframeCoder
	leafWavefront    pframeWavefront
	leafHME          hmeState
	leafLF           loopFilterApplier
	leafCdefApp      cdefApplier
	leafRefState     parser.ReferenceState
	leafPrevLFDeltas LoopFilterDeltas
	leafPayloads     [1]TilePayload
	leafTuGroup      []byte
	leafTuScratch    []byte
	pipeLeafRecon    SourceFrame420
	pipeLeafCtx      frameCDFs
	// Light-PD1 per-SB statistics (pframe_lpd1_level.go): the picture being
	// coded, the LAST and T1 references' published objects (swapped on
	// refresh, like SVT's EbReferenceObject sb_* arrays), and the pipelined
	// leaf's own picture object.
	lpd1Cur, lpd1Last, lpd1T1, lpd1Leaf *lpd1Frame
	// lpd1FrameBias corrects e.frameIndex for the pipelined base, coded
	// before the counter advances past its leaf.
	lpd1FrameBias int
	// Persistent leaf worker (parked between frames; no per-frame closures).
	leafWork     chan struct{}
	leafDone     chan error
	leafStarted  bool
	leafWorkerWG sync.WaitGroup // persistent helper, separate from per-frame completion
	leafJob      struct {
		src, refRecon SourceFrame420
		prevCtx       *frameCDFs
		tu            []byte
		cntZeroMV     int
		lpd1Ref       *lpd1Frame
		cdefRef       svtCDEFRefStrengths
		frameIndex    int
	}
}

type tileWorkRange struct {
	first  int
	stride int
	limit  int
	key    bool
}

// RateControlConfig describes closed-loop CBR rate control: a target bitrate
// at a fixed frame rate, with the working qindex clamped to [MinQIndex,
// MaxQIndex].
type RateControlConfig struct {
	TargetBitsPerSecond int
	FramesPerSecond     int
	MinQIndex           uint8
	MaxQIndex           uint8
}

// NewVideoEncoder creates a streaming encoder. Render dimensions may be odd or
// non-multiple-of-eight; the stream is coded at padded dimensions and signals
// the requested render size. qIndex must be non-zero.
func NewVideoEncoder(width, height int, qIndex uint8) (*VideoEncoder, error) {
	return newVideoEncoderWithColor(width, height, qIndex, defaultVideoColorConfig())
}

func newVideoEncoderWithColor(width, height int, qIndex uint8, color SequenceColorConfig) (*VideoEncoder, error) {
	if width < 16 || height < 16 {
		return nil, fmt.Errorf("encoder: dimensions must be at least 16x16, got %dx%d", width, height)
	}
	if qIndex == 0 {
		return nil, fmt.Errorf("encoder: qindex must be non-zero")
	}
	if color.BitDepth == 0 {
		color.BitDepth = 8
	}
	if !videoColorSupported(color) {
		return nil, ErrUnsupported
	}
	if err := validateSequenceColorConfig(videoColorProfile(color), color); err != nil {
		return nil, err
	}
	// The block coders work on whole 8x8 cells; frames whose dimensions are
	// not multiples of eight encode at the padded coded size (right/bottom
	// edge replication) and signal the true size through render_size, the
	// standard coded-vs-display split.
	codedW := (width + 7) &^ 7
	codedH := (height + 7) &^ 7
	e := &VideoEncoder{
		width: codedW, height: codedH,
		renderWidth: width, renderHeight: height,
		color:  color,
		qIndex: qIndex, goldenEvery: 0,
	}
	e.tileColsLog2 = defaultTileColsLog2(codedW)
	e.keyframeTileColsLog2 = e.tileColsLog2
	return e, nil
}

// padSource copies src into the encoder's coded-size scratch frame,
// replicating the last source column and row across the padding.
func (e *VideoEncoder) padSource(src SourceFrame420) SourceFrame420 {
	color := e.parserColorConfig()
	cw, ch := chromaWidthForColor(e.width, color), chromaHeightForColor(e.height, color)
	srcCW, srcCH := chromaWidthForColor(src.Width, color), chromaHeightForColor(src.Height, color)
	if e.padded.Y == nil {
		e.padded = SourceFrame420{
			Y:            make([]byte, e.width*e.height),
			U:            make([]byte, cw*ch),
			V:            make([]byte, cw*ch),
			YStride:      e.width,
			ChromaStride: cw,
			Width:        e.width,
			Height:       e.height,
		}
	}
	padPlane := func(dst []byte, dstStride int, src []byte, srcStride, sw, sh, dw, dh int) {
		for y := 0; y < dh; y++ {
			sy := min(y, sh-1)
			drow := dst[y*dstStride : y*dstStride+dw]
			copy(drow, src[sy*srcStride:sy*srcStride+sw])
			for x := sw; x < dw; x++ {
				drow[x] = drow[sw-1]
			}
		}
	}
	padPlane(e.padded.Y, e.width, src.Y, src.YStride, src.Width, src.Height, e.width, e.height)
	padPlane(e.padded.U, cw, src.U, src.ChromaStride, srcCW, srcCH, cw, ch)
	padPlane(e.padded.V, cw, src.V, src.ChromaStride, srcCW, srcCH, cw, ch)
	return e.padded
}

// SetGoldenInterval sets how many base-layer inter frames pass between
// golden-anchor refreshes; zero disables golden references entirely.
func (e *VideoEncoder) SetGoldenInterval(n int) {
	e.goldenEvery = n
}

// SetContentHint selects SVT-AV1 screen_content_mode semantics: ContentAuto
// detects screen content on each keyframe (svt_aom_is_screen_content),
// ContentCamera and ContentScreen force it. Like SVT, the classes are derived
// at keyframes, so a change takes effect at the next keyframe.
func (e *VideoEncoder) SetContentHint(content ContentHint) error {
	if e == nil {
		return fmt.Errorf("encoder: nil video encoder")
	}
	if !content.Valid() {
		return ErrInvalidConfig
	}
	e.content = content
	return nil
}

// SetEffortLevel selects the realtime encoder effort level. The default zero
// preserves the current quality/speed balance; WebRTCMinEffortLevel disables
// subpel search for the fastest valid bitstream path.
func (e *VideoEncoder) SetEffortLevel(level int8) error {
	if e == nil {
		return fmt.Errorf("encoder: nil video encoder")
	}
	if !validWebRTCEffortLevel(level) {
		return ErrInvalidConfig
	}
	e.effortLevel = level
	e.pc.effortLevel = level
	for i := range e.tilePCs {
		e.tilePCs[i].effortLevel = level
	}
	return nil
}

// SetDecisionStatsEnabled toggles encoder decision diagnostics. The default
// is disabled; when disabled, hot block paths only see a nil stats pointer.
func (e *VideoEncoder) SetDecisionStatsEnabled(enabled bool) {
	e.decisionStatsEnabled = enabled
	e.configureDecisionStats(0)
}

// ResetDecisionStats clears the accumulated encoder decision diagnostics.
func (e *VideoEncoder) ResetDecisionStats() {
	e.decisionStats.Reset()
}

// DecisionStats returns a copy of the accumulated encoder decision diagnostics.
func (e *VideoEncoder) DecisionStats() EncoderDecisionStats {
	return e.decisionStats
}

func (e *VideoEncoder) configureDecisionStats(nTiles int) {
	e.pc.effortLevel = e.effortLevel
	e.pc.decisionStatsEnabled = e.decisionStatsEnabled
	for i := range e.tilePCs {
		if nTiles > 0 && i >= nTiles {
			break
		}
		e.tilePCs[i].effortLevel = e.effortLevel
		e.tilePCs[i].decisionStatsEnabled = e.decisionStatsEnabled
	}
}

func (e *VideoEncoder) fastestRealtimeEffort() bool {
	return e != nil && e.effortLevel <= WebRTCMinEffortLevel
}

func (e *VideoEncoder) collectDecisionStats(keyframe bool, nTiles int, keyTileZeroUsesPrimary bool) {
	if !e.decisionStatsEnabled {
		return
	}
	e.decisionStats.Frames++
	if keyframe {
		e.decisionStats.Keyframes++
	} else {
		e.decisionStats.InterFrames++
	}
	if nTiles <= 1 {
		e.decisionStats.add(e.pc.decisionStats)
		return
	}
	if keyTileZeroUsesPrimary {
		e.decisionStats.add(e.pc.decisionStats)
		for t := 1; t < nTiles; t++ {
			e.decisionStats.add(e.tilePCs[t].decisionStats)
		}
		return
	}
	for t := 0; t < nTiles; t++ {
		e.decisionStats.add(e.tilePCs[t].decisionStats)
	}
}

func (e *VideoEncoder) sequenceHeader(width, height int) SequenceHeader {
	seq := losslessKeyframeSequence(width, height)
	if e != nil {
		seq.Profile = videoColorProfile(e.color)
		seq.ColorConfig = e.color
		if !videoColorSupportsInLoopFilters(e.parserColorConfig()) {
			seq.EnableCDEF = false
		}
	}
	// SVT signals both per frame (sequence_control_set.c: seq_force_*
	// = 2).
	seq.SeqForceScreenContentTools = SequenceSelectScreenContentTools
	seq.SeqForceIntegerMV = SequenceSelectIntegerMV
	return seq
}

func (e *VideoEncoder) parserColorConfig() parser.ColorConfig {
	if e == nil {
		return parserColorConfig(defaultVideoColorConfig())
	}
	return parserColorConfig(e.color)
}

// applyContentHintToFrameHeader is svt_aom_sig_deriv_multi_processes_rtc's
// header policy: intrabc_level 0 and palette_level 3 on I-slices of sc_class1
// pictures only, so allow_screen_content_tools is set on screen keyframes
// and force_integer_mv is 0 (resource_coordination_process.c).
func (e *VideoEncoder) applyContentHintToFrameHeader(prefix *FrameHeaderPrefix) {
	if e == nil || prefix == nil {
		return
	}
	intra := prefix.FrameType.keyOrIntra()
	prefix.AllowScreenContentTools = intra && e.sc.class1
	prefix.ForceIntegerMV = intra
}

// resizeBytes returns b at length n, reusing its storage when it fits.
func resizeBytes(b []byte, n int) []byte {
	if cap(b) >= n {
		return b[:n]
	}
	return make([]byte, n)
}

// copyFrameInto deep-copies src into dst, reusing dst's buffers when they fit.
func copyFrameInto(dst *SourceFrame420, src SourceFrame420) {
	dst.Y = resizeBytes(dst.Y, len(src.Y))
	dst.U = resizeBytes(dst.U, len(src.U))
	dst.V = resizeBytes(dst.V, len(src.V))
	copy(dst.Y, src.Y)
	copy(dst.U, src.U)
	copy(dst.V, src.V)
	dst.YStride, dst.ChromaStride = src.YStride, src.ChromaStride
	dst.Width, dst.Height = src.Width, src.Height
}

// defaultTileColsLog2 picks the inter-frame tile-column split: two columns
// once a frame is wide enough that per-tile entropy coding pays for the
// goroutine handoff, four at HD, and thirty-ish legal columns at full HD.
// The full-HD split intentionally saturates the persistent worker pool; the
// tile syntax clamps the requested power-of-two layout to the legal SB grid.
func defaultTileColsLog2(width int) uint8 {
	switch {
	case width >= 1920:
		return 5
	case width >= 1280:
		return 2
	case width >= 512:
		return 1
	}
	return 0
}

// SetTileColumns overrides the inter-frame tile column count (rounded down
// to a power of two, clamped to the legal range for the frame size at encode
// time). One column disables tile parallelism.
func (e *VideoEncoder) SetTileColumns(cols int) {
	e.tileColsLog2 = tileColumnsLog2(cols)
	e.keyframeTileColsLog2 = e.tileColsLog2
}

func (e *VideoEncoder) SetMaxThreads(n int) {
	if e == nil {
		return
	}
	e.singleThread = n == 1
	if n > 1 {
		// Inter frames prefer the intra-tile SB-row wavefront (single tile
		// column, n decision lanes) over tile-column parallelism: it keeps one
		// entropy context per frame, avoiding the ~0.5 dB the per-tile CDF
		// split costs on complex motion, while still saturating the cores.
		// Keyframes have no decision pass to wavefront, so they keep n tile
		// columns for parallelism. Callers wanting tiled inter frames call
		// SetTileColumns.
		e.tileColsLog2 = 0
		e.keyframeTileColsLog2 = tileColumnsLog2(n)
		e.wavefrontLanes = n
		return
	}
	e.wavefrontLanes = 0
	if n == 1 {
		e.SetTileColumns(1)
		return
	}
	e.setDefaultTileColumns()
}

// configurePCWavefront wires the single-tile coder to the encoder-owned
// SB-row wavefront. Single-threaded encoders keep the serial decision walk.
func (e *VideoEncoder) configurePCWavefront() {
	lanes := e.wavefrontLanes
	if e.singleThread {
		lanes = 0
	}
	e.pc.wavefront = &e.wavefront
	e.pc.wavefrontLanes = lanes
}

func (e *VideoEncoder) setDefaultTileColumns() {
	if e != nil {
		e.tileColsLog2 = defaultTileColsLog2(e.width)
		e.keyframeTileColsLog2 = e.tileColsLog2
	}
}

// runHME runs the frame's open-loop ME (hme.go). temporalID decides
// me_ctx->is_ref: base pictures and the L1T3 middle layer are referenced.
func (e *VideoEncoder) runHME(src SourceFrame420, temporalID uint8) {
	isRef := temporalID == 0 || (e.temporalLayers == 3 && temporalID == 1)
	// pa_me searches the frame's own reference: LAST (slot 0) for the base
	// layer, the L1T2 leaf and the L1T3 middle layer and leaf after a base,
	// the middle layer (slot 1) for the L1T3 leaf that follows it.
	ref, store := 0, -1
	if e.temporalLayers == 3 && temporalID == 2 && e.lastTemporalID == 1 {
		ref = 1
	}
	switch {
	case temporalID == 0:
		store = 0
	case e.temporalLayers == 3 && temporalID == 1:
		store = 1
	}
	// get_me_reference's picture distance in SVT's low-delay layering:
	// a base picture references the previous base (2 or 4 pictures back with
	// two or three layers), the L1T3 middle layer the base 2 back, a leaf
	// the picture before it.
	e.hme.dist = 1
	switch {
	case e.temporalLayers == 3 && temporalID == 0:
		e.hme.dist = 4
	case e.temporalLayers == 3 && temporalID == 1, e.temporalLayers == 2 && temporalID == 0:
		e.hme.dist = 2
	}
	e.hme.ctrl = e.meControls(src.Width, src.Height)
	if e.singleThread {
		e.hme.runSerial(src, isRef, ref, store)
		return
	}
	e.hme.run(src, isRef, ref, store)
}

func tileColumnsLog2(cols int) uint8 {
	log2 := uint8(0)
	for (2 << log2) <= cols {
		log2++
	}
	return log2
}

// NewVideoEncoderCBR creates a streaming encoder under CBR rate control. The
// encoder starts in the middle of the configured qindex range and adjusts the
// working qindex after every frame from the bit-budget buffer.
func NewVideoEncoderCBR(width, height int, rc RateControlConfig) (*VideoEncoder, error) {
	return newVideoEncoderCBRWithColor(width, height, rc, defaultVideoColorConfig())
}

// QIndex reports the working qindex the next frame will use.
func (e *VideoEncoder) QIndex() uint8 {
	return e.qIndex
}

// Close waits for background work to finish and releases persistent workers.
func (e *VideoEncoder) Close() error {
	if e == nil {
		return nil
	}
	err := e.joinFilter()
	tileWork, filterWork, leafWork := e.tileWork, e.filterWork, e.leafWork
	if tileWork != nil {
		e.tileWait.Wait()
		close(tileWork)
	}
	if filterWork != nil {
		close(filterWork)
	}
	if leafWork != nil {
		close(leafWork)
	}
	e.tileWorkerWG.Wait()
	e.filterWorkerWG.Wait()
	e.leafWorkerWG.Wait()
	e.tileWork, e.tileWorkers = nil, 0
	e.filterWork, e.filterDone, e.filterStarted = nil, nil, false
	e.filterPending = false
	e.filterParams.out = nil
	e.leafWork, e.leafDone, e.leafStarted = nil, nil, false
	e.leafJob.src, e.leafJob.refRecon = SourceFrame420{}, SourceFrame420{}
	e.leafJob.prevCtx, e.leafJob.tu = nil, nil
	e.leafJob.cntZeroMV = 0
	e.tileJobParams.src, e.tileJobParams.refRecon = SourceFrame420{}, SourceFrame420{}
	e.tileJobParams.golden, e.tileJobParams.out = nil, nil
	e.tileJobParams.prevCtx, e.tileJobParams.lfMap = nil, nil
	e.tileJobParams.payloads, e.tileJobParams.errs = nil, nil
	e.wavefront.close()
	e.leafWavefront.close()
	e.lf.close()
	e.cdefApp.close()
	e.hme.close()
	e.leafLF.close()
	e.leafCdefApp.close()
	e.leafHME.close()
	return err
}

// Flush waits for background work from the previous Encode call to complete
// without closing the encoder or resetting stream state.
func (e *VideoEncoder) Flush() error {
	if e == nil {
		return nil
	}
	return e.joinFilter()
}

// SetTemporalLayers selects the temporal-layer count: 1 (default) or 2 for
// the WebRTC L1T2 pattern, where odd frames are droppable (they refresh no
// reference slot and always predict from the latest layer-0 frame).
func (e *VideoEncoder) SetTemporalLayers(n int) error {
	if n < 1 || n > 3 {
		return fmt.Errorf("encoder: unsupported temporal layer count %d", n)
	}
	if e.temporalLayers == n {
		return nil
	}
	e.temporalLayers = n
	return nil
}

// TemporalID reports the temporal layer the next frame will be coded in.
func (e *VideoEncoder) TemporalID() uint8 {
	if !e.haveKey {
		return 0
	}
	switch e.temporalLayers {
	case 2:
		if e.frameIndex%2 == 1 {
			return 1
		}
	case 3:
		// The L1T3 group: T0, T2, T1, T2.
		switch e.frameIndex % 4 {
		case 1, 3:
			return 2
		case 2:
			return 1
		}
	}
	return 0
}

// Encode encodes one frame and returns its temporal unit plus whether it was
// coded as a keyframe. The first frame (and any frame with forceKey set) is a
// keyframe; every other frame predicts from the latest layer-0 reconstruction.
// Encode codes one source frame into a temporal unit. The returned slice
// aliases an encoder-owned buffer reused by the next call; callers that
// retain it must copy.
func (e *VideoEncoder) Encode(src SourceFrame420, forceKey bool) ([]byte, bool, error) {
	return e.EncodeWithTemporalID(src, forceKey, e.TemporalID())
}

// EncodeWithTemporalID encodes one frame while forcing the temporal layer used
// for an inter frame. It is used by WebRTC SVC controllers whose per-spatial
// temporal schedules differ from the encoder's default low-delay cadence.
func (e *VideoEncoder) EncodeWithTemporalID(src SourceFrame420, forceKey bool, temporalID uint8) ([]byte, bool, error) {
	if src.Width != e.renderWidth || src.Height != e.renderHeight {
		return nil, false, fmt.Errorf("encoder: frame %dx%d does not match stream %dx%d", src.Width, src.Height, e.renderWidth, e.renderHeight)
	}
	if e.renderWidth != e.width || e.renderHeight != e.height {
		src = e.padSource(src)
	}
	if !e.haveKey || forceKey {
		tu, err := e.encodeKeyWithSequenceMax(src, e.width, e.height)
		if err != nil {
			return nil, false, err
		}
		return tu, true, nil
	}
	if temporalID >= uint8(max(e.temporalLayers, 1)) {
		return nil, false, ErrInvalidFrame
	}
	e.runHME(src, temporalID)
	tu, err := e.encodePReusing(src, temporalID)
	if err != nil {
		return nil, false, err
	}
	e.frameIndex++
	e.lastTemporalID = temporalID
	e.rcUpdate(len(tu)*8, temporalID)
	return tu, false, nil
}

func (e *VideoEncoder) encodeKeyWithSequenceMax(src SourceFrame420, maxWidth, maxHeight int) ([]byte, error) {
	e.dropRefPads()
	// Periodic keyframes reuse the stream's recon buffer and tile-coder pool,
	// so a scene cut allocates only its temporal unit.
	nTiles := 1
	if e.keyframeTileColsLog2 > 0 {
		nTiles = 1 << e.keyframeTileColsLog2
	}
	if len(e.tilePCs) < nTiles {
		e.tilePCs = make([]pframeCoder, nTiles)
	}
	if cap(e.payloads) < nTiles {
		e.payloads = make([]TilePayload, nTiles)
	}
	if cap(e.tileErrs) < nTiles {
		e.tileErrs = make([]error, nTiles)
	}
	e.configureDecisionStats(nTiles)
	// The keyframe path reuses the filter appliers a backgrounded pass may
	// still hold.
	if err := e.joinFilter(); err != nil {
		return nil, err
	}
	e.lazyCDEF.pending = false
	// perform_sc_detection: the I-slice derives the picture's sc_class and
	// every following inter frame inherits it.
	e.sc = e.scDet.classes(e.content, src.Y, src.YStride, src.Width, src.Height)
	// Keyframe quality boost: every later frame predicts (directly or
	// transitively) from the key, so it earns a finer quantizer than the
	// working point; rate control pays the debt back over the next frames
	// through the buffer.
	keyQ := e.rcKeyQ(src.Width, src.Height, e.svtScreen())
	tu, recon, err := encodeKeyframeFilteredTiles(src, keyQ, &e.lf, e.renderWidth, e.renderHeight, &e.keyRecon, nil, &e.cdefApp, e.keyframeTileColsLog2, keyframeTileOptions{
		payloads:          e.payloads[:nTiles],
		errs:              e.tileErrs[:nTiles],
		groupScratch:      &e.tuGroup,
		outScratch:        &e.tuScratch,
		stream:            e,
		sequenceMaxWidth:  maxWidth,
		sequenceMaxHeight: maxHeight,
	})
	if err != nil {
		return nil, err
	}
	e.collectDecisionStats(true, nTiles, true)
	e.hme.prime(src)
	e.recon = recon
	e.lastRecon = recon
	e.lpd1KeyRefs(src.Width, src.Height)
	e.haveKey = true
	e.frameIndex = 1
	// A shown keyframe refreshes every slot with its context-update tile's
	// frame-end CDFs (refresh_frame_context; tile zero runs on e.pc), so
	// the following inter frames chain from them through primary_ref_frame
	// LAST, as SVT-AV1's do, instead of restarting from defaults.
	e.haveCtx, e.haveCtxT1 = false, false
	if videoColorIs420(e.parserColorConfig()) {
		if err := e.saveKeyframeCDFs(keyQ); err != nil {
			return nil, err
		}
		e.haveCtx, e.haveCtxT1 = true, true
	}
	e.lastTemporalID = 0
	// A shown keyframe refreshes every reference slot, so slot 1 (the GOLDEN
	// name) now holds the keyframe; seed the search copy.
	if e.goldenEvery > 0 {
		e.refPad.invalidate(&e.golden)
		copyFrameInto(&e.golden, recon)
		e.sinceGoldenFresh = 0
	}
	e.rcUpdate(len(tu)*8, 0)
	return tu, nil
}

func (e *VideoEncoder) encodeReferencePFrameWithSequenceMax(src SourceFrame420, ref SourceFrame420, codedRefBuffer uint8, settings FrameEncodeSettings, maxWidth, maxHeight int) ([]byte, error) {
	e.dropRefPads()
	if src.Width != e.renderWidth || src.Height != e.renderHeight {
		return nil, fmt.Errorf("encoder: frame %dx%d does not match stream %dx%d", src.Width, src.Height, e.renderWidth, e.renderHeight)
	}
	if settings.ReferenceCount == 0 || codedRefBuffer >= encoderRefFrames ||
		(settings.UpdateBufferSet && settings.UpdateBuffer >= encoderRefFrames) {
		return nil, ErrInvalidFrame
	}
	if settings.TemporalID >= uint8(max(e.temporalLayers, 1)) {
		return nil, ErrInvalidFrame
	}
	if e.renderWidth != e.width || e.renderHeight != e.height {
		src = e.padSource(src)
	}
	if err := e.joinFilter(); err != nil {
		return nil, err
	}
	e.lazyCDEF.pending = false

	hmeArmed := e.hme.havePrev
	e.runHME(src, settings.TemporalID)
	e.pc.st.hme = &e.hme
	for t := range e.tilePCs {
		e.tilePCs[t].st.hme = &e.hme
	}

	out := &e.t2Recon
	if settings.UpdateBufferSet {
		out = &e.reconBufs[e.reconIdx]
	}
	if out.Y == nil || len(out.Y) != len(src.Y) || len(out.U) != len(src.U) || len(out.V) != len(src.V) {
		*out = SourceFrame420{
			Y:            make([]byte, len(src.Y)),
			U:            make([]byte, len(src.U)),
			V:            make([]byte, len(src.V)),
			YStride:      src.YStride,
			ChromaStride: src.ChromaStride,
			Width:        src.Width,
			Height:       src.Height,
		}
	}

	refresh := uint8(0)
	if settings.UpdateBufferSet {
		refresh = 1 << settings.UpdateBuffer
	}
	seq := e.sequenceHeader(maxWidth, maxHeight)
	effQ := e.rcInterQ(settings.TemporalID, svtRefLast, e.frameIndex, src.Width, src.Height, e.svtScreen())
	// This path signals no delta_q; drop any grid a previous frame left.
	e.pc.st.sbQPending = nil
	for t := range e.tilePCs {
		e.tilePCs[t].st.sbQPending = nil
	}
	header, _ := repeatPFrameHeader(src.Width, src.Height, effQ, refresh)
	if interTransformMode(e.parserColorConfig(), effQ) == TransformModeLargest {
		header.TransformRef.TransformMode = TransformModeLargest
	}
	e.applyContentHintToFrameHeader(&header.Prefix)
	header.Prefix.FrameSizeOverride = src.Width != maxWidth || src.Height != maxHeight
	header.Size.RefFrameIdx = [7]uint8{}
	for i := range header.Size.RefFrameIdx {
		header.Size.RefFrameIdx[i] = codedRefBuffer
	}
	e.refState = referenceStateForFrame(ref.Width, ref.Height)
	header.References = &e.refState
	if e.renderWidth != e.width || e.renderHeight != e.height {
		header.Size.RenderWidth = uint32(e.renderWidth)
		header.Size.RenderHeight = uint32(e.renderHeight)
		header.Size.HaveRenderSize = true
	}
	if e.tileColsLog2 > 0 || header.Prefix.FrameSizeOverride {
		tiles, err := interTileInfoForSequence(seq, src.Width, src.Height, e.tileColsLog2)
		if err != nil {
			return nil, fmt.Errorf("tile info: %w", err)
		}
		header.Tile = tiles
	}

	nTiles := int(header.Tile.Cols)
	if len(e.tilePCs) < nTiles {
		e.tilePCs = make([]pframeCoder, nTiles)
	}
	e.configureDecisionStats(nTiles)
	if cap(e.payloads) < nTiles {
		e.payloads = make([]TilePayload, nTiles)
	}
	if cap(e.tileErrs) < nTiles {
		e.tileErrs = make([]error, nTiles)
	}
	payloads := e.payloads[:nTiles]
	for i := range payloads {
		payloads[i] = TilePayload{}
	}
	color := e.parserColorConfig()
	if !videoColorSupportsInLoopFilters(color) {
		header.CDEF = CDEFParams{}
	}
	// Depth removal stays off here (Phase 1 arms only the standard
	// temporal-layer path in encodePReusing) so no stale level carries over.
	e.pc.st.depthRemovalLevel = 0
	e.pc.forceSplit = e.forceSplit
	e.configurePCWavefront()
	for t := range e.tilePCs {
		e.tilePCs[t].st.depthRemovalLevel = 0
		e.tilePCs[t].forceSplit = e.forceSplit
	}
	if nTiles == 1 {
		data, err := e.pc.encodeTileWithOptionsColor(src, ref, nil, out, effQ, nil, parser.ReferenceModeSingle, header.Prefix.ForceIntegerMV, header.Prefix.AllowScreenContentTools, color, 0, uint16(src.Width/4))
		if err != nil {
			return nil, fmt.Errorf("encode tile: %w", err)
		}
		payloads[0].Data = data
	} else {
		miCols := uint16(src.Width / 4)
		errs := e.tileErrs[:nTiles]
		for i := range errs {
			errs[i] = nil
		}
		tj := &e.tileJobParams
		tj.src, tj.refRecon, tj.golden, tj.out = src, ref, nil, out
		tj.effQ, tj.prevCtx = effQ, nil
		tj.referenceMode = parser.ReferenceModeSingle
		tj.forceIntegerMV = header.Prefix.ForceIntegerMV
		tj.allowScreen = header.Prefix.AllowScreenContentTools
		tj.color = color
		tj.tile, tj.miCols = header.Tile, miCols
		tj.lfMap = nil
		tj.payloads, tj.errs = payloads, errs
		jobs := nTiles - 1
		if jobs > 15 {
			jobs = 15
		}
		e.startTileWorkers(jobs)
		e.tileWait.Add(jobs)
		for j := 0; j < jobs; j++ {
			e.tileWork <- tileWorkRange{first: j + 1, stride: jobs, limit: nTiles}
		}
		c0, c1 := tileColBounds(header.Tile, 0, miCols)
		data, err := e.tilePCs[0].encodeTileWithOptionsColor(src, ref, nil, out, effQ, nil, parser.ReferenceModeSingle, header.Prefix.ForceIntegerMV, header.Prefix.AllowScreenContentTools, color, c0, c1)
		e.tileWait.Wait()
		if err != nil {
			return nil, fmt.Errorf("encode tile 0: %w", err)
		}
		payloads[0].Data = data
		for t := 1; t < nTiles; t++ {
			if errs[t] != nil {
				return nil, fmt.Errorf("encode tile %d: %w", t, errs[t])
			}
		}
	}
	tu, err := assembleInterTU(seq, header, payloads, settings.TemporalID, &e.tuGroup, &e.tuScratch)
	if err != nil {
		return nil, err
	}
	e.collectDecisionStats(false, nTiles, false)
	e.lastRecon = *out
	if settings.UpdateBufferSet {
		e.recon = *out
		e.reconIdx ^= 1
		e.noteRCRefresh(svtRefLast, effQ, settings.TemporalID)
	}
	e.lpd1InvalidateRefs()
	e.haveKey = true
	e.frameIndex++
	e.haveCtx = false
	e.haveCtxT1 = false
	e.lastTemporalID = settings.TemporalID
	if !hmeArmed {
		e.hme.prime(src)
	}
	if e.goldenEvery > 0 {
		e.refPad.invalidate(&e.golden)
		copyFrameInto(&e.golden, *out)
		e.sinceGoldenFresh = 0
	}
	e.rcUpdate(len(tu)*8, settings.TemporalID)
	return tu, nil
}

// encodePReusing is the steady-state P-frame path: it reuses the encoder-owned
// coder state and double-buffered reconstruction planes so per-frame work
// allocates only the emitted temporal unit.
// Prewarm runs the whole encode machinery once on a throwaway frame and
// resets the stream state, so every lazily sized buffer, worker pool and
// per-coder scratch exists before the first real frame: steady-state encoding
// allocates nothing and the first frame pays no initialization latency.
func (e *VideoEncoder) Prewarm() error {
	// A fresh single-layer encoder has no learned controller/stream state to
	// preserve. The keyframe below already allocates the golden snapshot;
	// refreshing it later only copies into that same storage. Temporal-layer
	// and reused encoders keep the legacy walk because their retained rate
	// controller and motion history can depend on the number of warmup frames.
	freshSingleLayer := !e.haveKey && e.frameIndex == 0 && e.avgFrameLowMotion == 0 &&
		e.temporalLayers <= 1 && !e.decisionStatsEnabled
	color := e.parserColorConfig()
	cw := chromaWidthForColor(e.width, color)
	ch := chromaHeightForColor(e.height, color)
	src := SourceFrame420{
		Y:            make([]byte, e.width*e.height),
		U:            make([]byte, cw*ch),
		V:            make([]byte, cw*ch),
		YStride:      e.width,
		ChromaStride: cw,
		Width:        e.renderWidth,
		Height:       e.renderHeight,
	}
	savedQ := e.qIndex
	savedCREnd := e.cr.sbEnd
	savedRC, savedRef := e.svtRC, e.svtRef
	// Size the assembly buffers for real content up front; the throwaway
	// frame below is black and would leave them near-empty.
	if bound := e.width * e.height; cap(e.tuScratch) < bound {
		e.tuScratch = make([]byte, 0, bound)
		e.tuGroup = make([]byte, 0, bound)
	}
	if _, _, err := e.Encode(src, true); err != nil {
		return err
	}
	// Exercise every reconstruction buffer and the golden-refresh cadence so
	// delayed scratch growth happens before the first externally visible frame.
	// Two full temporal-layer periods: every layer, and both base-layer
	// reconstruction buffers, code at least once.
	frames := 2
	if e.temporalLayers > 1 {
		frames = 2 << (e.temporalLayers - 1)
	}
	if e.goldenEvery > 0 && frames < e.goldenEvery+1 {
		frames = e.goldenEvery + 1
	}
	if freshSingleLayer && frames > 3 {
		// Two P frames visit both reconstruction buffers. Keep the legacy
		// parity so the returned encoder retains the same buffer ownership.
		frames = 2 + frames&1
	}
	for i := 0; i < frames; i++ {
		if _, _, err := e.Encode(src, false); err != nil {
			return err
		}
	}
	if err := e.joinFilter(); err != nil {
		return err
	}
	e.prewarmPipeline(src)
	// The CDEF search frames record their tile symbol streams; size the
	// logs for real content (the black frames above barely use them).
	sbCols := (e.width + 63) / 64
	logTiles := min(max(1<<e.keyframeTileColsLog2, 1<<e.tileColsLog2), sbCols)
	e.cdefLogs.reserve(e.width, e.height, logTiles)
	// The black frames search to zero strengths, so the CDEF band workers
	// would otherwise first start on a real frame.
	if e.cdefApp.bound && !e.singleThread {
		e.cdefApp.startWorkers()
	}
	e.haveKey = false
	e.haveCtx = false
	e.haveCtxT1 = false
	e.frameIndex = 0
	e.lastTemporalID = 0
	e.qIndex = savedQ
	e.cr.sbEnd = savedCREnd
	e.svtRC, e.svtRef = savedRC, savedRef
	e.svtCurOn = false
	e.sinceGoldenFresh = 0
	e.hme.havePrev = false
	e.hme.fme.valid = false
	e.hme.svt.haveCur, e.hme.svt.havePrev = false, false
	for s := range e.hme.refY {
		e.hme.refY[s].valid = false
	}
	e.hme.svt.slotOK = [hmeRefSlots]bool{}
	e.lastRecon = SourceFrame420{}
	return nil
}

// tileColBounds is the MI column range of one tile column.
func tileColBounds(tile TileInfo, t int, miCols uint16) (uint16, uint16) {
	c0 := tile.ColStartSB[t] * 16
	c1 := tile.ColStartSB[t+1] * 16
	if c1 > miCols {
		c1 = miCols
	}
	return c0, c1
}

func tilePayloadColBounds(tile TileInfo, t int, miCols uint16) (uint16, uint16) {
	if tile.Cols <= 1 {
		return 0, miCols
	}
	return tileColBounds(tile, t, miCols)
}

// startTileWorkers grows the persistent tile-column worker pool to the number
// needed by the current tile layout. Workers park on the job channel between
// frames, and narrow streams avoid spawning idle goroutines they never use.
func (e *VideoEncoder) startTileWorkers(workers int) {
	if workers <= e.tileWorkers {
		return
	}
	if e.tileWork == nil {
		e.tileWork = make(chan tileWorkRange, 16)
	}
	workCh := e.tileWork
	for ; e.tileWorkers < workers; e.tileWorkers++ {
		e.tileWorkerWG.Add(1)
		go func() {
			defer e.tileWorkerWG.Done()
			for work := range workCh {
				tj := &e.tileJobParams
				for t := work.first; t < work.limit; t += work.stride {
					c0, c1 := tileColBounds(tj.tile, t, tj.miCols)
					var data []byte
					var err error
					if work.key {
						data, err = e.tilePCs[t].encodeKeyframeTileWithColorOptions(tj.src, tj.out, tj.effQ, c0, c1, tj.lfMap, tj.allowScreen, tj.color)
					} else {
						data, err = e.tilePCs[t].encodeTileWithOptionsColor(tj.src, tj.refRecon, tj.golden, tj.out, tj.effQ, tj.prevCtx, tj.referenceMode, tj.forceIntegerMV, tj.allowScreen, tj.color, c0, c1)
					}
					if err != nil {
						tj.errs[t] = err
					} else {
						tj.payloads[t].Data = data
					}
				}
				e.tileWait.Done()
			}
		}()
	}
}

func (e *VideoEncoder) runKeyframeTileWorkers(req keyframeTileRun) error {
	nTiles := len(req.payloads)
	if nTiles == 1 {
		c0, c1 := tileColBounds(req.tile, 0, req.miCols)
		data, err := e.pc.encodeKeyframeTileWithColorOptions(req.src, req.recon, req.qIndex, c0, c1, req.lfMap, req.allowScreenContentTools, req.color)
		if err != nil {
			return fmt.Errorf("encode tile 0: %w", err)
		}
		req.payloads[0].Data = data
		return nil
	}
	tj := &e.tileJobParams
	tj.src, tj.out = req.src, req.recon
	tj.effQ = req.qIndex
	tj.allowScreen = req.allowScreenContentTools
	tj.color = req.color
	tj.tile, tj.miCols = req.tile, req.miCols
	tj.lfMap = req.lfMap
	tj.payloads, tj.errs = req.payloads, req.errs
	jobs := nTiles - 1
	if jobs > 15 {
		jobs = 15
	}
	e.startTileWorkers(jobs)
	e.tileWait.Add(jobs)
	for j := 0; j < jobs; j++ {
		e.tileWork <- tileWorkRange{first: j + 1, stride: jobs, limit: nTiles, key: true}
	}
	c0, c1 := tileColBounds(req.tile, 0, req.miCols)
	data, err := e.pc.encodeKeyframeTileWithColorOptions(req.src, req.recon, req.qIndex, c0, c1, req.lfMap, req.allowScreenContentTools, req.color)
	e.tileWait.Wait()
	if err != nil {
		return fmt.Errorf("encode tile 0: %w", err)
	}
	req.payloads[0].Data = data
	for t := 1; t < nTiles; t++ {
		if req.errs[t] != nil {
			return fmt.Errorf("encode tile %d: %w", t, req.errs[t])
		}
	}
	return nil
}

// startFilterWorker launches the persistent in-loop filter worker once; it
// parks between frames and reads its inputs from filterParams.
func (e *VideoEncoder) startFilterWorker() {
	if e.filterStarted {
		return
	}
	work := make(chan struct{}, 1)
	done := make(chan error, 1)
	e.filterDone, e.filterWork = done, work
	e.filterWorkerWG.Add(1)
	go func() {
		defer e.filterWorkerWG.Done()
		for range work {
			p := &e.filterParams
			// The background worker only runs for the multithread branch
			// (dispatched under !e.singleThread), so it always uses a parallel
			// apply — never re-read e.singleThread or e.tileColsLog2 here, which
			// SetConfig may be mutating concurrently; the single-tile-vs-multi-tile
			// choice is snapshotted into filterParams on the dispatch goroutine.
			done <- e.applyInLoopFilters(p.out, p.lfLevel, p.cdef, p.cdefIdx, false, p.parallelMasks)
		}
	}()
	e.filterStarted = true
}

func (e *VideoEncoder) applyInLoopFilters(out *SourceFrame420, lfLevel uint8, cdef parser.CDEFParams, cdefIdx []uint8, singleThread, parallelMasks bool) error {
	lf := parser.LoopFilterParams{
		LevelY: [2]uint8{lfLevel, lfLevel},
		LevelU: lfLevel,
		LevelV: lfLevel,
	}
	applyLF := e.lf.apply
	applyCDEF := e.cdefApp.apply
	switch {
	case singleThread:
		// Single-threaded inter frames code a single tile (SetMaxThreads(1)
		// forces SetTileColumns(1)), so the mask build carries context across
		// the whole frame width and is byte-identical to the edge-list pass.
		applyLF = e.lf.applySerialMasks
		applyCDEF = e.cdefApp.applySerial
	case parallelMasks:
		// All multithread inter frames route here, INCLUDING the multi-tile
		// default (defaultTileColsLog2(1920)=5, ~30 tile columns). The dav1d
		// bitmask apply is byte-identical to the edge-list sweep across tile
		// columns because BuildFromMap derives each block's left/above tx-context
		// directly from the shared record map -- the same neighbour the sweep
		// reads -- instead of carrying it forward, so it reproduces the sweep's
		// per-edge width at superblock AND tile-column boundaries (gated by
		// TestRealCMaskDefaultMultiTile: encode realC, decode, assert recon ==
		// decode byte-for-byte at 30 tile columns). This is the same apply as the
		// single-thread path, fanned across the applier's band workers, and is
		// much faster than the parallel edge-list planner+apply on edge-dense
		// frames. The decision is snapshotted by the caller (dispatch goroutine)
		// so this can run on the detached filter worker without racing SetConfig.
		applyLF = e.lf.applyParallelMasks
	}
	// lfLevel 0 (SVT-AV1 preset 12 rtc: dlf_level 0) leaves the deblocking
	// pass out; the loop-filter records still feed CDEF.
	if lfLevel > 0 {
		if err := applyLF(out, lf); err != nil {
			return fmt.Errorf("loop filter apply: %w", err)
		}
	}
	if err := applyCDEF(out, cdef, &e.lf.filtMap, cdefIdx); err != nil {
		return fmt.Errorf("cdef apply: %w", err)
	}
	return nil
}

// joinFilter blocks until a backgrounded in-loop filter pass finishes. It
// must precede any read of the filtered reconstruction and any reuse of the
// loop-filter or CDEF applier state.
func (e *VideoEncoder) joinFilter() error {
	if e.filterPending {
		e.filterPending = false
		if err := <-e.filterDone; err != nil {
			e.filterErr = err
		}
	}
	err := e.filterErr
	e.filterErr = nil
	return err
}

// Adaptive golden refresh, ported from libaom's one-pass realtime golden
// update (av1/encoder/ratectrl.c): set_golden_update drops the interval to 16
// display frames when avg_frame_low_motion < 40, and
// av1_adjust_gf_refresh_qp_one_pass_rt forces a refresh on sustained high
// motion (avg_frame_low_motion < 20) once 10 display frames have passed.
// (The upstream function's companion QP rules — force-refresh when qindex
// runs below 87% of the average, defer when above it — measured red on the
// complex-motion corpus clip and are deliberately not ported.) Frame-count
// constants are libaom's divided by two because the base layer, whose frames
// sinceGoldenFresh counts, carries every other display frame at L1T2.
const (
	goldenIntervalLowMotion      = 8
	goldenForceRefreshHighMotion = 5
)

func (e *VideoEncoder) encodePReusing(src SourceFrame420, temporalID uint8) ([]byte, error) {
	// The previous frame's filters may still be running; they own the
	// applier state and the reference planes the tiles are about to read.
	if err := e.joinFilter(); err != nil {
		return nil, err
	}
	e.lazyCDEF.pending = false
	droppable := temporalID > 0
	// L1T3: the middle layer (T1) is referenced by the following T2, so it
	// saves its reconstruction to slot 2 and exports a context for that T2
	// to chain from; T2 leaves update nothing. L1T2 keeps its single
	// droppable layer in t1Recon.
	isT1 := e.temporalLayers == 3 && temporalID == 1
	afterT1 := e.temporalLayers == 3 && temporalID == 2 && e.lastTemporalID == 1
	var out *SourceFrame420
	switch {
	case !droppable:
		out = &e.reconBufs[e.reconIdx]
	case e.temporalLayers == 3 && temporalID == 2:
		out = &e.t2Recon
	default:
		out = &e.t1Recon
	}
	if out.Y == nil {
		*out = SourceFrame420{
			Y:            make([]byte, len(src.Y)),
			U:            make([]byte, len(src.U)),
			V:            make([]byte, len(src.V)),
			YStride:      src.YStride,
			ChromaStride: src.ChromaStride,
			Width:        src.Width,
			Height:       src.Height,
		}
	}
	refresh := uint8(0x01)
	refreshGolden := false
	// SVT-AV1 preset 12 rtc low delay with three layers and one reference
	// (mrp_level 0, ld_reduce_ref_buffs 2; pd_process.c LOW_DELAY CBR
	// branch): TL0 refreshes slot 0 and clears 2..7 (1 << lay0_toggle |
	// 0xfc), TL1 refreshes slot 1, TL2 refreshes nothing; every reference
	// name points at the one picture the frame predicts from.
	svtLD := e.temporalLayers == 3 && e.goldenEvery == 0
	if svtLD && !droppable {
		refresh = 0x01 | 0xfc
	}
	if isT1 {
		refresh = 0x04 // slot 2 carries the middle layer for its T2 leaves
		if svtLD {
			refresh = 0x02
		}
	} else if droppable {
		refresh = 0 // leaf-layer frames are never referenced
	} else if e.goldenEvery > 0 {
		e.sinceGoldenFresh++
		// set_golden_update (libaom av1/encoder/ratectrl.c): content whose
		// avg_frame_low_motion runs under 40 shortens the golden interval to
		// 16 frames (8 base-layer frames).
		interval := e.goldenEvery
		if e.avgFrameLowMotion > 0 && e.avgFrameLowMotion < 40 && interval > goldenIntervalLowMotion {
			interval = goldenIntervalLowMotion
		}
		if e.sinceGoldenFresh >= interval ||
			(e.avgFrameLowMotion > 0 && e.avgFrameLowMotion < 20 && e.sinceGoldenFresh >= goldenForceRefreshHighMotion) {
			refresh |= 0x02 // this frame becomes the new golden anchor
			refreshGolden = true
		}
	}
	seq := e.sequenceHeader(src.Width, src.Height)
	rcRef := svtRefLast
	if afterT1 {
		rcRef = svtRefT1
	}
	effQ := e.rcInterQ(temporalID, rcRef, e.frameIndex+e.lpd1FrameBias, src.Width, src.Height, e.svtScreen())
	header, refState := repeatPFrameHeader(src.Width, src.Height, effQ, refresh)
	e.refState = refState
	if interTransformMode(e.parserColorConfig(), effQ) == TransformModeLargest {
		header.TransformRef.TransformMode = TransformModeLargest
	}
	e.applyContentHintToFrameHeader(&header.Prefix)
	if e.renderWidth != e.width || e.renderHeight != e.height {
		header.Size.RenderWidth = uint32(e.renderWidth)
		header.Size.RenderHeight = uint32(e.renderHeight)
		header.Size.HaveRenderSize = true
	}
	header.References = &e.refState
	// In-loop deblocking: signal the q-derived filter levels and collect the
	// per-MI records the frame-level pass needs; after the tiles finish the
	// encoder runs the decoder's own loop filter over the reconstruction so
	// recon stays bit-exact with decoder output.
	lfLevel := uint8(0)
	color := e.parserColorConfig()
	filterSupported := videoColorSupportsInLoopFilters(color)
	if filterSupported && !e.fastestRealtimeEffort() && src.Width*src.Height <= loopFilterMaxArea {
		lfLevel = filterLevelFromQIndex(effQ, false)
	}
	if !filterSupported {
		header.CDEF = CDEFParams{}
	}
	// SVT-AV1 rtc CDEF (cdef_frame.go): the picture's cdef_search_level and
	// the reference-derived filters, decided before the tiles so a
	// searching picture records its tile symbol streams.
	cdefRef, cdefRefStats := e.cdefRefLast, e.lpd1Last
	if afterT1 {
		cdefRef, cdefRefStats = e.cdefRefT1, e.lpd1T1
	}
	cdefPlan := planSVTCDEF(filterSupported && seq.EnableCDEF && !e.fastestRealtimeEffort() && src.Width*src.Height <= loopFilterMaxArea,
		false, temporalID == 0, !droppable || isT1, e.svtScreen(), cdefRef, lpd1RefSkipPerc(cdefRefStats))
	if lfLevel > 0 || cdefPlan.on {
		if !e.lf.bound {
			if err := e.lf.init(src.Width, src.Height); err != nil {
				return nil, fmt.Errorf("loop filter init: %w", err)
			}
		}
		if err := e.lf.reset(); err != nil {
			return nil, fmt.Errorf("loop filter reset: %w", err)
		}
		dlf := svtRTCDeblockLevel(lfLevel)
		header.LoopFilter.LevelY = [2]uint8{dlf, dlf}
		header.LoopFilter.LevelU = dlf
		header.LoopFilter.LevelV = dlf
		// The loop-filter records double as the CDEF index map and per-block
		// skip source; the strengths are decided after the tiles.
		if !e.cdefApp.bound {
			if err := e.cdefApp.init(src.Width, src.Height, cdefParserParams(header.CDEF)); err != nil {
				return nil, fmt.Errorf("cdef init: %w", err)
			}
		}
	}
	if afterT1 {
		// The trailing T2 predicts from the middle layer: LAST names slot 2.
		header.Size.RefFrameIdx[0] = 2
	}
	if svtLD {
		slot := uint8(0)
		if afterT1 {
			slot = 1
		}
		for i := range header.Size.RefFrameIdx {
			header.Size.RefFrameIdx[i] = slot
		}
	}
	var prevCtx *frameCDFs
	contextChainSupported := videoColorIs420(color)
	if contextChainSupported && afterT1 && e.haveCtxT1 {
		// Chain from the middle layer's saved state (slot 2 via LAST).
		header.Prefix.ErrorResilientMode = false
		header.Prefix.PrimaryRefFrame = 0
		e.prevLFDeltas = defaultLoopFilterDeltas()
		header.PreviousLFDeltas = &e.prevLFDeltas
		prevCtx = &e.frameCtxT1
	} else if contextChainSupported && !afterT1 && e.haveCtx {
		// Chain the symbol contexts from slot 0's saved frame state. The
		// header then codes loop-filter deltas relative to that frame's,
		// which this encoder keeps at the defaults.
		header.Prefix.ErrorResilientMode = false
		header.Prefix.PrimaryRefFrame = 0
		e.prevLFDeltas = defaultLoopFilterDeltas()
		header.PreviousLFDeltas = &e.prevLFDeltas
		prevCtx = &e.frameCtx
	}
	if e.tileColsLog2 > 0 {
		tiles, err := interTileInfo(src.Width, src.Height, e.tileColsLog2)
		if err != nil {
			return nil, fmt.Errorf("tile info: %w", err)
		}
		header.Tile = tiles
	}
	var golden *SourceFrame420
	if e.goldenEvery > 0 && e.golden.Y != nil {
		golden = &e.golden
	}
	// Hierarchical coarse-search seeds (computed in Encode) recenter every
	// tile's full-pel refinement windows (read-only during tiles).
	e.pc.st.hme = &e.hme
	for t := range e.tilePCs {
		e.tilePCs[t].st.hme = &e.hme
	}
	nTiles := int(header.Tile.Cols)
	if len(e.tilePCs) < nTiles {
		e.tilePCs = make([]pframeCoder, nTiles)
	}
	e.configureDecisionStats(nTiles)
	if cap(e.payloads) < nTiles {
		e.payloads = make([]TilePayload, nTiles)
	}
	if cap(e.tileErrs) < nTiles {
		e.tileErrs = make([]error, nTiles)
	}
	payloads := e.payloads[:nTiles]
	for i := range payloads {
		payloads[i] = TilePayload{}
	}
	var cdefCoders []*pframeCoder
	if cdefPlan.search {
		cdefCoders = e.interTileCoders(nTiles)
		e.cdefLogs.arm(cdefCoders)
	}
	if lfLevel > 0 || cdefPlan.on {
		e.pc.st.lfMap = &e.lf.filtMap
		for t := range e.tilePCs {
			e.tilePCs[t].st.lfMap = &e.lf.filtMap
		}
	} else {
		e.pc.st.lfMap = nil
		for t := range e.tilePCs {
			e.tilePCs[t].st.lfMap = nil
		}
	}
	// A serial single-tile block walk has the same ordering as the decoder.
	// Build edge masks here while its above/left transform contexts are live,
	// retaining the record map for CDEF and level resolution.
	e.pc.st.lfMaskBuilder = nil
	if lfLevel > 0 && e.singleThread && nTiles == 1 {
		if err := e.lf.orderedBuilder.Reset(&e.lf.masks, encoderLoopFilterSBLog2); err != nil {
			return nil, fmt.Errorf("loop filter mask reset: %w", err)
		}
		e.lf.orderedMasks = true
		e.pc.st.lfMaskBuilder = &e.lf.orderedBuilder
	}
	// The light-PD1 MDS0 candidate decision runs on every block of every
	// inter frame, as SVT's generate_md_stage_0_cand_light_pd1 does; the
	// temporal-layer modulation comes from the per-SB light-PD1 level
	// (pframe_lpd1_level.go).
	e.pc.st.mds0Level = 1
	// Depth removal (SVT set_depth_removal_level_controls; see
	// pframe_depth_removal.go) follows the same is_base modulation as
	// mds0Level: base frames run level 9, droppable leaves 14 (preset-12
	// 1080p ladder, enc_mode_config.c:9287-9295). is_ref covers the L1T3
	// middle layer too (referenced by its trailing T2).
	e.pc.st.depthRemovalLevel = depthRemovalLevelForFrame(src.Width, src.Height, !droppable)
	e.pc.st.depthRemovalIsRef = !droppable || isT1
	// Light-PD1 TX shortcut detectors (SVT set_lpd1_tx_ctrls level 3, the
	// rtc-forced level; see pframe_lpd1_tx.go). Base pictures keep the
	// chroma detector armed; droppable leaves add the neighbour-gated
	// skip-TX arm, mirroring pic_lpd1_lvl = is_base ? 3 : 7.
	e.pc.st.setLPD1TxCtrls(true)
	e.pc.forceSplit = e.forceSplit
	e.configurePCWavefront()
	for t := range e.tilePCs {
		e.tilePCs[t].st.mds0Level = e.pc.st.mds0Level
		e.tilePCs[t].st.depthRemovalLevel = e.pc.st.depthRemovalLevel
		e.tilePCs[t].st.depthRemovalIsRef = e.pc.st.depthRemovalIsRef
		e.tilePCs[t].st.lpd1Tx = e.pc.st.lpd1Tx
		e.tilePCs[t].forceSplit = e.forceSplit
	}
	refRecon := e.recon
	if afterT1 {
		refRecon = e.t1Recon
	}
	// Light-PD1 level machinery (pframe_lpd1_level.go): pic_lpd1_lvl, the
	// per-SB statistics and the reference whose statistics the detector reads.
	lpd1Ref := e.lpd1Last
	if afterT1 {
		lpd1Ref = e.lpd1T1
	}
	lpd1Setup := lpd1PictureSetup{ref: lpd1Ref, picLvl: lpd1PicLevel(false, !droppable, e.svtScreen()), notLastLayer: !droppable || isT1, armed: true}
	// update_lambda: the highest temporal layer is LF_UPDATE (180), every
	// other layer (flat IPP included) ARF or INTNL_ARF (150); low delay
	// marks every MAX_GF_INTERVAL-th frame GF_UPDATE (set_frame_update_type).
	lpd1Setup.tlFactor = 150
	if e.temporalLayers > 1 && droppable && int(temporalID) == e.temporalLayers-1 {
		lpd1Setup.tlFactor = 180
	}
	lpd1Setup.gfUpdate = (e.frameIndex+e.lpd1FrameBias)%lpd1MaxGFInterval == 0
	// pcs->skip_intra = (is_ref || ref_intra_percentage > 50) ? 0 : 1 for
	// preset 12 rtc inter pictures without a scene transition.
	lpd1Setup.skipIntra = droppable && !isT1 && lpd1RefIntraPerc(lpd1Ref) <= 50
	lpd1Setup.screen = e.svtScreen()
	if videoColorIs420(color) {
		if e.lpd1Cur == nil {
			e.lpd1Cur = new(lpd1Frame)
		}
		e.lpd1Cur.begin(src.Width, src.Height, &e.hme, effQ)
		// The pipelined base runs before frameIndex advances (lpd1FrameBias).
		e.lpd1Cur.poc = e.frameIndex + e.lpd1FrameBias
		lpd1Setup.cur = e.lpd1Cur
	}
	e.pc.st.lpd1Pending = lpd1Setup
	for t := range e.tilePCs {
		e.tilePCs[t].st.lpd1Pending = lpd1Setup
	}
	// Cyclic refresh (cyclic_refresh.go): CBR inter frames of the flat
	// structure or the base layer lower the qindex of a sliding superblock
	// window and signal it with delta_q (svt_aom_cyclic_refresh_init:
	// apply_cyclic_refresh = use_flat_ipp || temporal_layer_index == 0).
	var sbQ []uint8
	if e.rcEnabled && e.svtCurOn && videoColorIs420(color) && effQ > 0 &&
		refRecon.Width == src.Width && refRecon.Height == src.Height {
		sbQ, e.svtCurCR = e.cr.plan(&e.svtRC, svtCRFrame{
			screen:            e.svtScreen(),
			flat:              e.temporalLayers <= 1,
			temporalLayer:     int(temporalID),
			hierLevels:        e.svtRCHierLevels(),
			avgFrameLowMotion: e.avgFrameLowMotion,
		}, effQ, src.Width, src.Height, &e.hme)
	}
	header.Delta = DeltaParams{DeltaQPresent: sbQ != nil}
	sbQCols := (src.Width + 63) / 64
	e.pc.st.sbQPending, e.pc.st.sbQPendingCols = sbQ, sbQCols
	for t := range e.tilePCs {
		e.tilePCs[t].st.sbQPending, e.tilePCs[t].st.sbQPendingCols = sbQ, sbQCols
	}
	referenceMode := parser.ReferenceModeSingle
	if golden != nil && compoundGoldenLikely(&e.pc.st, src, refRecon, golden) {
		header.TransformRef.ReferenceMode = ReferenceModeSelect
		referenceMode = parser.ReferenceModeSelect
	}
	// Padded references (refpad.go, SVT's svt_aom_generate_padding): the
	// frame's output buffer is about to be rewritten, so its cached copy goes.
	e.refPad.invalidate(out)
	var pads *refPads
	if videoColorIs420(color) && refRecon.Width == src.Width && refRecon.Height == src.Height {
		pads = e.refPad.padsFor(refRecon, golden, chromaWidthForColor(src.Width, color), chromaHeightForColor(src.Height, color))
	}
	e.pc.st.refPads = pads
	for t := range e.tilePCs {
		e.tilePCs[t].st.refPads = pads
	}
	if nTiles == 1 {
		data, err := e.pc.encodeTileWithOptionsColor(src, refRecon, golden, out, effQ, prevCtx, referenceMode, header.Prefix.ForceIntegerMV, header.Prefix.AllowScreenContentTools, color, 0, uint16(src.Width/4))
		if err != nil {
			return nil, fmt.Errorf("encode tile: %w", err)
		}
		payloads[0].Data = data
	} else {
		// One persistent worker per tile column beyond the first: tiles
		// share the read-only source and reference planes and write
		// disjoint column ranges of the output reconstruction. Tile zero
		// runs on the calling goroutine (as tilePCs[0], the context-update
		// tile the CDF export reads) so a two-tile frame pays a single
		// handoff; the workers park on a channel between frames so the
		// steady state spawns no goroutines and allocates no closures.
		miCols := uint16(src.Width / 4)
		errs := e.tileErrs[:nTiles]
		for i := range errs {
			errs[i] = nil
		}
		tj := &e.tileJobParams
		tj.src, tj.refRecon, tj.golden, tj.out = src, refRecon, golden, out
		tj.effQ, tj.prevCtx = effQ, prevCtx
		tj.referenceMode = referenceMode
		tj.forceIntegerMV = header.Prefix.ForceIntegerMV
		tj.allowScreen = header.Prefix.AllowScreenContentTools
		tj.color = color
		tj.tile, tj.miCols = header.Tile, miCols
		tj.lfMap = nil
		tj.payloads, tj.errs = payloads, errs
		jobs := nTiles - 1
		if jobs > 15 {
			jobs = 15
		}
		e.startTileWorkers(jobs)
		e.tileWait.Add(jobs)
		for j := 0; j < jobs; j++ {
			e.tileWork <- tileWorkRange{first: j + 1, stride: jobs, limit: nTiles}
		}
		c0, c1 := tileColBounds(header.Tile, 0, miCols)
		data, err := e.tilePCs[0].encodeTileWithOptionsColor(src, refRecon, golden, out, effQ, prevCtx, referenceMode, header.Prefix.ForceIntegerMV, header.Prefix.AllowScreenContentTools, color, c0, c1)
		e.tileWait.Wait()
		if err != nil {
			return nil, fmt.Errorf("encode tile 0: %w", err)
		}
		payloads[0].Data = data
		for t := 1; t < nTiles; t++ {
			if errs[t] != nil {
				return nil, fmt.Errorf("encode tile %d: %w", t, errs[t])
			}
		}
	}
	// The padded copies mirror this frame's references only.
	e.pc.st.refPads = nil
	for t := range e.tilePCs {
		e.tilePCs[t].st.refPads = nil
	}
	// update_motion_stat (libaom av1/encoder/encoder.c): the percentage of
	// MI units coded as near-zero-motion LAST blocks, smoothed 3:1 into
	// avg_frame_low_motion (seeded with the first observation).
	if mi := int(src.Width/4) * int(src.Height/4); mi > 0 {
		cnt := e.pc.st.cntZeroMV
		if nTiles > 1 {
			cnt = 0
			for t := 0; t < nTiles; t++ {
				cnt += e.tilePCs[t].st.cntZeroMV
			}
		}
		pct := 100 * cnt / mi
		if e.avgFrameLowMotion == 0 {
			e.avgFrameLowMotion = pct
		} else {
			e.avgFrameLowMotion = (3*e.avgFrameLowMotion + pct) / 4
		}
	}
	var cdefIdx []uint8
	if cdefPlan.on {
		cdefPlan.lambda = svtCDEFFullLambda(effQ, false, lpd1Setup.gfUpdate, lpd1Setup.tlFactor)
		header.CDEF, cdefIdx = cdefPlan.decide(&e.cdefSearch, &src, out, &e.lf.filtMap, effQ)
		if cdefPlan.search {
			err := e.cdefLogs.replay(payloads, cdefCoders, header.Tile, uint16(src.Width/4), src.Height, cdefIdx, header.CDEF.Bits)
			e.cdefLogs.disarm(cdefCoders)
			if err != nil {
				return nil, fmt.Errorf("cdef replay: %w", err)
			}
		}
	}
	if !droppable {
		e.cdefRefLast = svtCDEFRefFromParams(header.CDEF)
	} else if isT1 {
		e.cdefRefT1 = svtCDEFRefFromParams(header.CDEF)
	}
	filterCDEF := cdefParserParams(header.CDEF)
	// SVT applies CDEF to a picture'"'"'s reconstruction only when something
	// reads it (cdef_process.c: ppcs->is_ref || recon_enabled). A
	// non-reference picture (refresh_frame_flags 0) is read only through
	// Recon(), so its filter pass is deferred to that call (lazyCDEF) and
	// dropped if the next frame is encoded first. Deblocking is off here.
	filtersOn := lfLevel > 0 || cdefPlan.on
	if filtersOn && droppable && !isT1 && svtRTCDeblockLevel(lfLevel) == 0 {
		filtersOn = false
		e.lazyCDEF = lazyCDEFPass{out: out, cdef: filterCDEF, idx: cdefIdx, pending: true}
	}
	if filtersOn && !e.singleThread {
		// The filters run behind the temporal-unit assembly and the next
		// frame's source-only stages; everything reading the filtered
		// planes or the applier state joins first.
		e.startFilterWorker()
		e.filterParams.out = out
		e.filterParams.lfLevel = svtRTCDeblockLevel(lfLevel)
		e.filterParams.cdef = filterCDEF
		e.filterParams.cdefIdx = cdefIdx
		// The dav1d-style deblocking bitmask apply is byte-identical to the
		// edge-list sweep for BOTH single- and multi-tile frames: BuildFromMap
		// derives each block's left/above tx-context directly from the shared
		// record map (the same neighbour the sweep reads), so it reproduces the
		// sweep's per-edge width across superblock AND tile-column boundaries.
		// Always route through the parallel mask apply (much faster than the
		// edge-list planner on edge-dense frames).
		e.filterParams.parallelMasks = true
		e.filterPending = true
		e.filterWork <- struct{}{}
	}
	tu, err := assembleInterTU(seq, header, payloads, temporalID, &e.tuGroup, &e.tuScratch)
	if err != nil {
		return nil, err
	}
	if filtersOn && e.singleThread {
		if err := e.applyInLoopFilters(out, svtRTCDeblockLevel(lfLevel), filterCDEF, cdefIdx, true, false); err != nil {
			return nil, err
		}
	}
	e.collectDecisionStats(false, nTiles, false)
	if lpd1StatsEnabled {
		dumpLPD1Stats(uint64(e.frameIndex))
	}
	// Rate-control facts of the refreshed reference slot.
	if !droppable {
		e.noteRCRefresh(svtRefLast, effQ, temporalID)
	} else if isT1 {
		e.noteRCRefresh(svtRefT1, effQ, temporalID)
	}
	// Publish the light-PD1 statistics with the refreshed reference slot
	// (rest_process.c copy_statistics_to_ref_obj_ect).
	if !droppable {
		e.lpd1Last = e.publishLPD1(e.lpd1Last, lpd1Setup.cur != nil)
	} else if isT1 {
		e.lpd1T1 = e.publishLPD1(e.lpd1T1, lpd1Setup.cur != nil)
	}
	e.lastRecon = *out
	if isT1 {
		// The middle layer's frame-end state is what the decoder saves into
		// slot 2; untouched families carry the values T1 itself loaded.
		if prevCtx != nil {
			e.frameCtxT1 = *prevCtx
		} else if !e.haveCtxT1 {
			if err := e.frameCtxT1.InitDefault(effQ); err != nil {
				return nil, err
			}
		}
		exp := &e.pc
		if nTiles > 1 {
			exp = &e.tilePCs[0]
		}
		if err := exp.exportCDFs(&e.frameCtxT1); err != nil {
			return nil, err
		}
		e.haveCtxT1 = true
	}
	if !droppable {
		e.recon = *out
		e.reconIdx ^= 1
		// The decoder saves the context-update tile's frame-end state into
		// every refreshed slot; tile zero is that tile. The first export
		// seeds the families this encoder never codes at the same defaults
		// the decoder initialized for this frame.
		if !e.haveCtx {
			if err := e.frameCtx.InitDefault(e.qIndex); err != nil {
				return nil, err
			}
		}
		exp := &e.pc
		if nTiles > 1 {
			exp = &e.tilePCs[0]
		}
		if err := exp.exportCDFs(&e.frameCtx); err != nil {
			return nil, err
		}
		e.haveCtx = true
	}
	if refreshGolden {
		// The golden snapshot copies the filtered planes.
		if err := e.joinFilter(); err != nil {
			return nil, err
		}
		e.refPad.invalidate(&e.golden)
		copyFrameInto(&e.golden, *out)
		e.sinceGoldenFresh = 0
	}
	return tu, nil
}

// Recon returns the most recent frame's reconstruction (what a conformant
// decoder outputs for it). The returned planes alias encoder-owned buffers
// that are reused two frames later; callers needing longer-lived snapshots
// must copy.
func (e *VideoEncoder) Recon() SourceFrame420 {
	// A backgrounded filter pass may still be writing these planes; a
	// failure here resurfaces on the next encode call.
	if err := e.joinFilter(); err != nil {
		e.filterErr = err
		return SourceFrame420{}
	}
	if err := e.finishLazyCDEF(); err != nil {
		e.filterErr = err
		return SourceFrame420{}
	}
	r := e.lastRecon
	if r.Y == nil {
		r = e.recon
	}
	r.Width, r.Height = e.renderWidth, e.renderHeight
	return r
}

// saveKeyframeCDFs is the context a keyframe leaves in every reference slot:
// the defaults for its base_q_idx (setup_past_independence), with the
// families the keyframe codes taken from tile zero's frame-end state and the
// symbol counters reset.
func (e *VideoEncoder) saveKeyframeCDFs(keyQ uint8) error {
	if err := e.frameCtx.InitDefault(keyQ); err != nil {
		return err
	}
	pc := &e.pc
	e.frameCtx.Partition = pc.partCDFs
	e.frameCtx.Mode = pc.st.modeCDFs
	e.frameCtx.Intra = pc.st.intraCDFs
	e.frameCtx.Coeff = pc.st.coeffCDFs
	e.frameCtx.TransformType = pc.st.txCDFs
	if err := e.frameCtx.ResetCDFCounts(); err != nil {
		return err
	}
	e.frameCtxT1 = e.frameCtx
	return nil
}
