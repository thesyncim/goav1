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
	rcPerFrameBits int
	rcBuffer       int
	// rcRecentBits holds the last two coded sizes for the single-layer
	// controller; temporal SVC keeps the same state per temporal layer below.
	rcRecentBits           [2]int
	rcMinQ, rcMaxQ         uint8
	rcTemporalQ            [WebRTCMaxTemporalLayers]uint8
	rcTemporalBuffer       [WebRTCMaxTemporalLayers]int
	rcTemporalRecentBits   [WebRTCMaxTemporalLayers][2]int
	rcTemporalPerFrameBits [WebRTCMaxTemporalLayers]int

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

	// fusedPipeline forces the legacy fused (search+write interleaved)
	// P-frame tile coder instead of the decision/write split. Test-only: the
	// split must produce byte-identical bitstreams, and the differential test
	// uses this switch to encode both ways.
	fusedPipeline bool

	// wavefront runs the split pipeline's decision pass over SB rows with the
	// top-right dependency (pframe_wavefront.go) on single-tile inter frames;
	// wavefrontLanes is its worker budget (0 disables). Only e.pc — the
	// single-tile coder — is wired to it: multi-tile frames parallelize
	// across tile columns instead.
	wavefront      pframeWavefront
	wavefrontLanes int
	lf           loopFilterApplier
	cdefApp      cdefApplier
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
	filterDone    chan error
	filterWork    chan struct{}
	filterStarted bool
	filterPending bool
	filterErr     error
	filterParams  struct {
		out     *SourceFrame420
		lfLevel uint8
		cdef    parser.CDEFParams
	}

	// frameCtx chains the adapted symbol contexts across the base layer:
	// each non-droppable frame names its predecessor through
	// primary_ref_frame and starts from the saved state instead of the
	// defaults. haveCtx arms after the first keyframe.
	frameCtx     frameCDFs
	haveCtx      bool
	prevLFDeltas LoopFilterDeltas

	// interpOnlyRegular carries libaom's fix_interp_filter observation
	// (av1/encoder/encoder_utils.c) one base frame forward: when a base
	// frame's filter search lands every coded block on REGULAR, the next
	// base frame collapses its header to the fixed EIGHTTAP filter and pays
	// no per-block filter syntax; the shadow search on such frames re-arms
	// SWITCHABLE as soon as any block would pick another filter.
	interpOnlyRegular bool
	refState          parser.ReferenceState

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
	sceneCutKeyframes bool

	content                 ContentHint
	screenContentSelectable bool

	decisionStatsEnabled bool
	decisionStats        EncoderDecisionStats
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
		qIndex: qIndex, goldenEvery: 16, sceneCutKeyframes: true,
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

// SetSceneCutKeyframes controls whether the encoder may promote a delta frame
// to a keyframe when the motion search sees a hard cut.
func (e *VideoEncoder) SetSceneCutKeyframes(enabled bool) {
	e.sceneCutKeyframes = enabled
}

// SetContentHint selects the content mode used for subsequently emitted AV1
// frame headers when screen-content selection is enabled.
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

// SetScreenContentSelection controls whether sequence headers allow per-frame
// screen-content signaling. WebRTC streams enable this so camera/screen
// switches can be applied without forcing a new coded sequence.
func (e *VideoEncoder) SetScreenContentSelection(enabled bool) {
	if e != nil {
		e.screenContentSelectable = enabled
	}
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
	if e != nil && e.screenContentSelectable {
		seq.SeqForceScreenContentTools = SequenceSelectScreenContentTools
		seq.SeqForceIntegerMV = SequenceSelectIntegerMV
	}
	return seq
}

func (e *VideoEncoder) parserColorConfig() parser.ColorConfig {
	if e == nil {
		return parserColorConfig(defaultVideoColorConfig())
	}
	return parserColorConfig(e.color)
}

func (e *VideoEncoder) applyContentHintToFrameHeader(prefix *FrameHeaderPrefix) {
	if e == nil || !e.screenContentSelectable || prefix == nil {
		return
	}
	screen := e.content == ContentScreen
	prefix.AllowScreenContentTools = screen
	if prefix.FrameType.keyOrIntra() {
		prefix.ForceIntegerMV = true
		return
	}
	prefix.ForceIntegerMV = screen
}

// copyFrameInto deep-copies src into dst, reusing dst's buffers when sized.
func copyFrameInto(dst *SourceFrame420, src SourceFrame420) {
	if len(dst.Y) != len(src.Y) {
		dst.Y = make([]byte, len(src.Y))
		dst.U = make([]byte, len(src.U))
		dst.V = make([]byte, len(src.V))
	}
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

func (e *VideoEncoder) runHME(src SourceFrame420) {
	if e.singleThread {
		e.hme.runSerial(src)
		return
	}
	e.hme.run(src)
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

func newVideoEncoderCBRWithColor(width, height int, rc RateControlConfig, color SequenceColorConfig) (*VideoEncoder, error) {
	perFrameBits, err := rateControlPerFrameBits(rc)
	if err != nil {
		return nil, err
	}
	e, err := newVideoEncoderWithColor(width, height, rc.MinQIndex/2+rc.MaxQIndex/2, color)
	if err != nil {
		return nil, err
	}
	e.rcEnabled = true
	e.rcTargetBits = rc.TargetBitsPerSecond
	e.rcFramesPerSec = rc.FramesPerSecond
	e.rcPerFrameBits = perFrameBits
	e.rcMinQ = rc.MinQIndex
	e.rcMaxQ = rc.MaxQIndex
	e.resetRCTemporalState()
	return e, nil
}

// SetQIndex switches future frames to fixed-quality encoding without
// disturbing reference state.
func (e *VideoEncoder) SetQIndex(qIndex uint8) error {
	if e == nil {
		return fmt.Errorf("encoder: nil video encoder")
	}
	if qIndex == 0 {
		return fmt.Errorf("encoder: qindex must be non-zero")
	}
	e.rcEnabled = false
	e.qIndex = qIndex
	e.rcTargetBits = 0
	e.rcFramesPerSec = 0
	e.rcPerFrameBits = 0
	e.rcBuffer = 0
	e.rcRecentBits = [2]int{}
	e.rcTemporalQ = [WebRTCMaxTemporalLayers]uint8{}
	e.rcTemporalBuffer = [WebRTCMaxTemporalLayers]int{}
	e.rcTemporalRecentBits = [WebRTCMaxTemporalLayers][2]int{}
	e.rcTemporalPerFrameBits = [WebRTCMaxTemporalLayers]int{}
	return nil
}

func rateControlPerFrameBits(rc RateControlConfig) (int, error) {
	if rc.TargetBitsPerSecond <= 0 || rc.FramesPerSecond <= 0 {
		return 0, fmt.Errorf("encoder: invalid rate control target %d bps @ %d fps", rc.TargetBitsPerSecond, rc.FramesPerSecond)
	}
	if rc.MinQIndex == 0 || rc.MaxQIndex <= rc.MinQIndex {
		return 0, fmt.Errorf("encoder: invalid qindex range [%d, %d]", rc.MinQIndex, rc.MaxQIndex)
	}
	perFrameBits := rc.TargetBitsPerSecond / rc.FramesPerSecond
	if perFrameBits <= 0 {
		perFrameBits = 1
	}
	return perFrameBits, nil
}

// SetRateControlConfig atomically updates the CBR target used for future
// frames without disturbing reference state.
func (e *VideoEncoder) SetRateControlConfig(rc RateControlConfig) error {
	if e == nil {
		return fmt.Errorf("encoder: nil video encoder")
	}
	perFrameBits, err := rateControlPerFrameBits(rc)
	if err != nil {
		return err
	}
	e.rcEnabled = true
	e.rcTargetBits = rc.TargetBitsPerSecond
	e.rcFramesPerSec = rc.FramesPerSecond
	e.rcPerFrameBits = perFrameBits
	e.rcMinQ = rc.MinQIndex
	e.rcMaxQ = rc.MaxQIndex
	if e.qIndex < e.rcMinQ {
		e.qIndex = e.rcMinQ
	} else if e.qIndex > e.rcMaxQ {
		e.qIndex = e.rcMaxQ
	}
	e.rcBuffer = 0
	e.rcRecentBits = [2]int{}
	e.resetRCTemporalState()
	return nil
}

// UpdateRateControlConfig changes compatible CBR settings while preserving the
// controller's learned buffer and temporal-layer q state. Use SetRateControlConfig
// for a hard controller reinitialization.
func (e *VideoEncoder) UpdateRateControlConfig(rc RateControlConfig) error {
	if e == nil {
		return fmt.Errorf("encoder: nil video encoder")
	}
	if !e.rcEnabled {
		return e.SetRateControlConfig(rc)
	}
	if err := applyRateControlConfigPreservingState(rc, e.temporalLayers, e.rcSurplusFrameLimit(), &e.qIndex, &e.rcTargetBits, &e.rcFramesPerSec, &e.rcPerFrameBits, &e.rcMinQ, &e.rcMaxQ, &e.rcBuffer, &e.rcTemporalQ, &e.rcTemporalBuffer, &e.rcTemporalPerFrameBits); err != nil {
		return err
	}
	e.rcEnabled = true
	return nil
}

func (e *VideoEncoder) resetRCTemporalState() {
	resetRateControlTemporalState(e.qIndex, e.rcTargetBits, e.rcFramesPerSec, e.temporalLayers, e.rcPerFrameBits, &e.rcTemporalQ, &e.rcTemporalBuffer, &e.rcTemporalRecentBits, &e.rcTemporalPerFrameBits)
}

// rcUpdate feeds one frame's actual size into the layer-local leaky-bucket
// controller and steps that layer's qindex toward its per-frame budget.
func (e *VideoEncoder) rcUpdate(frameBits int, temporalID uint8) {
	if !e.rcEnabled {
		return
	}
	idx := rateControlTemporalLayerIndex(e.temporalLayers, temporalID)
	if e.temporalLayers > 1 {
		e.rcTemporalQ[idx] = rcUpdateState(frameBits, e.rcTemporalPerFrameBits[idx], e.rcSurplusFrameLimit(), e.rcMinQ, e.rcMaxQ, e.rcTemporalQ[idx], &e.rcTemporalBuffer[idx], &e.rcTemporalRecentBits[idx])
		e.qIndex = e.rcTemporalQ[0]
		return
	}
	e.qIndex = rcUpdateState(frameBits, e.rcPerFrameBits, e.rcSurplusFrameLimit(), e.rcMinQ, e.rcMaxQ, e.qIndex, &e.rcBuffer, &e.rcRecentBits)
}

func (e *VideoEncoder) rcSurplusFrameLimit() int {
	if e.temporalLayers == 2 {
		return 2
	}
	return 8
}

func (e *VideoEncoder) keyframeQIndex() uint8 {
	qIndex := e.qIndex
	perFrameBits := e.rcPerFrameBits
	buffer := e.rcBuffer
	if e.rcEnabled && e.temporalLayers > 1 {
		qIndex = e.rcTemporalQ[0]
		perFrameBits = e.rcTemporalPerFrameBits[0]
		buffer = e.rcTemporalBuffer[0]
	}
	return rcKeyframeQIndex(qIndex, e.rcEnabled, perFrameBits, buffer, e.rcMinQ, e.rcMaxQ)
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
	if e.tileWork != nil {
		close(e.tileWork)
		e.tileWork = nil
		e.tileWorkers = 0
	}
	if e.filterWork != nil {
		close(e.filterWork)
		e.filterWork = nil
		e.filterStarted = false
	}
	e.lf.close()
	e.cdefApp.close()
	e.hme.close()
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
	if e.rcEnabled {
		e.resetRCTemporalState()
	}
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
	// The hierarchical coarse search doubles as the scene-cut detector:
	// when most regions find no quarter-res match within reach, predictive
	// coding would cost near-keyframe bits at worse quality, so restart the
	// chain with a real keyframe instead. The spacing guard keeps noisy
	// content from forcing key storms.
	e.runHME(src)
	if e.sceneCutKeyframes && e.frameIndex >= 4 && e.hme.cutDetected() {
		return e.Encode(src, true)
	}
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
	// Keyframe quality boost: every later frame predicts (directly or
	// transitively) from the key, so it earns a finer quantizer than the
	// working point; rate control pays the debt back over the next frames
	// through the buffer.
	keyQ := e.keyframeQIndex()
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
	e.haveKey = true
	e.frameIndex = 1
	// The first inter frame after a keyframe starts from default contexts
	// (primary_ref NONE); chaining arms from its own state.
	e.haveCtx = false
	e.interpOnlyRegular = false
	e.haveCtxT1 = false
	e.lastTemporalID = 0
	// A shown keyframe refreshes every reference slot, so slot 1 (the GOLDEN
	// name) now holds the keyframe; seed the search copy.
	if e.goldenEvery > 0 {
		copyFrameInto(&e.golden, recon)
		e.sinceGoldenFresh = 0
	}
	e.rcUpdate(len(tu)*8, 0)
	return tu, nil
}

func (e *VideoEncoder) encodeReferencePFrameWithSequenceMax(src SourceFrame420, ref SourceFrame420, codedRefBuffer uint8, settings FrameEncodeSettings, maxWidth, maxHeight int) ([]byte, error) {
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

	hmeArmed := e.hme.armed
	e.runHME(src)
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
	effQ := e.layerQIndex(settings.TemporalID)
	header, _ := repeatPFrameHeader(src.Width, src.Height, effQ, refresh)
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
	// This path codes a fixed EIGHTTAP frame filter; clear any switchable
	// filter state a previous base frame left on the reused coders.
	e.pc.st.interpSearch = false
	e.pc.st.interpShadow = false
	e.pc.fusedPipeline = e.fusedPipeline
	e.configurePCWavefront()
	for t := range e.tilePCs {
		e.tilePCs[t].st.interpSearch = false
		e.tilePCs[t].st.interpShadow = false
		e.tilePCs[t].fusedPipeline = e.fusedPipeline
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
	}
	e.haveKey = true
	e.frameIndex++
	e.haveCtx = false
	e.interpOnlyRegular = false
	e.haveCtxT1 = false
	e.lastTemporalID = settings.TemporalID
	if !hmeArmed {
		e.hme.prime(src)
	}
	if e.goldenEvery > 0 {
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
	frames := e.temporalLayers
	if frames < 2 {
		frames = 2
	} else {
		frames *= 2
	}
	if e.goldenEvery > 0 && frames < e.goldenEvery+1 {
		frames = e.goldenEvery + 1
	}
	for i := 0; i < frames; i++ {
		if _, _, err := e.Encode(src, false); err != nil {
			return err
		}
	}
	if err := e.joinFilter(); err != nil {
		return err
	}
	e.haveKey = false
	e.haveCtx = false
	e.interpOnlyRegular = false
	e.haveCtxT1 = false
	e.frameIndex = 0
	e.lastTemporalID = 0
	e.qIndex = savedQ
	e.rcBuffer = 0
	e.rcRecentBits = [2]int{}
	e.sinceGoldenFresh = 0
	e.hme.armed = false
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
	for ; e.tileWorkers < workers; e.tileWorkers++ {
		go func() {
			for work := range e.tileWork {
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
	e.filterDone = make(chan error, 1)
	e.filterWork = make(chan struct{}, 1)
	go func() {
		for range e.filterWork {
			p := &e.filterParams
			e.filterDone <- e.applyInLoopFilters(p.out, p.lfLevel, p.cdef)
		}
	}()
	e.filterStarted = true
}

func (e *VideoEncoder) applyInLoopFilters(out *SourceFrame420, lfLevel uint8, cdef parser.CDEFParams) error {
	lf := parser.LoopFilterParams{
		LevelY: [2]uint8{lfLevel, lfLevel},
		LevelU: lfLevel,
		LevelV: lfLevel,
	}
	applyLF := e.lf.apply
	applyCDEF := e.cdefApp.apply
	if e.singleThread {
		applyLF = e.lf.applySerial
		applyCDEF = e.cdefApp.applySerial
	}
	if err := applyLF(out, lf); err != nil {
		return fmt.Errorf("loop filter apply: %w", err)
	}
	if err := applyCDEF(out, cdef, &e.lf.filtMap); err != nil {
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

func (e *VideoEncoder) layerQIndex(temporalID uint8) uint8 {
	if e.rcEnabled && e.temporalLayers > 1 {
		return rateControlTemporalLayerQIndex(e.rcTemporalQ, e.rcMinQ, e.rcMaxQ, e.temporalLayers, temporalID)
	}
	return e.qIndex
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
	if isT1 {
		refresh = 0x04 // slot 2 carries the middle layer for its T2 leaves
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
	effQ := e.layerQIndex(temporalID)
	header, refState := repeatPFrameHeader(src.Width, src.Height, effQ, refresh)
	e.refState = refState
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
	if lfLevel > 0 {
		if !e.lf.bound {
			if err := e.lf.init(src.Width, src.Height); err != nil {
				return nil, fmt.Errorf("loop filter init: %w", err)
			}
		}
		if err := e.lf.reset(); err != nil {
			return nil, fmt.Errorf("loop filter reset: %w", err)
		}
		header.LoopFilter.LevelY = [2]uint8{lfLevel, lfLevel}
		header.LoopFilter.LevelU = lfLevel
		header.LoopFilter.LevelV = lfLevel
		// The q-derived CDEF strengths ride the same gate: the loop-filter
		// records double as the index map and per-block skip source.
		header.CDEF = cdefHeaderParams(effQ, false)
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
	if lfLevel > 0 {
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
	// The light-PD1 MDS0 candidate decision runs on every inter frame, with
	// droppable leaves on the lighter detector-gated level, mirroring
	// SVT-AV1 preset 12's temporal-layer modulation (enc_mode_config.c:
	// pic_lpd1_lvl = is_base ? 4 : 7).
	if droppable {
		e.pc.st.mds0Level = 2
	} else {
		e.pc.st.mds0Level = 1
	}
	e.pc.fusedPipeline = e.fusedPipeline
	e.configurePCWavefront()
	for t := range e.tilePCs {
		e.tilePCs[t].st.mds0Level = e.pc.st.mds0Level
		e.tilePCs[t].fusedPipeline = e.fusedPipeline
	}
	refRecon := e.recon
	if afterT1 {
		refRecon = e.t1Recon
	}
	// Base frames run the per-block switchable interpolation filter search,
	// so their headers signal SWITCHABLE (SVT-AV1 enc_mode_config.c:
	// interpolation_search_level stays 4 for is_base pictures at preset 12
	// and frm_hdr->interpolation_filter = SWITCHABLE). Droppable leaves,
	// integer-MV screen frames, scaled-reference frames, and non-420 paths
	// keep the fixed EIGHTTAP header and code no filter symbols.
	// A base frame whose predecessor used only REGULAR keeps the fixed
	// EIGHTTAP header (libaom fix_interp_filter, av1/encoder/encoder_utils.c,
	// applied with one frame of lag in this one-pass pipeline) and runs the
	// search in shadow so a content change re-arms SWITCHABLE.
	interpBase := !droppable && videoColorIs420(color) && !header.Prefix.ForceIntegerMV &&
		refRecon.Width == src.Width && refRecon.Height == src.Height
	interpSearch := interpBase && !e.interpOnlyRegular
	if interpSearch {
		header.Tile.InterpolationFilter = InterpolationSwitchable
	}
	e.pc.st.interpSearch = interpSearch
	e.pc.st.interpShadow = interpBase && !interpSearch
	e.pc.st.interpDual = seq.EnableDualFilter
	for t := range e.tilePCs {
		e.tilePCs[t].st.interpSearch = interpSearch
		e.tilePCs[t].st.interpShadow = e.pc.st.interpShadow
		e.tilePCs[t].st.interpDual = seq.EnableDualFilter
	}
	referenceMode := parser.ReferenceModeSingle
	if golden != nil && compoundGoldenLikely(&e.pc.st, src, refRecon, golden) {
		header.TransformRef.ReferenceMode = ReferenceModeSelect
		referenceMode = parser.ReferenceModeSelect
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
	// fix_interp_filter (libaom av1/encoder/encoder_utils.c): when a
	// switchable (or shadow) base frame's search used only one filter and it
	// is REGULAR, the next base frame codes the fixed EIGHTTAP header; any
	// other winner re-arms SWITCHABLE. Frames with no searched blocks keep
	// their current state, as libaom keeps SWITCHABLE on zero counts.
	if interpBase {
		var used [3]uint32
		if nTiles == 1 {
			used = e.pc.st.interpUsed
		} else {
			for t := 0; t < nTiles; t++ {
				for f := range used {
					used[f] += e.tilePCs[t].st.interpUsed[f]
				}
			}
		}
		if used[0]+used[1]+used[2] > 0 {
			e.interpOnlyRegular = used[1] == 0 && used[2] == 0
		}
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
	filterCDEF := cdefParserParams(header.CDEF)
	if lfLevel > 0 && !e.singleThread {
		// The filters run behind the temporal-unit assembly and the next
		// frame's source-only stages; everything reading the filtered
		// planes or the applier state joins first.
		e.startFilterWorker()
		e.filterParams.out = out
		e.filterParams.lfLevel = lfLevel
		e.filterParams.cdef = filterCDEF
		e.filterPending = true
		e.filterWork <- struct{}{}
	}
	tu, err := assembleInterTU(seq, header, payloads, temporalID, &e.tuGroup, &e.tuScratch)
	if err != nil {
		return nil, err
	}
	if lfLevel > 0 && e.singleThread {
		if err := e.applyInLoopFilters(out, lfLevel, filterCDEF); err != nil {
			return nil, err
		}
	}
	e.collectDecisionStats(false, nTiles, false)
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
	r := e.lastRecon
	if r.Y == nil {
		r = e.recon
	}
	r.Width, r.Height = e.renderWidth, e.renderHeight
	return r
}
