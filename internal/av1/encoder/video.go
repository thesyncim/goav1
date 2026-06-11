package encoder

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/parser"
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
	padded                    SourceFrame420
	keyRecon                  SourceFrame420
	qIndex                    uint8
	recon                     SourceFrame420
	haveKey                   bool

	rcEnabled      bool
	rcPerFrameBits int
	rcBuffer       int
	// rcRecentBits holds the last two frames' coded sizes (one full
	// layer pair), the static-content signal for the layer offset.
	rcRecentBits   [2]int
	rcMinQ, rcMaxQ uint8

	pc        pframeCoder
	reconBufs [2]SourceFrame420
	reconIdx  int

	// Tile-column parallelism: tileColsLog2 > 0 splits inter frames into
	// uniform tile columns, each encoded by its own coder (independent CDFs
	// and entropy stream per AV1 tile semantics) on its own goroutine.
	tileColsLog2 uint8
	tilePCs      []pframeCoder
	lf           loopFilterApplier
	cdefApp      cdefApplier
	hme          hmeState

	temporalLayers int
	t2Recon        SourceFrame420
	frameCtxT1     frameCDFs
	haveCtxT1      bool
	frameIndex     int
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
	tileWork      chan int
	tileDone      chan struct{}
	tileStarted   bool
	tileJobParams struct {
		src, refRecon SourceFrame420
		golden        *SourceFrame420
		out           *SourceFrame420
		effQ          uint8
		prevCtx       *frameCDFs
		referenceMode parser.ReferenceMode
		tile          TileInfo
		miCols        uint16
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
	frameCtx frameCDFs
	haveCtx  bool

	// Golden anchor: slot 1 holds an older high-quality reference refreshed
	// every goldenEvery base-layer frames (0 disables block-level use). The
	// encoder keeps its own copy for motion search.
	golden           SourceFrame420
	goldenEvery      int
	sinceGoldenFresh int
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

// NewVideoEncoder creates a streaming encoder. Dimensions must be positive
// multiples of 64 (the current P-frame constraint) and qIndex non-zero.
func NewVideoEncoder(width, height int, qIndex uint8) (*VideoEncoder, error) {
	if width < 16 || height < 16 || width%2 != 0 || height%2 != 0 {
		return nil, fmt.Errorf("encoder: dimensions must be even and at least 16x16, got %dx%d", width, height)
	}
	if qIndex == 0 {
		return nil, fmt.Errorf("encoder: qindex must be non-zero")
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
		qIndex: qIndex, goldenEvery: 16,
	}
	e.tileColsLog2 = defaultTileColsLog2(codedW)
	return e, nil
}

// padSource copies src into the encoder's coded-size scratch frame,
// replicating the last source column and row across the padding.
func (e *VideoEncoder) padSource(src SourceFrame420) SourceFrame420 {
	cw, ch := e.width/2, e.height/2
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
	padPlane(e.padded.U, cw, src.U, src.ChromaStride, (src.Width+1)/2, (src.Height+1)/2, cw, ch)
	padPlane(e.padded.V, cw, src.V, src.ChromaStride, (src.Width+1)/2, (src.Height+1)/2, cw, ch)
	return e.padded
}

// SetGoldenInterval sets how many base-layer inter frames pass between
// golden-anchor refreshes; zero disables golden references entirely.
func (e *VideoEncoder) SetGoldenInterval(n int) {
	e.goldenEvery = n
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
// goroutine handoff, four at HD, sixteen at full HD - narrow tiles balance
// the per-frame join across a modern core count for a fifth of a percent of
// rate.
func defaultTileColsLog2(width int) uint8 {
	switch {
	case width >= 1920:
		return 3
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
	log2 := uint8(0)
	for (2 << log2) <= cols {
		log2++
	}
	e.tileColsLog2 = log2
}

// NewVideoEncoderCBR creates a streaming encoder under CBR rate control. The
// encoder starts in the middle of the configured qindex range and adjusts the
// working qindex after every frame from the bit-budget buffer.
func NewVideoEncoderCBR(width, height int, rc RateControlConfig) (*VideoEncoder, error) {
	if rc.TargetBitsPerSecond <= 0 || rc.FramesPerSecond <= 0 {
		return nil, fmt.Errorf("encoder: invalid rate control target %d bps @ %d fps", rc.TargetBitsPerSecond, rc.FramesPerSecond)
	}
	if rc.MinQIndex == 0 || rc.MaxQIndex <= rc.MinQIndex {
		return nil, fmt.Errorf("encoder: invalid qindex range [%d, %d]", rc.MinQIndex, rc.MaxQIndex)
	}
	e, err := NewVideoEncoder(width, height, rc.MinQIndex/2+rc.MaxQIndex/2)
	if err != nil {
		return nil, err
	}
	e.rcEnabled = true
	e.rcPerFrameBits = rc.TargetBitsPerSecond / rc.FramesPerSecond
	e.rcMinQ = rc.MinQIndex
	e.rcMaxQ = rc.MaxQIndex
	return e, nil
}

// rcUpdate feeds one frame's actual size into the leaky-bucket controller and
// steps the working qindex toward the per-frame budget. The step grows with
// the buffer excursion so keyframe overshoot recovers within a few frames.
func (e *VideoEncoder) rcUpdate(frameBits int) {
	if !e.rcEnabled {
		return
	}
	e.rcRecentBits[0], e.rcRecentBits[1] = e.rcRecentBits[1], frameBits

	e.rcBuffer += e.rcPerFrameBits - frameBits
	// Asymmetric clamp: debt (a boosted keyframe) stays on the books for
	// a second so the controller actually repays it, while surplus stays
	// short so a static stretch cannot bank bits for a burst.
	if limit := 24 * e.rcPerFrameBits; e.rcBuffer < -limit {
		e.rcBuffer = -limit
	}
	if limit := 8 * e.rcPerFrameBits; e.rcBuffer > limit {
		e.rcBuffer = limit
	}
	// Proportional step: a quarter qindex unit per quarter-frame of buffer
	// excursion, clamped so a clamped buffer moves the quantizer twelve
	// units per frame at most. Small excursions round to zero, which keeps
	// the steady-state deadband.
	q := int(e.qIndex)
	step := -e.rcBuffer * 4 / e.rcPerFrameBits
	if step > 12 {
		step = 12
	} else if step < -12 {
		step = -12
	}
	q += step
	if q < int(e.rcMinQ) {
		q = int(e.rcMinQ)
	}
	if q > int(e.rcMaxQ) {
		q = int(e.rcMaxQ)
	}
	e.qIndex = uint8(q)
}

func (e *VideoEncoder) keyframeQIndex() uint8 {
	keyQ := e.qIndex
	if !e.rcEnabled {
		return keyQ
	}
	// The boost scales with the per-frame budget: rich streams can
	// repay a much finer key inside the controller's debt horizon,
	// while starved ones would queue behind an oversized key.
	boost := e.rcPerFrameBits / 1600
	if boost > 50 {
		boost = 50
	} else if boost < 12 {
		boost = 12
	}
	if e.rcBuffer < 0 {
		horizon := 24 * e.rcPerFrameBits
		credit := horizon + e.rcBuffer
		if credit < 0 {
			credit = 0
		}
		// Keep most of the reference-quality boost, but trim it as the
		// leaky bucket fills with debt so repeated keyframe requests do not
		// repeatedly spend the same recovery budget.
		boost = boost * (3*horizon + credit) / (4 * horizon)
	}
	if int(keyQ)-boost > int(e.rcMinQ) {
		keyQ -= uint8(boost)
	} else {
		keyQ = e.rcMinQ
	}
	// Forced keyframes can arrive while inter frames are pinned at
	// max Q; cap key quality in the top third of the configured
	// range so the recovery picture stays useful without spending
	// a full high-quality keyframe during debt.
	if e.rcBuffer >= 0 {
		maxKeyQ := int(e.rcMinQ) + (int(e.rcMaxQ)-int(e.rcMinQ))*2/3
		if int(keyQ) > maxKeyQ {
			keyQ = uint8(maxKeyQ)
		}
	}
	return keyQ
}

// QIndex reports the working qindex the next frame will use.
func (e *VideoEncoder) QIndex() uint8 {
	return e.qIndex
}

// SetTemporalLayers selects the temporal-layer count: 1 (default) or 2 for
// the WebRTC L1T2 pattern, where odd frames are droppable (they refresh no
// reference slot and always predict from the latest layer-0 frame).
func (e *VideoEncoder) SetTemporalLayers(n int) error {
	if n < 1 || n > 3 {
		return fmt.Errorf("encoder: unsupported temporal layer count %d", n)
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
	if src.Width != e.renderWidth || src.Height != e.renderHeight {
		return nil, false, fmt.Errorf("encoder: frame %dx%d does not match stream %dx%d", src.Width, src.Height, e.renderWidth, e.renderHeight)
	}
	if e.renderWidth != e.width || e.renderHeight != e.height {
		src = e.padSource(src)
	}
	if !e.haveKey || forceKey {
		// Periodic keyframes reuse the stream's recon buffer and tile-coder
		// pool, so a scene cut allocates only its temporal unit.
		nTiles := 1
		if e.tileColsLog2 > 0 {
			nTiles = 1 << e.tileColsLog2
		}
		if len(e.tilePCs) < nTiles {
			e.tilePCs = make([]pframeCoder, nTiles)
		}
		// The keyframe path reuses the filter appliers a backgrounded pass
		// may still hold.
		if err := e.joinFilter(); err != nil {
			return nil, false, err
		}
		// Keyframe quality boost: every later frame predicts (directly or
		// transitively) from the key, so it earns a finer quantizer than
		// the working point; rate control pays the debt back over the next
		// frames through the buffer.
		keyQ := e.keyframeQIndex()
		tu, recon, err := encodeKeyframeFilteredTiles(src, keyQ, &e.lf, e.renderWidth, e.renderHeight, &e.keyRecon, func(t int) *pframeCoder {
			if t == 0 {
				return &e.pc
			}
			return &e.tilePCs[t]
		}, &e.cdefApp, e.tileColsLog2)
		if err != nil {
			return nil, false, err
		}
		e.hme.prime(src)
		e.recon = recon
		e.lastRecon = recon
		e.haveKey = true
		e.frameIndex = 1
		// The first inter frame after a keyframe starts from default
		// contexts (primary_ref NONE); chaining arms from its own state.
		e.haveCtx = false
		e.haveCtxT1 = false
		// A shown keyframe refreshes every reference slot, so slot 1 (the
		// GOLDEN name) now holds the keyframe; seed the search copy.
		if e.goldenEvery > 0 {
			copyFrameInto(&e.golden, recon)
			e.sinceGoldenFresh = 0
		}
		e.rcUpdate(len(tu) * 8)
		return tu, true, nil
	}
	// The hierarchical coarse search doubles as the scene-cut detector:
	// when most regions find no quarter-res match within reach, predictive
	// coding would cost near-keyframe bits at worse quality, so restart the
	// chain with a real keyframe instead. The spacing guard keeps noisy
	// content from forcing key storms.
	e.hme.run(src)
	if e.frameIndex >= 4 && e.hme.cutDetected() {
		return e.Encode(src, true)
	}
	tid := e.TemporalID()
	tu, err := e.encodePReusing(src, tid)
	if err != nil {
		return nil, false, err
	}
	e.frameIndex++
	e.rcUpdate(len(tu) * 8)
	return tu, false, nil
}

// encodePReusing is the steady-state P-frame path: it reuses the encoder-owned
// coder state and double-buffered reconstruction planes so per-frame work
// allocates only the emitted temporal unit.
// Prewarm runs the whole encode machinery once on a throwaway frame and
// resets the stream state, so every lazily sized buffer, worker pool and
// per-coder scratch exists before the first real frame: steady-state encoding
// allocates nothing and the first frame pays no initialization latency.
func (e *VideoEncoder) Prewarm() error {
	src := SourceFrame420{
		Y:            make([]byte, e.width*e.height),
		U:            make([]byte, e.width*e.height/4),
		V:            make([]byte, e.width*e.height/4),
		YStride:      e.width,
		ChromaStride: e.width / 2,
		Width:        e.renderWidth,
		Height:       e.renderHeight,
	}
	savedQ := e.qIndex
	// Size the assembly buffers for real content up front; the throwaway
	// frame below is black and would leave them near-empty.
	if bound := e.width * e.height / 2; cap(e.tuScratch) < bound {
		e.tuScratch = make([]byte, 0, bound)
		e.tuGroup = make([]byte, 0, bound)
	}
	if _, _, err := e.Encode(src, true); err != nil {
		return err
	}
	// One frame per temporal layer exercises every reconstruction buffer.
	frames := e.temporalLayers
	if frames < 2 {
		frames = 2
	} else {
		frames *= 2
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
	e.haveCtxT1 = false
	e.frameIndex = 0
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

// startTileWorkers launches the persistent tile-column workers once; they
// park on the job channel between frames so the steady-state encode path
// spawns no goroutines.
func (e *VideoEncoder) startTileWorkers() {
	if e.tileStarted {
		return
	}
	e.tileWork = make(chan int, 16)
	e.tileDone = make(chan struct{}, 16)
	for range 15 {
		go func() {
			for t := range e.tileWork {
				tj := &e.tileJobParams
				c0, c1 := tileColBounds(tj.tile, t, tj.miCols)
				data, err := e.tilePCs[t].encodeTile(tj.src, tj.refRecon, tj.golden, tj.out, tj.effQ, tj.prevCtx, tj.referenceMode, c0, c1)
				if err != nil {
					e.tileErrs[t] = err
				} else {
					e.payloads[t].Data = data
				}
				e.tileDone <- struct{}{}
			}
		}()
	}
	e.tileStarted = true
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
			if err := e.lf.apply(p.out, parser.LoopFilterParams{
				LevelY: [2]uint8{p.lfLevel, p.lfLevel},
				LevelU: p.lfLevel,
				LevelV: p.lfLevel,
			}); err != nil {
				e.filterDone <- fmt.Errorf("loop filter apply: %w", err)
				continue
			}
			if err := e.cdefApp.apply(p.out, p.cdef, &e.lf.filtMap); err != nil {
				e.filterDone <- fmt.Errorf("cdef apply: %w", err)
				continue
			}
			e.filterDone <- nil
		}
	}()
	e.filterStarted = true
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

// layerQIndexOffset is the quantizer-index boost applied per temporal layer:
// droppable frames are never referenced, so their extra distortion does not
// propagate, while the bits they give back flow (through rate control) to the
// frames every later frame predicts from. The shape mirrors the layered QP
// offsets realtime encoders assign inside a low-delay mini-GOP.
const layerQIndexOffset = 40

// layerQIndex is the effective base quantizer index for a frame at the given
// temporal layer. The offset shrinks as the coarse search reports the scene
// static: the boost pays off by freeing leaf bits for the reference frames,
// and static regions have none to free - their leaves are already nearly
// empty, so the extra quantization is pure damage.
func (e *VideoEncoder) layerQIndex(temporalID uint8) uint8 {
	offset := int(temporalID) * layerQIndexOffset
	if e.rcEnabled && offset > 0 && e.hme.staticFraction() > 192 {
		// Both signals must agree the scene is static: the coarse search
		// seeing mostly clean zero-motion matches separates static from
		// merely cheap-to-code motion, and the recent coded sizes scale
		// how far the offset shrinks.
		recent := (e.rcRecentBits[0] + e.rcRecentBits[1]) / 2
		full := e.rcPerFrameBits / 2
		if recent < full {
			offset = offset * recent / full
		}
	}
	q := int(e.qIndex) + offset
	if q > 255 {
		q = 255
	}
	return uint8(q)
}

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
	afterT1 := e.temporalLayers == 3 && temporalID == 2 && e.frameIndex%4 == 3
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
		if e.sinceGoldenFresh >= e.goldenEvery {
			refresh |= 0x02 // this frame becomes the new golden anchor
			refreshGolden = true
		}
	}
	seq := losslessKeyframeSequence(src.Width, src.Height)
	effQ := e.layerQIndex(temporalID)
	header, refState := repeatPFrameHeader(src.Width, src.Height, effQ, refresh)
	if e.renderWidth != e.width || e.renderHeight != e.height {
		header.Size.RenderWidth = uint32(e.renderWidth)
		header.Size.RenderHeight = uint32(e.renderHeight)
		header.Size.HaveRenderSize = true
	}
	header.References = &refState
	// In-loop deblocking: signal the q-derived filter levels and collect the
	// per-MI records the frame-level pass needs; after the tiles finish the
	// encoder runs the decoder's own loop filter over the reconstruction so
	// recon stays bit-exact with decoder output.
	lfLevel := uint8(0)
	if src.Width*src.Height <= loopFilterMaxArea {
		lfLevel = filterLevelFromQIndex(effQ, false)
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
	if afterT1 && e.haveCtxT1 {
		// Chain from the middle layer's saved state (slot 2 via LAST).
		header.Prefix.ErrorResilientMode = false
		header.Prefix.PrimaryRefFrame = 0
		prev := defaultLoopFilterDeltas()
		header.PreviousLFDeltas = &prev
		prevCtx = &e.frameCtxT1
	} else if !afterT1 && e.haveCtx {
		// Chain the symbol contexts from slot 0's saved frame state. The
		// header then codes loop-filter deltas relative to that frame's,
		// which this encoder keeps at the defaults.
		header.Prefix.ErrorResilientMode = false
		header.Prefix.PrimaryRefFrame = 0
		prev := defaultLoopFilterDeltas()
		header.PreviousLFDeltas = &prev
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
	if cap(e.payloads) < nTiles {
		e.payloads = make([]TilePayload, nTiles)
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
	refRecon := e.recon
	if afterT1 {
		refRecon = e.t1Recon
	}
	referenceMode := parser.ReferenceModeSingle
	if golden != nil && compoundGoldenLikely(&e.pc.st, src, refRecon, golden) {
		header.TransformRef.ReferenceMode = ReferenceModeSelect
		referenceMode = parser.ReferenceModeSelect
	}
	if nTiles == 1 {
		data, err := e.pc.encodeTile(src, refRecon, golden, out, effQ, prevCtx, referenceMode, 0, uint16(src.Width/4))
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
		tj.tile, tj.miCols = header.Tile, miCols
		e.startTileWorkers()
		for t := 1; t < nTiles; t++ {
			e.tileWork <- t
		}
		c0, c1 := tileColBounds(header.Tile, 0, miCols)
		data, err := e.tilePCs[0].encodeTile(src, refRecon, golden, out, effQ, prevCtx, referenceMode, c0, c1)
		for t := 1; t < nTiles; t++ {
			<-e.tileDone
		}
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
	if lfLevel > 0 {
		// The filters run behind the temporal-unit assembly and the next
		// frame's source-only stages; everything reading the filtered
		// planes or the applier state joins first.
		e.startFilterWorker()
		e.filterParams.out = out
		e.filterParams.lfLevel = lfLevel
		e.filterParams.cdef = cdefParserParams(header.CDEF)
		e.filterPending = true
		e.filterWork <- struct{}{}
	}
	tu, err := assembleInterTU(seq, header, payloads, temporalID, &e.tuGroup, &e.tuScratch)
	if err != nil {
		return nil, err
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
