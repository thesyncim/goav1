package encoder

import (
	"fmt"
	"sync"

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
	hme          hmeState

	temporalLayers int
	t2Recon        SourceFrame420
	frameCtxT1     frameCDFs
	haveCtxT1      bool
	frameIndex     int
	t1Recon        SourceFrame420
	lastRecon      SourceFrame420

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
		qIndex: qIndex, goldenEvery: 32,
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
		return 4
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
	e.rcBuffer += e.rcPerFrameBits - frameBits
	// Clamp the buffer so one huge keyframe cannot dominate forever.
	if limit := 8 * e.rcPerFrameBits; e.rcBuffer < -limit {
		e.rcBuffer = -limit
	} else if e.rcBuffer > limit {
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
		tu, recon, err := encodeKeyframeFiltered(src, e.qIndex, &e.lf, e.renderWidth, e.renderHeight, &e.keyRecon, func(t int) *pframeCoder {
			if t == 0 {
				return &e.pc
			}
			return &e.tilePCs[t]
		})
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
func (e *VideoEncoder) encodePReusing(src SourceFrame420, temporalID uint8) ([]byte, error) {
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
	header, refState := repeatPFrameHeader(src.Width, src.Height, e.qIndex, refresh)
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
		lfLevel = filterLevelFromQIndex(e.qIndex, false)
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
	payloads := make([]TilePayload, nTiles)
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
	if nTiles == 1 {
		data, err := e.pc.encodeTile(src, refRecon, golden, out, e.qIndex, prevCtx, 0, uint16(src.Width/4))
		if err != nil {
			return nil, fmt.Errorf("encode tile: %w", err)
		}
		payloads[0].Data = data
	} else {
		// One goroutine per tile column beyond the first: tiles share the
		// read-only source and reference planes and write disjoint column
		// ranges of the output reconstruction. Tile zero runs on the calling
		// goroutine so a two-tile frame pays a single handoff.
		miCols := uint16(src.Width / 4)
		bounds := func(t int) (uint16, uint16) {
			c0 := header.Tile.ColStartSB[t] * 16
			c1 := header.Tile.ColStartSB[t+1] * 16
			if c1 > miCols {
				c1 = miCols
			}
			return c0, c1
		}
		var wg sync.WaitGroup
		errs := make([]error, nTiles)
		for t := 1; t < nTiles; t++ {
			colStart, colEnd := bounds(t)
			wg.Add(1)
			go func(t int, c0, c1 uint16) {
				defer wg.Done()
				data, err := e.tilePCs[t].encodeTile(src, refRecon, golden, out, e.qIndex, prevCtx, c0, c1)
				if err != nil {
					errs[t] = err
					return
				}
				payloads[t].Data = data
			}(t, colStart, colEnd)
		}
		c0, c1 := bounds(0)
		data, err := e.tilePCs[0].encodeTile(src, refRecon, golden, out, e.qIndex, prevCtx, c0, c1)
		wg.Wait()
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
		if err := e.lf.apply(out, parser.LoopFilterParams{
			LevelY: [2]uint8{lfLevel, lfLevel},
			LevelU: lfLevel,
			LevelV: lfLevel,
		}); err != nil {
			return nil, fmt.Errorf("loop filter apply: %w", err)
		}
	}
	tu, err := assembleInterTU(seq, header, payloads, temporalID)
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
			if err := e.frameCtxT1.InitDefault(e.qIndex); err != nil {
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
	r := e.lastRecon
	if r.Y == nil {
		r = e.recon
	}
	r.Width, r.Height = e.renderWidth, e.renderHeight
	return r
}
