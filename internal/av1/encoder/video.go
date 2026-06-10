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
	width, height int
	qIndex        uint8
	recon         SourceFrame420
	haveKey       bool

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

	temporalLayers int
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
	if width <= 0 || height <= 0 || width%8 != 0 || height%8 != 0 {
		return nil, fmt.Errorf("encoder: dimensions must be positive multiples of 8, got %dx%d", width, height)
	}
	if qIndex == 0 {
		return nil, fmt.Errorf("encoder: qindex must be non-zero")
	}
	e := &VideoEncoder{width: width, height: height, qIndex: qIndex, goldenEvery: 32}
	e.tileColsLog2 = defaultTileColsLog2(width)
	return e, nil
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
	if n != 1 && n != 2 {
		return fmt.Errorf("encoder: unsupported temporal layer count %d", n)
	}
	e.temporalLayers = n
	return nil
}

// TemporalID reports the temporal layer the next frame will be coded in.
func (e *VideoEncoder) TemporalID() uint8 {
	if e.temporalLayers == 2 && e.haveKey && e.frameIndex%2 == 1 {
		return 1
	}
	return 0
}

// Encode encodes one frame and returns its temporal unit plus whether it was
// coded as a keyframe. The first frame (and any frame with forceKey set) is a
// keyframe; every other frame predicts from the latest layer-0 reconstruction.
func (e *VideoEncoder) Encode(src SourceFrame420, forceKey bool) ([]byte, bool, error) {
	if src.Width != e.width || src.Height != e.height {
		return nil, false, fmt.Errorf("encoder: frame %dx%d does not match stream %dx%d", src.Width, src.Height, e.width, e.height)
	}
	if !e.haveKey || forceKey {
		tu, recon, err := EncodeKeyframe(src, e.qIndex)
		if err != nil {
			return nil, false, err
		}
		e.recon = recon
		e.lastRecon = recon
		e.haveKey = true
		e.frameIndex = 1
		// The first inter frame after a keyframe starts from default
		// contexts (primary_ref NONE); chaining arms from its own state.
		e.haveCtx = false
		// A shown keyframe refreshes every reference slot, so slot 1 (the
		// GOLDEN name) now holds the keyframe; seed the search copy.
		if e.goldenEvery > 0 {
			copyFrameInto(&e.golden, recon)
			e.sinceGoldenFresh = 0
		}
		e.rcUpdate(len(tu) * 8)
		return tu, true, nil
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
	var out *SourceFrame420
	if droppable {
		out = &e.t1Recon
	} else {
		out = &e.reconBufs[e.reconIdx]
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
	if droppable {
		refresh = 0 // layer-1 frames are never referenced
	} else if e.goldenEvery > 0 {
		e.sinceGoldenFresh++
		if e.sinceGoldenFresh >= e.goldenEvery {
			refresh |= 0x02 // this frame becomes the new golden anchor
			refreshGolden = true
		}
	}
	seq := losslessKeyframeSequence(src.Width, src.Height)
	header, refState := repeatPFrameHeader(src.Width, src.Height, e.qIndex, refresh)
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
	var prevCtx *frameCDFs
	if e.haveCtx {
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
	if nTiles == 1 {
		data, err := e.pc.encodeTile(src, e.recon, golden, out, e.qIndex, prevCtx, 0, uint16(src.Width/4))
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
				data, err := e.tilePCs[t].encodeTile(src, e.recon, golden, out, e.qIndex, prevCtx, c0, c1)
				if err != nil {
					errs[t] = err
					return
				}
				payloads[t].Data = data
			}(t, colStart, colEnd)
		}
		c0, c1 := bounds(0)
		data, err := e.tilePCs[0].encodeTile(src, e.recon, golden, out, e.qIndex, prevCtx, c0, c1)
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
	if e.lastRecon.Y != nil {
		return e.lastRecon
	}
	return e.recon
}
