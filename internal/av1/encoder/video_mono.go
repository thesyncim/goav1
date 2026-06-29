package encoder

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// MonochromeVideoEncoder encodes a same-sized native AV1 monochrome stream.
// It intentionally mirrors the public realtime VideoEncoder lifecycle while
// reusing the already-proven monochrome keyframe and P-frame tile coders.
type MonochromeVideoEncoder struct {
	width, height             int
	renderWidth, renderHeight int
	padded                    SourceFrameMono

	qIndex      uint8
	content     ContentHint
	effortLevel int8

	screenContentSelectable bool

	rcEnabled              bool
	rcTargetBits           int
	rcFramesPerSec         int
	rcPerFrameBits         int
	rcBuffer               int
	rcRecentBits           [2]int
	rcMinQ, rcMaxQ         uint8
	rcTemporalQ            [WebRTCMaxTemporalLayers]uint8
	rcTemporalBuffer       [WebRTCMaxTemporalLayers]int
	rcTemporalRecentBits   [WebRTCMaxTemporalLayers][2]int
	rcTemporalPerFrameBits [WebRTCMaxTemporalLayers]int

	pc        pframeCoder
	keyRecon  SourceFrameMono
	recon     SourceFrameMono
	reconBufs [2]SourceFrameMono
	reconIdx  int
	t1Recon   SourceFrameMono
	t2Recon   SourceFrameMono
	lastRecon SourceFrameMono
	golden    SourceFrameMono

	haveKey          bool
	temporalLayers   int
	frameIndex       int
	lastTemporalID   uint8
	goldenEvery      int
	sinceGoldenFresh int

	tileColsLog2 uint8
	tilePCs      []pframeCoder
	payloads     []TilePayload
	tuGroup      []byte
	tuScratch    []byte
}

// NewMonochromeVideoEncoder creates a native monochrome streaming encoder.
// Render dimensions may be odd or non-multiple-of-eight; the bitstream is coded
// at padded dimensions and signals render_size when needed. qIndex must be
// non-zero.
func NewMonochromeVideoEncoder(width, height int, qIndex uint8) (*MonochromeVideoEncoder, error) {
	if width < 16 || height < 16 {
		return nil, fmt.Errorf("encoder: dimensions must be at least 16x16, got %dx%d", width, height)
	}
	if qIndex == 0 {
		return nil, fmt.Errorf("encoder: qindex must be non-zero")
	}
	codedW := (width + 7) &^ 7
	codedH := (height + 7) &^ 7
	return &MonochromeVideoEncoder{
		width: codedW, height: codedH,
		renderWidth: width, renderHeight: height,
		qIndex: qIndex, goldenEvery: 16,
	}, nil
}

// NewMonochromeVideoEncoderCBR creates a native monochrome streaming encoder
// under the same CBR controller used by the 4:2:0 realtime encoder.
func NewMonochromeVideoEncoderCBR(width, height int, rc RateControlConfig) (*MonochromeVideoEncoder, error) {
	perFrameBits, err := rateControlPerFrameBits(rc)
	if err != nil {
		return nil, err
	}
	e, err := NewMonochromeVideoEncoder(width, height, rc.MinQIndex/2+rc.MaxQIndex/2)
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

// SetTemporalLayers selects the temporal-layer count: 1, 2, or 3.
func (e *MonochromeVideoEncoder) SetTemporalLayers(n int) error {
	if n < 1 || n > 3 {
		return fmt.Errorf("encoder: unsupported temporal layer count %d", n)
	}
	e.temporalLayers = n
	if e.rcEnabled {
		e.resetRCTemporalState()
	}
	return nil
}

// SetTileColumns overrides the inter-frame tile column count (rounded down
// to a power of two, clamped to the legal range for the frame size at encode
// time). One column disables multi-tile output.
func (e *MonochromeVideoEncoder) SetTileColumns(cols int) {
	if e != nil {
		e.tileColsLog2 = tileColumnsLog2(cols)
	}
}

// SetGoldenInterval sets how many base-layer inter frames pass between
// golden-anchor refreshes; zero disables golden references entirely.
func (e *MonochromeVideoEncoder) SetGoldenInterval(n int) {
	if e != nil {
		e.goldenEvery = n
	}
}

func (e *MonochromeVideoEncoder) SetMaxThreads(n int) {
	if e == nil {
		return
	}
	if n > 0 {
		e.SetTileColumns(n)
		return
	}
	e.setDefaultTileColumns()
}

func (e *MonochromeVideoEncoder) setDefaultTileColumns() {
	if e != nil {
		e.tileColsLog2 = 0
	}
}

// TemporalID reports the temporal layer the next frame will be coded in.
func (e *MonochromeVideoEncoder) TemporalID() uint8 {
	if e == nil || !e.haveKey {
		return 0
	}
	switch e.temporalLayers {
	case 2:
		if e.frameIndex%2 == 1 {
			return 1
		}
	case 3:
		switch e.frameIndex % 4 {
		case 1, 3:
			return 2
		case 2:
			return 1
		}
	}
	return 0
}

// SetContentHint selects the content mode used for subsequently emitted AV1
// frame headers when screen-content selection is enabled.
func (e *MonochromeVideoEncoder) SetContentHint(content ContentHint) error {
	if e == nil {
		return fmt.Errorf("encoder: nil monochrome video encoder")
	}
	if !content.Valid() {
		return ErrInvalidConfig
	}
	e.content = content
	return nil
}

// SetScreenContentSelection controls whether sequence headers allow per-frame
// screen-content signaling.
func (e *MonochromeVideoEncoder) SetScreenContentSelection(enabled bool) {
	if e != nil {
		e.screenContentSelectable = enabled
	}
}

// SetEffortLevel selects the realtime encoder effort level. The default zero
// preserves the current quality/speed balance; WebRTCMinEffortLevel disables
// subpel search for the fastest valid bitstream path.
func (e *MonochromeVideoEncoder) SetEffortLevel(level int8) error {
	if e == nil {
		return fmt.Errorf("encoder: nil monochrome video encoder")
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

// SetQIndex switches future frames to fixed-quality encoding.
func (e *MonochromeVideoEncoder) SetQIndex(qIndex uint8) error {
	if e == nil {
		return fmt.Errorf("encoder: nil monochrome video encoder")
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

// SetRateControlConfig atomically updates future-frame CBR settings.
func (e *MonochromeVideoEncoder) SetRateControlConfig(rc RateControlConfig) error {
	if e == nil {
		return fmt.Errorf("encoder: nil monochrome video encoder")
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

// QIndex reports the working qindex the next frame will use.
func (e *MonochromeVideoEncoder) QIndex() uint8 {
	if e == nil {
		return 0
	}
	return e.qIndex
}

// Close releases resources. It is safe to call more than once.
func (e *MonochromeVideoEncoder) Close() error {
	return nil
}

// Recon returns the latest reconstruction.
func (e *MonochromeVideoEncoder) Recon() SourceFrameMono {
	if e == nil {
		return SourceFrameMono{}
	}
	if e.lastRecon.Y != nil {
		return e.lastRecon
	}
	return e.recon
}

// Encode encodes one frame and returns its temporal unit plus whether it was
// coded as a keyframe.
func (e *MonochromeVideoEncoder) Encode(src SourceFrameMono, forceKey bool) ([]byte, bool, error) {
	return e.EncodeWithTemporalID(src, forceKey, e.TemporalID())
}

// EncodeWithTemporalID encodes one frame while forcing the temporal layer used
// for an inter frame.
func (e *MonochromeVideoEncoder) EncodeWithTemporalID(src SourceFrameMono, forceKey bool, temporalID uint8) ([]byte, bool, error) {
	if e == nil {
		return nil, false, fmt.Errorf("encoder: nil monochrome video encoder")
	}
	if err := e.validateRenderSource(src); err != nil {
		return nil, false, err
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
	tu, err := e.encodePReusing(src, temporalID)
	if err != nil {
		return nil, false, err
	}
	e.frameIndex++
	e.lastTemporalID = temporalID
	e.rcUpdate(len(tu)*8, temporalID)
	return tu, false, nil
}

// Prewarm sizes reusable buffers without advancing externally visible stream
// state.
func (e *MonochromeVideoEncoder) Prewarm() error {
	if e == nil {
		return nil
	}
	src := SourceFrameMono{
		Y:       make([]byte, e.width*e.height),
		YStride: e.width,
		Width:   e.renderWidth,
		Height:  e.renderHeight,
	}
	savedQ := e.qIndex
	if bound := e.width * e.height / 2; cap(e.tuScratch) < bound {
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
	e.haveKey = false
	e.frameIndex = 0
	e.lastTemporalID = 0
	e.qIndex = savedQ
	e.rcBuffer = 0
	e.rcRecentBits = [2]int{}
	e.sinceGoldenFresh = 0
	e.lastRecon = SourceFrameMono{}
	return nil
}

func (e *MonochromeVideoEncoder) validateRenderSource(src SourceFrameMono) error {
	if src.Width != e.renderWidth || src.Height != e.renderHeight {
		return fmt.Errorf("encoder: frame %dx%d does not match stream %dx%d", src.Width, src.Height, e.renderWidth, e.renderHeight)
	}
	if src.YStride < src.Width {
		return fmt.Errorf("encoder: monochrome Y stride %d is smaller than width %d", src.YStride, src.Width)
	}
	if src.Height > 0 && src.YStride > (int(^uint(0)>>1)-(src.Width-1))/(src.Height-1) {
		return fmt.Errorf("encoder: monochrome Y plane dimensions overflow int")
	}
	need := (src.Height-1)*src.YStride + src.Width
	if len(src.Y) < need {
		return fmt.Errorf("encoder: monochrome Y plane is too short: got %d bytes, need %d", len(src.Y), need)
	}
	return nil
}

func (e *MonochromeVideoEncoder) padSource(src SourceFrameMono) SourceFrameMono {
	if e.padded.Y == nil {
		e.padded = SourceFrameMono{
			Y:       make([]byte, e.width*e.height),
			YStride: e.width,
			Width:   e.width,
			Height:  e.height,
		}
	}
	for y := 0; y < e.height; y++ {
		sy := min(y, src.Height-1)
		drow := e.padded.Y[y*e.width : y*e.width+e.width]
		copy(drow, src.Y[sy*src.YStride:sy*src.YStride+src.Width])
		for x := src.Width; x < e.width; x++ {
			drow[x] = drow[src.Width-1]
		}
	}
	return e.padded
}

func (e *MonochromeVideoEncoder) sequenceHeader(width, height int) SequenceHeader {
	seq := lossyMonochromeKeyframeSequence(width, height)
	if e != nil && e.screenContentSelectable {
		seq.SeqForceScreenContentTools = SequenceSelectScreenContentTools
		seq.SeqForceIntegerMV = SequenceSelectIntegerMV
	}
	return seq
}

func (e *MonochromeVideoEncoder) applyContentHintToFrameHeader(prefix *FrameHeaderPrefix) {
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

func (e *MonochromeVideoEncoder) encodeKeyWithSequenceMax(src SourceFrameMono, maxWidth, maxHeight int) ([]byte, error) {
	keyQ := e.keyframeQIndex()
	seq := e.sequenceHeader(maxWidth, maxHeight)
	header := lossyMonochromeKeyframeHeaderForSequence(seq, src.Width, src.Height, keyQ)
	e.applyContentHintToFrameHeader(&header.Prefix)
	if e.renderWidth != e.width || e.renderHeight != e.height {
		header.Size.RenderWidth = uint32(e.renderWidth)
		header.Size.RenderHeight = uint32(e.renderHeight)
		header.Size.HaveRenderSize = true
	}
	allocMonoFrame(&e.keyRecon, src)
	payloads, err := e.encodeKeyTilePayloads(seq, src, &e.keyRecon, keyQ, &header)
	if err != nil {
		return nil, err
	}
	out, err := e.assembleKeyTU(seq, header, payloads)
	if err != nil {
		return nil, err
	}
	e.recon = e.keyRecon
	e.lastRecon = e.keyRecon
	e.haveKey = true
	e.frameIndex = 1
	e.lastTemporalID = 0
	if e.goldenEvery > 0 {
		copyMonoFrameInto(&e.golden, e.keyRecon)
		e.sinceGoldenFresh = 0
	}
	e.rcUpdate(len(out)*8, 0)
	return out, nil
}

func (e *MonochromeVideoEncoder) encodePReusing(src SourceFrameMono, temporalID uint8) ([]byte, error) {
	droppable := temporalID > 0
	isT1 := e.temporalLayers == 3 && temporalID == 1
	afterT1 := e.temporalLayers == 3 && temporalID == 2 && e.lastTemporalID == 1
	var out *SourceFrameMono
	switch {
	case !droppable:
		out = &e.reconBufs[e.reconIdx]
	case e.temporalLayers == 3 && temporalID == 2:
		out = &e.t2Recon
	default:
		out = &e.t1Recon
	}
	allocMonoFrame(out, src)

	refresh := uint8(0x01)
	refreshGolden := false
	if isT1 {
		refresh = 0x04
	} else if droppable {
		refresh = 0
	} else if e.goldenEvery > 0 {
		e.sinceGoldenFresh++
		if e.sinceGoldenFresh >= e.goldenEvery {
			refresh |= 0x02
			refreshGolden = true
		}
	}
	seq := e.sequenceHeader(src.Width, src.Height)
	effQ := e.layerQIndex(temporalID)
	header, refState := repeatPFrameHeader(src.Width, src.Height, effQ, refresh)
	e.applyContentHintToFrameHeader(&header.Prefix)
	header.References = &refState
	header.CDEF = CDEFParams{}
	if e.renderWidth != e.width || e.renderHeight != e.height {
		header.Size.RenderWidth = uint32(e.renderWidth)
		header.Size.RenderHeight = uint32(e.renderHeight)
		header.Size.HaveRenderSize = true
	}
	ref := e.recon
	if afterT1 {
		if e.t1Recon.Y == nil {
			return nil, ErrInvalidFrame
		}
		ref = e.t1Recon
		header.Size.RefFrameIdx[0] = 2
	}
	var golden *SourceFrameMono
	referenceMode := parser.ReferenceModeSingle
	if e.goldenEvery > 0 && e.golden.Y != nil {
		golden = &e.golden
		src420 := SourceFrame420{Y: src.Y, YStride: src.YStride, Width: src.Width, Height: src.Height}
		ref420 := SourceFrame420{Y: ref.Y, YStride: ref.YStride, Width: ref.Width, Height: ref.Height}
		golden420 := SourceFrame420{Y: golden.Y, YStride: golden.YStride, Width: golden.Width, Height: golden.Height}
		if compoundGoldenLikely(&e.pc.st, src420, ref420, &golden420) {
			header.TransformRef.ReferenceMode = ReferenceModeSelect
			referenceMode = parser.ReferenceModeSelect
		}
	}
	payloads, err := e.encodePTilePayloads(seq, src, ref, golden, out, effQ, referenceMode, &header)
	if err != nil {
		return nil, err
	}
	tu, err := assembleInterTU(seq, header, payloads, temporalID, &e.tuGroup, &e.tuScratch)
	if err != nil {
		return nil, err
	}
	e.lastRecon = *out
	if !droppable {
		e.recon = *out
		e.reconIdx ^= 1
	}
	if refreshGolden {
		copyMonoFrameInto(&e.golden, *out)
		e.sinceGoldenFresh = 0
	}
	return tu, nil
}

func (e *MonochromeVideoEncoder) encodeReferencePFrameWithSequenceMax(src SourceFrameMono, ref SourceFrameMono, codedRefBuffer uint8, settings FrameEncodeSettings, maxWidth, maxHeight int) ([]byte, error) {
	if err := e.validateRenderSource(src); err != nil {
		return nil, err
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

	out := &e.t2Recon
	if settings.UpdateBufferSet {
		out = &e.reconBufs[e.reconIdx]
	}
	allocMonoFrame(out, src)

	refresh := uint8(0)
	if settings.UpdateBufferSet {
		refresh = 1 << settings.UpdateBuffer
	}
	seq := e.sequenceHeader(maxWidth, maxHeight)
	effQ := e.layerQIndex(settings.TemporalID)
	header, refState := repeatPFrameHeader(src.Width, src.Height, effQ, refresh)
	e.applyContentHintToFrameHeader(&header.Prefix)
	header.Prefix.FrameSizeOverride = src.Width != maxWidth || src.Height != maxHeight
	header.Size.RefFrameIdx = [7]uint8{}
	for i := range header.Size.RefFrameIdx {
		header.Size.RefFrameIdx[i] = codedRefBuffer
	}
	refState = referenceStateForFrame(ref.Width, ref.Height)
	header.References = &refState
	header.CDEF = CDEFParams{}
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

	payloads, err := e.encodePTilePayloads(seq, src, ref, nil, out, effQ, parser.ReferenceModeSingle, &header)
	if err != nil {
		return nil, err
	}
	tu, err := assembleInterTU(seq, header, payloads, settings.TemporalID, &e.tuGroup, &e.tuScratch)
	if err != nil {
		return nil, err
	}
	e.lastRecon = *out
	if settings.UpdateBufferSet {
		e.recon = *out
		e.reconIdx ^= 1
	}
	e.haveKey = true
	e.frameIndex++
	e.lastTemporalID = settings.TemporalID
	e.rcUpdate(len(tu)*8, settings.TemporalID)
	return tu, nil
}

func (e *MonochromeVideoEncoder) encodeKeyTilePayloads(seq SequenceHeader, src SourceFrameMono, recon *SourceFrameMono, qIndex uint8, header *IntraFrameHeaderParams) ([]TilePayload, error) {
	nTiles, err := e.configureTilePayloads(seq, src.Width, src.Height, &header.Tile)
	if err != nil {
		return nil, err
	}
	payloads := e.ensureTilePayloads(nTiles)
	miCols := uint16(src.Width / 4)
	for t := 0; t < nTiles; t++ {
		pc := &e.pc
		if t > 0 {
			pc = &e.tilePCs[t]
		}
		c0, c1 := tilePayloadColBounds(header.Tile, t, miCols)
		data, err := pc.encodeMonochromeKeyframeTileWithOptions(src, recon, qIndex, c0, c1, header.Prefix.AllowScreenContentTools)
		if err != nil {
			return nil, fmt.Errorf("encode tile %d: %w", t, err)
		}
		payloads[t].Data = data
	}
	return payloads, nil
}

func (e *MonochromeVideoEncoder) encodePTilePayloads(seq SequenceHeader, src SourceFrameMono, ref SourceFrameMono, golden *SourceFrameMono, out *SourceFrameMono, qIndex uint8, referenceMode parser.ReferenceMode, header *InterFrameHeaderParams) ([]TilePayload, error) {
	nTiles, err := e.configureTilePayloads(seq, src.Width, src.Height, &header.Tile)
	if err != nil {
		return nil, err
	}
	payloads := e.ensureTilePayloads(nTiles)
	miCols := uint16(src.Width / 4)
	for t := 0; t < nTiles; t++ {
		pc := &e.pc
		if t > 0 {
			pc = &e.tilePCs[t]
		}
		c0, c1 := tilePayloadColBounds(header.Tile, t, miCols)
		data, err := pc.encodeMonochromeTileWithOptions(src, ref, golden, out, qIndex, nil, referenceMode, header.Prefix.ForceIntegerMV, header.Prefix.AllowScreenContentTools, c0, c1)
		if err != nil {
			return nil, fmt.Errorf("encode tile %d: %w", t, err)
		}
		payloads[t].Data = data
	}
	return payloads, nil
}

func (e *MonochromeVideoEncoder) configureTilePayloads(seq SequenceHeader, width, height int, tiles *TileInfo) (int, error) {
	if e.tileColsLog2 > 0 || tiles.Cols > 1 {
		info, err := interTileInfoForSequence(seq, width, height, e.tileColsLog2)
		if err != nil {
			return 0, fmt.Errorf("tile info: %w", err)
		}
		*tiles = info
		return int(info.Cols), nil
	}
	return 1, nil
}

func (e *MonochromeVideoEncoder) ensureTilePayloads(nTiles int) []TilePayload {
	if nTiles < 1 {
		nTiles = 1
	}
	if len(e.tilePCs) < nTiles {
		e.tilePCs = make([]pframeCoder, nTiles)
	}
	e.pc.effortLevel = e.effortLevel
	for i := 0; i < nTiles; i++ {
		e.tilePCs[i].effortLevel = e.effortLevel
	}
	if cap(e.payloads) < nTiles {
		e.payloads = make([]TilePayload, nTiles)
	}
	e.payloads = e.payloads[:nTiles]
	for i := range e.payloads {
		e.payloads[i] = TilePayload{}
	}
	return e.payloads
}

func (e *MonochromeVideoEncoder) assembleKeyTU(seq SequenceHeader, header IntraFrameHeaderParams, payloads []TilePayload) ([]byte, error) {
	headerSize, err := LowOverheadCompleteIntraHeaderTemporalUnitSize(seq, header)
	if err != nil {
		return nil, fmt.Errorf("size header TU: %w", err)
	}
	endTile := uint16(len(payloads) - 1)
	groupSize, err := TileGroupPayloadSize(header.Tile, 0, endTile, payloads)
	if err != nil {
		return nil, fmt.Errorf("size tile group: %w", err)
	}
	if cap(e.tuGroup) < groupSize {
		e.tuGroup = make([]byte, 0, groupSize+groupSize/2)
	}
	group := e.tuGroup[:0]
	group, err = AppendTileGroupPayload(group, header.Tile, 0, endTile, payloads)
	if err != nil {
		return nil, fmt.Errorf("append tile group: %w", err)
	}
	e.tuGroup = group
	groupOBU := OBU{Type: obu.TypeTileGroup, Payload: group}
	groupOBUSize, err := LowOverheadOBUSize(groupOBU)
	if err != nil {
		return nil, err
	}
	total := headerSize + groupOBUSize
	if cap(e.tuScratch) < total {
		e.tuScratch = make([]byte, 0, total+total/2)
	}
	out := e.tuScratch[:0]
	out, err = AppendLowOverheadCompleteIntraHeaderTemporalUnit(out, seq, header)
	if err != nil {
		return nil, fmt.Errorf("append header TU: %w", err)
	}
	out, err = AppendLowOverheadOBU(out, groupOBU)
	if err != nil {
		return nil, fmt.Errorf("append tile group OBU: %w", err)
	}
	e.tuScratch = out
	return out, nil
}

func (e *MonochromeVideoEncoder) resetRCTemporalState() {
	resetRateControlTemporalState(e.qIndex, e.rcTargetBits, e.rcFramesPerSec, e.temporalLayers, e.rcPerFrameBits, &e.rcTemporalQ, &e.rcTemporalBuffer, &e.rcTemporalRecentBits, &e.rcTemporalPerFrameBits)
}

func (e *MonochromeVideoEncoder) rcUpdate(frameBits int, temporalID uint8) {
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

func (e *MonochromeVideoEncoder) rcSurplusFrameLimit() int {
	if e.temporalLayers == 2 {
		return 2
	}
	return 8
}

func (e *MonochromeVideoEncoder) keyframeQIndex() uint8 {
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

func (e *MonochromeVideoEncoder) layerQIndex(temporalID uint8) uint8 {
	if e.rcEnabled && e.temporalLayers > 1 {
		return rateControlTemporalLayerQIndex(e.rcTemporalQ, e.rcMinQ, e.rcMaxQ, e.temporalLayers, temporalID)
	}
	return e.qIndex
}

func allocMonoFrame(dst *SourceFrameMono, src SourceFrameMono) {
	need := (src.Height-1)*src.YStride + src.Width
	if len(dst.Y) != need {
		dst.Y = make([]byte, need)
	}
	dst.YStride = src.YStride
	dst.Width = src.Width
	dst.Height = src.Height
}

func copyMonoFrameInto(dst *SourceFrameMono, src SourceFrameMono) {
	allocMonoFrame(dst, src)
	for y := 0; y < src.Height; y++ {
		copy(dst.Y[y*dst.YStride:y*dst.YStride+src.Width], src.Y[y*src.YStride:y*src.YStride+src.Width])
	}
}
