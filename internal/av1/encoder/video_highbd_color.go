package encoder

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// HighBitDepth420VideoEncoder encodes a same-sized native 10/12-bit AV1 color
// stream. The public constructor keeps the historical 4:2:0 name; WebRTC uses
// the color-aware constructor for native 4:2:0, 4:2:2, and 4:4:4 streams.
type HighBitDepth420VideoEncoder struct {
	width, height             int
	renderWidth, renderHeight int
	profile                   Profile
	color                     SequenceColorConfig
	padded                    SourceFrame42016

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
	keyRecon  SourceFrame42016
	recon     SourceFrame42016
	reconBufs [2]SourceFrame42016
	reconIdx  int
	t1Recon   SourceFrame42016
	t2Recon   SourceFrame42016
	lastRecon SourceFrame42016

	haveKey        bool
	temporalLayers int
	frameIndex     int
	lastTemporalID uint8

	tileColsLog2 uint8
	tilePCs      []pframeCoder
	payloads     []TilePayload
	tuGroup      []byte
	tuScratch    []byte
}

// NewHighBitDepth420VideoEncoder creates a native 10/12-bit 4:2:0 streaming
// encoder. Render dimensions must be even; the bitstream is coded at padded
// multiples of eight and signals render_size when needed. qIndex must be
// non-zero.
func NewHighBitDepth420VideoEncoder(width, height int, bitDepth uint8, qIndex uint8) (*HighBitDepth420VideoEncoder, error) {
	profile := Profile0
	if bitDepth == 12 {
		profile = Profile2
	}
	color := SequenceColorConfig{BitDepth: bitDepth, SubsamplingX: true, SubsamplingY: true}
	return newHighBitDepth420VideoEncoderWithColor(width, height, profile, color, qIndex)
}

func newHighBitDepth420VideoEncoderWithColor(width, height int, profile Profile, color SequenceColorConfig, qIndex uint8) (*HighBitDepth420VideoEncoder, error) {
	if width < 16 || height < 16 || width%2 != 0 || height%2 != 0 {
		return nil, fmt.Errorf("encoder: dimensions must be even and at least 16x16, got %dx%d", width, height)
	}
	if color.BitDepth != 10 && color.BitDepth != 12 {
		return nil, fmt.Errorf("encoder: high-bit-depth color bit depth must be 10 or 12, got %d", color.BitDepth)
	}
	if !highBitDepthVideoColorSupported(profile, color) {
		return nil, ErrUnsupported
	}
	if qIndex == 0 {
		return nil, fmt.Errorf("encoder: qindex must be non-zero")
	}
	if err := validateSequenceColorConfig(profile, color); err != nil {
		return nil, err
	}
	codedW := (width + 7) &^ 7
	codedH := (height + 7) &^ 7
	return &HighBitDepth420VideoEncoder{
		width: codedW, height: codedH,
		renderWidth: width, renderHeight: height,
		profile: profile,
		color:   color,
		qIndex:  qIndex,
	}, nil
}

// NewHighBitDepth420VideoEncoderCBR creates a native 10/12-bit 4:2:0 streaming
// encoder under the same CBR controller used by the realtime encoder family.
func NewHighBitDepth420VideoEncoderCBR(width, height int, bitDepth uint8, rc RateControlConfig) (*HighBitDepth420VideoEncoder, error) {
	profile := Profile0
	if bitDepth == 12 {
		profile = Profile2
	}
	color := SequenceColorConfig{BitDepth: bitDepth, SubsamplingX: true, SubsamplingY: true}
	return newHighBitDepth420VideoEncoderCBRWithColor(width, height, profile, color, rc)
}

func newHighBitDepth420VideoEncoderCBRWithColor(width, height int, profile Profile, color SequenceColorConfig, rc RateControlConfig) (*HighBitDepth420VideoEncoder, error) {
	perFrameBits, err := rateControlPerFrameBits(rc)
	if err != nil {
		return nil, err
	}
	e, err := newHighBitDepth420VideoEncoderWithColor(width, height, profile, color, rc.MinQIndex/2+rc.MaxQIndex/2)
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

func (e *HighBitDepth420VideoEncoder) SetTemporalLayers(n int) error {
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
func (e *HighBitDepth420VideoEncoder) SetTileColumns(cols int) {
	if e != nil {
		e.tileColsLog2 = tileColumnsLog2(cols)
	}
}

func (e *HighBitDepth420VideoEncoder) SetMaxThreads(n int) {
	if e == nil {
		return
	}
	if n > 0 {
		e.SetTileColumns(n)
		return
	}
	e.setDefaultTileColumns()
}

func (e *HighBitDepth420VideoEncoder) setDefaultTileColumns() {
	if e != nil {
		e.tileColsLog2 = 0
	}
}

func (e *HighBitDepth420VideoEncoder) TemporalID() uint8 {
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
func (e *HighBitDepth420VideoEncoder) SetContentHint(content ContentHint) error {
	if e == nil {
		return fmt.Errorf("encoder: nil high-bit-depth 4:2:0 video encoder")
	}
	if !content.Valid() {
		return ErrInvalidConfig
	}
	e.content = content
	return nil
}

// SetScreenContentSelection controls whether sequence headers allow per-frame
// screen-content signaling.
func (e *HighBitDepth420VideoEncoder) SetScreenContentSelection(enabled bool) {
	if e != nil {
		e.screenContentSelectable = enabled
	}
}

// SetEffortLevel selects the realtime encoder effort level.
func (e *HighBitDepth420VideoEncoder) SetEffortLevel(level int8) error {
	if e == nil {
		return fmt.Errorf("encoder: nil high-bit-depth 4:2:0 video encoder")
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

func (e *HighBitDepth420VideoEncoder) SetQIndex(qIndex uint8) error {
	if e == nil {
		return fmt.Errorf("encoder: nil high-bit-depth 4:2:0 video encoder")
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

func (e *HighBitDepth420VideoEncoder) SetRateControlConfig(rc RateControlConfig) error {
	if e == nil {
		return fmt.Errorf("encoder: nil high-bit-depth 4:2:0 video encoder")
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

func (e *HighBitDepth420VideoEncoder) QIndex() uint8 {
	if e == nil {
		return 0
	}
	return e.qIndex
}

func (e *HighBitDepth420VideoEncoder) Close() error {
	return nil
}

func (e *HighBitDepth420VideoEncoder) Recon() SourceFrame42016 {
	if e == nil {
		return SourceFrame42016{}
	}
	if e.lastRecon.Y != nil {
		return e.lastRecon
	}
	return e.recon
}

func (e *HighBitDepth420VideoEncoder) Encode(src SourceFrame42016, forceKey bool) ([]byte, bool, error) {
	return e.EncodeWithTemporalID(src, forceKey, e.TemporalID())
}

func (e *HighBitDepth420VideoEncoder) EncodeWithTemporalID(src SourceFrame42016, forceKey bool, temporalID uint8) ([]byte, bool, error) {
	if e == nil {
		return nil, false, fmt.Errorf("encoder: nil high-bit-depth 4:2:0 video encoder")
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

func (e *HighBitDepth420VideoEncoder) Prewarm() error {
	if e == nil {
		return nil
	}
	color := e.parserColorConfig()
	cw, ch := chromaWidthForColor(e.width, color), chromaHeightForColor(e.height, color)
	src := SourceFrame42016{
		Y:            make([]uint16, e.width*e.height),
		U:            make([]uint16, cw*ch),
		V:            make([]uint16, cw*ch),
		YStride:      e.width,
		ChromaStride: cw,
		Width:        e.renderWidth,
		Height:       e.renderHeight,
		BitDepth:     e.color.BitDepth,
	}
	savedQ := e.qIndex
	if bound := e.width * e.height; cap(e.tuScratch) < bound {
		e.tuScratch = make([]byte, 0, bound)
		e.tuGroup = make([]byte, 0, bound)
	}
	if _, _, err := e.Encode(src, true); err != nil {
		return err
	}
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
	e.haveKey = false
	e.frameIndex = 0
	e.lastTemporalID = 0
	e.qIndex = savedQ
	e.rcBuffer = 0
	e.rcRecentBits = [2]int{}
	e.lastRecon = SourceFrame42016{}
	return nil
}

func (e *HighBitDepth420VideoEncoder) validateRenderSource(src SourceFrame42016) error {
	if src.Width != e.renderWidth || src.Height != e.renderHeight {
		return fmt.Errorf("encoder: frame %dx%d does not match stream %dx%d", src.Width, src.Height, e.renderWidth, e.renderHeight)
	}
	if src.BitDepth != e.color.BitDepth {
		return fmt.Errorf("encoder: source bit depth %d does not match stream bit depth %d", src.BitDepth, e.color.BitDepth)
	}
	if src.Width <= 0 || src.Height <= 0 || src.Width%2 != 0 || src.Height%2 != 0 {
		return fmt.Errorf("encoder: high-bit-depth color dimensions must be positive even values, got %dx%d", src.Width, src.Height)
	}
	color := e.parserColorConfig()
	return validateSourceRenderFrameColor16(src, color, videoColorLabel(color))
}

func (e *HighBitDepth420VideoEncoder) padSource(src SourceFrame42016) SourceFrame42016 {
	color := e.parserColorConfig()
	cw, ch := chromaWidthForColor(e.width, color), chromaHeightForColor(e.height, color)
	srcCW, srcCH := chromaWidthForColor(src.Width, color), chromaHeightForColor(src.Height, color)
	if e.padded.Y == nil {
		e.padded = SourceFrame42016{
			Y:            make([]uint16, e.width*e.height),
			U:            make([]uint16, cw*ch),
			V:            make([]uint16, cw*ch),
			YStride:      e.width,
			ChromaStride: cw,
			Width:        e.width,
			Height:       e.height,
			BitDepth:     e.color.BitDepth,
		}
	}
	padPlane := func(dst []uint16, dstStride int, src []uint16, srcStride, sw, sh, dw, dh int) {
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

func (e *HighBitDepth420VideoEncoder) sequenceHeader(width, height int) SequenceHeader {
	seq := losslessKeyframeSequence(width, height)
	seq.Profile = e.profile
	seq.ColorConfig = e.color
	seq.EnableCDEF = false
	if e != nil && e.screenContentSelectable {
		seq.SeqForceScreenContentTools = SequenceSelectScreenContentTools
		seq.SeqForceIntegerMV = SequenceSelectIntegerMV
	}
	return seq
}

func (e *HighBitDepth420VideoEncoder) parserColorConfig() parser.ColorConfig {
	if e == nil {
		return parser.ColorConfig{BitDepth: 10, SubsamplingX: true, SubsamplingY: true}
	}
	return parserColorConfig(e.color)
}

func (e *HighBitDepth420VideoEncoder) applyContentHintToFrameHeader(prefix *FrameHeaderPrefix) {
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

func (e *HighBitDepth420VideoEncoder) encodeKeyWithSequenceMax(src SourceFrame42016, maxWidth, maxHeight int) ([]byte, error) {
	keyQ := e.keyframeQIndex()
	seq := e.sequenceHeader(maxWidth, maxHeight)
	header := lossyKeyframeHeaderForSequence(seq, src.Width, src.Height, keyQ)
	e.applyContentHintToFrameHeader(&header.Prefix)
	if e.renderWidth != e.width || e.renderHeight != e.height {
		header.Size.RenderWidth = uint32(e.renderWidth)
		header.Size.RenderHeight = uint32(e.renderHeight)
		header.Size.HaveRenderSize = true
	}
	allocColor16Frame(&e.keyRecon, src, e.parserColorConfig())
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
	e.rcUpdate(len(out)*8, 0)
	return out, nil
}

func (e *HighBitDepth420VideoEncoder) encodePReusing(src SourceFrame42016, temporalID uint8) ([]byte, error) {
	droppable := temporalID > 0
	isT1 := e.temporalLayers == 3 && temporalID == 1
	afterT1 := e.temporalLayers == 3 && temporalID == 2 && e.lastTemporalID == 1
	var out *SourceFrame42016
	switch {
	case !droppable:
		out = &e.reconBufs[e.reconIdx]
	case e.temporalLayers == 3 && temporalID == 2:
		out = &e.t2Recon
	default:
		out = &e.t1Recon
	}
	allocColor16Frame(out, src, e.parserColorConfig())

	refresh := uint8(0x01)
	if isT1 {
		refresh = 0x04
	} else if droppable {
		refresh = 0
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
	payloads, err := e.encodePTilePayloads(seq, src, ref, out, effQ, &header)
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
	return tu, nil
}

func (e *HighBitDepth420VideoEncoder) encodeReferencePFrameWithSequenceMax(src SourceFrame42016, ref SourceFrame42016, codedRefBuffer uint8, settings FrameEncodeSettings, maxWidth, maxHeight int) ([]byte, error) {
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
	allocColor16Frame(out, src, e.parserColorConfig())

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

	payloads, err := e.encodePTilePayloads(seq, src, ref, out, effQ, &header)
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

func (e *HighBitDepth420VideoEncoder) encodeKeyTilePayloads(seq SequenceHeader, src SourceFrame42016, recon *SourceFrame42016, qIndex uint8, header *IntraFrameHeaderParams) ([]TilePayload, error) {
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
		data, err := pc.encodeHighBitDepthColorKeyframeTileWithOptions(src, recon, qIndex, c0, c1, header.Prefix.AllowScreenContentTools, e.parserColorConfig())
		if err != nil {
			return nil, fmt.Errorf("encode tile %d: %w", t, err)
		}
		payloads[t].Data = data
	}
	return payloads, nil
}

func (e *HighBitDepth420VideoEncoder) encodePTilePayloads(seq SequenceHeader, src SourceFrame42016, ref SourceFrame42016, out *SourceFrame42016, qIndex uint8, header *InterFrameHeaderParams) ([]TilePayload, error) {
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
		data, err := pc.encodeHighBitDepthColorPFrameTile(src, ref, out, qIndex, header.Prefix.ForceIntegerMV, header.Prefix.AllowScreenContentTools, e.parserColorConfig(), c0, c1)
		if err != nil {
			return nil, fmt.Errorf("encode tile %d: %w", t, err)
		}
		payloads[t].Data = data
	}
	return payloads, nil
}

func (e *HighBitDepth420VideoEncoder) configureTilePayloads(seq SequenceHeader, width, height int, tiles *TileInfo) (int, error) {
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

func (e *HighBitDepth420VideoEncoder) ensureTilePayloads(nTiles int) []TilePayload {
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

func (e *HighBitDepth420VideoEncoder) assembleKeyTU(seq SequenceHeader, header IntraFrameHeaderParams, payloads []TilePayload) ([]byte, error) {
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

func (e *HighBitDepth420VideoEncoder) resetRCTemporalState() {
	resetRateControlTemporalState(e.qIndex, e.rcTargetBits, e.rcFramesPerSec, e.temporalLayers, e.rcPerFrameBits, &e.rcTemporalQ, &e.rcTemporalBuffer, &e.rcTemporalRecentBits, &e.rcTemporalPerFrameBits)
}

func (e *HighBitDepth420VideoEncoder) rcUpdate(frameBits int, temporalID uint8) {
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

func (e *HighBitDepth420VideoEncoder) rcSurplusFrameLimit() int {
	if e.temporalLayers == 2 {
		return 2
	}
	return 8
}

func (e *HighBitDepth420VideoEncoder) keyframeQIndex() uint8 {
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

func (e *HighBitDepth420VideoEncoder) layerQIndex(temporalID uint8) uint8 {
	if e.rcEnabled && e.temporalLayers > 1 {
		return rateControlTemporalLayerQIndex(e.rcTemporalQ, e.rcMinQ, e.rcMaxQ, e.temporalLayers, temporalID)
	}
	return e.qIndex
}

func alloc42016Frame(dst *SourceFrame42016, src SourceFrame42016) {
	allocColor16Frame(dst, src, parser.ColorConfig{BitDepth: src.BitDepth, SubsamplingX: true, SubsamplingY: true})
}

func allocColor16Frame(dst *SourceFrame42016, src SourceFrame42016, color parser.ColorConfig) {
	chromaWidth, chromaHeight := chromaWidthForColor(src.Width, color), chromaHeightForColor(src.Height, color)
	yNeed := (src.Height-1)*src.YStride + src.Width
	chromaNeed := (chromaHeight-1)*src.ChromaStride + chromaWidth
	if len(dst.Y) != yNeed {
		dst.Y = make([]uint16, yNeed)
	}
	if len(dst.U) != chromaNeed {
		dst.U = make([]uint16, chromaNeed)
	}
	if len(dst.V) != chromaNeed {
		dst.V = make([]uint16, chromaNeed)
	}
	dst.YStride = src.YStride
	dst.ChromaStride = src.ChromaStride
	dst.Width = src.Width
	dst.Height = src.Height
	dst.BitDepth = src.BitDepth
}

func copy42016FrameInto(dst *SourceFrame42016, src SourceFrame42016) {
	copyColor16FrameInto(dst, src, parser.ColorConfig{BitDepth: src.BitDepth, SubsamplingX: true, SubsamplingY: true})
}

func copyColor16FrameInto(dst *SourceFrame42016, src SourceFrame42016, color parser.ColorConfig) {
	allocColor16Frame(dst, src, color)
	for y := 0; y < src.Height; y++ {
		copy(dst.Y[y*dst.YStride:y*dst.YStride+src.Width], src.Y[y*src.YStride:y*src.YStride+src.Width])
	}
	chromaWidth, chromaHeight := chromaWidthForColor(src.Width, color), chromaHeightForColor(src.Height, color)
	for y := 0; y < chromaHeight; y++ {
		copy(dst.U[y*dst.ChromaStride:y*dst.ChromaStride+chromaWidth], src.U[y*src.ChromaStride:y*src.ChromaStride+chromaWidth])
		copy(dst.V[y*dst.ChromaStride:y*dst.ChromaStride+chromaWidth], src.V[y*src.ChromaStride:y*src.ChromaStride+chromaWidth])
	}
}

func scaleSourceFrame42016Nearest(dst *SourceFrame42016, src SourceFrame42016, width, height int) (SourceFrame42016, error) {
	return scaleSourceFrameColor16Nearest(dst, src, width, height, parser.ColorConfig{BitDepth: src.BitDepth, SubsamplingX: true, SubsamplingY: true})
}

func scaleSourceFrameColor16Nearest(dst *SourceFrame42016, src SourceFrame42016, width, height int, color parser.ColorConfig) (SourceFrame42016, error) {
	if width <= 0 || height <= 0 || width%2 != 0 || height%2 != 0 {
		return SourceFrame42016{}, ErrInvalidFrame
	}
	chromaWidth, chromaHeight := chromaWidthForColor(width, color), chromaHeightForColor(height, color)
	srcChromaWidth, srcChromaHeight := chromaWidthForColor(src.Width, color), chromaHeightForColor(src.Height, color)
	if len(dst.Y) != width*height {
		dst.Y = make([]uint16, width*height)
	}
	if len(dst.U) != chromaWidth*chromaHeight {
		dst.U = make([]uint16, chromaWidth*chromaHeight)
	}
	if len(dst.V) != chromaWidth*chromaHeight {
		dst.V = make([]uint16, chromaWidth*chromaHeight)
	}
	dst.YStride = width
	dst.ChromaStride = chromaWidth
	dst.Width = width
	dst.Height = height
	dst.BitDepth = src.BitDepth
	scalePlaneNearest16(dst.Y, width, width, height, src.Y, src.YStride, src.Width, src.Height)
	scalePlaneNearest16(dst.U, chromaWidth, chromaWidth, chromaHeight, src.U, src.ChromaStride, srcChromaWidth, srcChromaHeight)
	scalePlaneNearest16(dst.V, chromaWidth, chromaWidth, chromaHeight, src.V, src.ChromaStride, srcChromaWidth, srcChromaHeight)
	return *dst, nil
}
