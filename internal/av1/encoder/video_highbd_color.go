package encoder

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/obu"
)

// HighBitDepth420VideoEncoder encodes a same-sized native 10/12-bit AV1 4:2:0
// stream. It mirrors the monochrome high-bit-depth streaming encoder while
// reusing the native high-bit-depth 4:2:0 keyframe and P-frame tile coders.
type HighBitDepth420VideoEncoder struct {
	width, height             int
	renderWidth, renderHeight int
	profile                   Profile
	color                     SequenceColorConfig
	padded                    SourceFrame42016

	qIndex  uint8
	content ContentHint

	screenContentSelectable bool

	rcEnabled      bool
	rcPerFrameBits int
	rcBuffer       int
	rcRecentBits   [2]int
	rcMinQ, rcMaxQ uint8

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

	payloads  [1]TilePayload
	tuGroup   []byte
	tuScratch []byte
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
		return nil, fmt.Errorf("encoder: high-bit-depth 4:2:0 bit depth must be 10 or 12, got %d", color.BitDepth)
	}
	if color.MonoChrome || !color.SubsamplingX || !color.SubsamplingY {
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
	e.rcPerFrameBits = perFrameBits
	e.rcMinQ = rc.MinQIndex
	e.rcMaxQ = rc.MaxQIndex
	return e, nil
}

func (e *HighBitDepth420VideoEncoder) SetTemporalLayers(n int) error {
	if n < 1 || n > 3 {
		return fmt.Errorf("encoder: unsupported temporal layer count %d", n)
	}
	e.temporalLayers = n
	return nil
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

func (e *HighBitDepth420VideoEncoder) SetQIndex(qIndex uint8) error {
	if e == nil {
		return fmt.Errorf("encoder: nil high-bit-depth 4:2:0 video encoder")
	}
	if qIndex == 0 {
		return fmt.Errorf("encoder: qindex must be non-zero")
	}
	e.rcEnabled = false
	e.qIndex = qIndex
	e.rcPerFrameBits = 0
	e.rcBuffer = 0
	e.rcRecentBits = [2]int{}
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
	e.rcUpdate(len(tu) * 8)
	return tu, false, nil
}

func (e *HighBitDepth420VideoEncoder) Prewarm() error {
	if e == nil {
		return nil
	}
	cw, ch := e.width/2, e.height/2
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
		return fmt.Errorf("encoder: 4:2:0 dimensions must be positive even values, got %dx%d", src.Width, src.Height)
	}
	chromaWidth, chromaHeight := src.Width/2, src.Height/2
	if src.YStride < src.Width {
		return fmt.Errorf("encoder: 4:2:0 Y stride %d is smaller than width %d", src.YStride, src.Width)
	}
	if src.ChromaStride < chromaWidth {
		return fmt.Errorf("encoder: 4:2:0 chroma stride %d is smaller than chroma width %d", src.ChromaStride, chromaWidth)
	}
	maxInt := int(^uint(0) >> 1)
	if src.Height > 0 && src.YStride > (maxInt-(src.Width-1))/(src.Height-1) {
		return fmt.Errorf("encoder: 4:2:0 Y plane dimensions overflow int")
	}
	if chromaHeight > 0 && src.ChromaStride > (maxInt-(chromaWidth-1))/(chromaHeight-1) {
		return fmt.Errorf("encoder: 4:2:0 chroma plane dimensions overflow int")
	}
	yNeed := (src.Height-1)*src.YStride + src.Width
	chromaNeed := (chromaHeight-1)*src.ChromaStride + chromaWidth
	if len(src.Y) < yNeed {
		return fmt.Errorf("encoder: 4:2:0 Y plane is too short: got %d samples, need %d", len(src.Y), yNeed)
	}
	if len(src.U) < chromaNeed {
		return fmt.Errorf("encoder: 4:2:0 U plane is too short: got %d samples, need %d", len(src.U), chromaNeed)
	}
	if len(src.V) < chromaNeed {
		return fmt.Errorf("encoder: 4:2:0 V plane is too short: got %d samples, need %d", len(src.V), chromaNeed)
	}
	maxSample := uint16((1 << src.BitDepth) - 1)
	if err := validateSourcePlane42016Samples("Y", src.Y, src.YStride, src.Width, src.Height, maxSample, src.BitDepth); err != nil {
		return err
	}
	if err := validateSourcePlane42016Samples("U", src.U, src.ChromaStride, chromaWidth, chromaHeight, maxSample, src.BitDepth); err != nil {
		return err
	}
	return validateSourcePlane42016Samples("V", src.V, src.ChromaStride, chromaWidth, chromaHeight, maxSample, src.BitDepth)
}

func (e *HighBitDepth420VideoEncoder) padSource(src SourceFrame42016) SourceFrame42016 {
	cw, ch := e.width/2, e.height/2
	srcCW, srcCH := src.Width/2, src.Height/2
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
	alloc42016Frame(&e.keyRecon, src)
	tilePayload, err := e.pc.encodeHighBitDepth420KeyframeTileWithOptions(src, &e.keyRecon, keyQ, 0, uint16(src.Width/4), header.Prefix.AllowScreenContentTools)
	if err != nil {
		return nil, fmt.Errorf("encode tile: %w", err)
	}

	out, err := e.assembleKeyTU(seq, header, tilePayload)
	if err != nil {
		return nil, err
	}
	e.recon = e.keyRecon
	e.lastRecon = e.keyRecon
	e.haveKey = true
	e.frameIndex = 1
	e.lastTemporalID = 0
	e.rcUpdate(len(out) * 8)
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
	alloc42016Frame(out, src)

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
	tilePayload, err := e.pc.encodeHighBitDepth420PFrameTile(src, ref, out, effQ, header.Prefix.ForceIntegerMV, header.Prefix.AllowScreenContentTools, 0, uint16(src.Width/4))
	if err != nil {
		return nil, fmt.Errorf("encode tile: %w", err)
	}
	e.payloads[0] = TilePayload{Data: tilePayload}
	tu, err := assembleInterTU(seq, header, e.payloads[:], temporalID, &e.tuGroup, &e.tuScratch)
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
	alloc42016Frame(out, src)

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
	if header.Prefix.FrameSizeOverride {
		tiles, err := interTileInfoForSequence(seq, src.Width, src.Height, 0)
		if err != nil {
			return nil, fmt.Errorf("tile info: %w", err)
		}
		header.Tile = tiles
	}

	tilePayload, err := e.pc.encodeHighBitDepth420PFrameTile(src, ref, out, effQ, header.Prefix.ForceIntegerMV, header.Prefix.AllowScreenContentTools, 0, uint16(src.Width/4))
	if err != nil {
		return nil, fmt.Errorf("encode tile: %w", err)
	}
	e.payloads[0] = TilePayload{Data: tilePayload}
	tu, err := assembleInterTU(seq, header, e.payloads[:], settings.TemporalID, &e.tuGroup, &e.tuScratch)
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
	e.rcUpdate(len(tu) * 8)
	return tu, nil
}

func (e *HighBitDepth420VideoEncoder) assembleKeyTU(seq SequenceHeader, header IntraFrameHeaderParams, tilePayload []byte) ([]byte, error) {
	headerSize, err := LowOverheadCompleteIntraHeaderTemporalUnitSize(seq, header)
	if err != nil {
		return nil, fmt.Errorf("size header TU: %w", err)
	}
	e.payloads[0] = TilePayload{Data: tilePayload}
	groupSize, err := TileGroupPayloadSize(header.Tile, 0, 0, e.payloads[:])
	if err != nil {
		return nil, fmt.Errorf("size tile group: %w", err)
	}
	if cap(e.tuGroup) < groupSize {
		e.tuGroup = make([]byte, 0, groupSize+groupSize/2)
	}
	group := e.tuGroup[:0]
	group, err = AppendTileGroupPayload(group, header.Tile, 0, 0, e.payloads[:])
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

func (e *HighBitDepth420VideoEncoder) rcUpdate(frameBits int) {
	if !e.rcEnabled {
		return
	}
	e.rcRecentBits[0], e.rcRecentBits[1] = e.rcRecentBits[1], frameBits
	e.rcBuffer += e.rcPerFrameBits - frameBits
	if limit := 24 * e.rcPerFrameBits; e.rcBuffer < -limit {
		e.rcBuffer = -limit
	}
	if limit := e.rcSurplusFrameLimit() * e.rcPerFrameBits; e.rcBuffer > limit {
		e.rcBuffer = limit
	}
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

func (e *HighBitDepth420VideoEncoder) rcSurplusFrameLimit() int {
	if e.temporalLayers == 2 {
		return 2
	}
	return 8
}

func (e *HighBitDepth420VideoEncoder) keyframeQIndex() uint8 {
	keyQ := e.qIndex
	if !e.rcEnabled {
		return keyQ
	}
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
		boost = boost * (3*horizon + credit) / (4 * horizon)
	}
	if int(keyQ)-boost > int(e.rcMinQ) {
		keyQ -= uint8(boost)
	} else {
		keyQ = e.rcMinQ
	}
	if e.rcBuffer >= 0 {
		maxKeyQ := int(e.rcMinQ) + (int(e.rcMaxQ)-int(e.rcMinQ))*2/3
		if int(keyQ) > maxKeyQ {
			keyQ = uint8(maxKeyQ)
		}
	}
	return keyQ
}

func (e *HighBitDepth420VideoEncoder) layerQIndex(temporalID uint8) uint8 {
	q := int(e.qIndex) + int(temporalID)*layerQIndexOffset
	if q > 255 {
		q = 255
	}
	return uint8(q)
}

func alloc42016Frame(dst *SourceFrame42016, src SourceFrame42016) {
	chromaWidth, chromaHeight := src.Width/2, src.Height/2
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
	alloc42016Frame(dst, src)
	for y := 0; y < src.Height; y++ {
		copy(dst.Y[y*dst.YStride:y*dst.YStride+src.Width], src.Y[y*src.YStride:y*src.YStride+src.Width])
	}
	chromaWidth, chromaHeight := src.Width/2, src.Height/2
	for y := 0; y < chromaHeight; y++ {
		copy(dst.U[y*dst.ChromaStride:y*dst.ChromaStride+chromaWidth], src.U[y*src.ChromaStride:y*src.ChromaStride+chromaWidth])
		copy(dst.V[y*dst.ChromaStride:y*dst.ChromaStride+chromaWidth], src.V[y*src.ChromaStride:y*src.ChromaStride+chromaWidth])
	}
}

func scaleSourceFrame42016Nearest(dst *SourceFrame42016, src SourceFrame42016, width, height int) (SourceFrame42016, error) {
	if width <= 0 || height <= 0 || width%2 != 0 || height%2 != 0 {
		return SourceFrame42016{}, ErrInvalidFrame
	}
	chromaWidth, chromaHeight := width/2, height/2
	srcChromaWidth, srcChromaHeight := src.Width/2, src.Height/2
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
