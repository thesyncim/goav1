package encoder

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/obu"
)

// HighBitDepthMonochromeVideoEncoder encodes a same-sized native 10/12-bit AV1
// monochrome stream. It mirrors MonochromeVideoEncoder while reusing the
// already-proven high-bit-depth monochrome keyframe and P-frame tile coders.
type HighBitDepthMonochromeVideoEncoder struct {
	width, height             int
	renderWidth, renderHeight int
	bitDepth                  uint8
	padded                    SourceFrameMono16

	qIndex  uint8
	content ContentHint

	screenContentSelectable bool

	rcEnabled      bool
	rcPerFrameBits int
	rcBuffer       int
	rcRecentBits   [2]int
	rcMinQ, rcMaxQ uint8

	pc        pframeCoder
	keyRecon  SourceFrameMono16
	recon     SourceFrameMono16
	reconBufs [2]SourceFrameMono16
	reconIdx  int
	t1Recon   SourceFrameMono16
	t2Recon   SourceFrameMono16
	lastRecon SourceFrameMono16

	haveKey        bool
	temporalLayers int
	frameIndex     int
	lastTemporalID uint8

	payloads  [1]TilePayload
	tuGroup   []byte
	tuScratch []byte
}

// NewHighBitDepthMonochromeVideoEncoder creates a native 10/12-bit monochrome
// streaming encoder. Render dimensions may be odd or non-multiple-of-eight; the
// bitstream is coded at padded dimensions and signals render_size when needed.
func NewHighBitDepthMonochromeVideoEncoder(width, height int, bitDepth uint8, qIndex uint8) (*HighBitDepthMonochromeVideoEncoder, error) {
	if width < 16 || height < 16 {
		return nil, fmt.Errorf("encoder: dimensions must be at least 16x16, got %dx%d", width, height)
	}
	if bitDepth != 10 && bitDepth != 12 {
		return nil, fmt.Errorf("encoder: high-bit-depth monochrome bit depth must be 10 or 12, got %d", bitDepth)
	}
	if qIndex == 0 {
		return nil, fmt.Errorf("encoder: qindex must be non-zero")
	}
	codedW := (width + 7) &^ 7
	codedH := (height + 7) &^ 7
	return &HighBitDepthMonochromeVideoEncoder{
		width: codedW, height: codedH,
		renderWidth: width, renderHeight: height,
		bitDepth: bitDepth,
		qIndex:   qIndex,
	}, nil
}

// NewHighBitDepthMonochromeVideoEncoderCBR creates a native 10/12-bit
// monochrome streaming encoder under the same CBR controller used by the
// realtime encoder family.
func NewHighBitDepthMonochromeVideoEncoderCBR(width, height int, bitDepth uint8, rc RateControlConfig) (*HighBitDepthMonochromeVideoEncoder, error) {
	perFrameBits, err := rateControlPerFrameBits(rc)
	if err != nil {
		return nil, err
	}
	e, err := NewHighBitDepthMonochromeVideoEncoder(width, height, bitDepth, rc.MinQIndex/2+rc.MaxQIndex/2)
	if err != nil {
		return nil, err
	}
	e.rcEnabled = true
	e.rcPerFrameBits = perFrameBits
	e.rcMinQ = rc.MinQIndex
	e.rcMaxQ = rc.MaxQIndex
	return e, nil
}

func (e *HighBitDepthMonochromeVideoEncoder) SetTemporalLayers(n int) error {
	if n < 1 || n > 3 {
		return fmt.Errorf("encoder: unsupported temporal layer count %d", n)
	}
	e.temporalLayers = n
	return nil
}

func (e *HighBitDepthMonochromeVideoEncoder) TemporalID() uint8 {
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
func (e *HighBitDepthMonochromeVideoEncoder) SetContentHint(content ContentHint) error {
	if e == nil {
		return fmt.Errorf("encoder: nil high-bit-depth monochrome video encoder")
	}
	if !content.Valid() {
		return ErrInvalidConfig
	}
	e.content = content
	return nil
}

// SetScreenContentSelection controls whether sequence headers allow per-frame
// screen-content signaling.
func (e *HighBitDepthMonochromeVideoEncoder) SetScreenContentSelection(enabled bool) {
	if e != nil {
		e.screenContentSelectable = enabled
	}
}

func (e *HighBitDepthMonochromeVideoEncoder) SetQIndex(qIndex uint8) error {
	if e == nil {
		return fmt.Errorf("encoder: nil high-bit-depth monochrome video encoder")
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

func (e *HighBitDepthMonochromeVideoEncoder) SetRateControlConfig(rc RateControlConfig) error {
	if e == nil {
		return fmt.Errorf("encoder: nil high-bit-depth monochrome video encoder")
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

func (e *HighBitDepthMonochromeVideoEncoder) QIndex() uint8 {
	if e == nil {
		return 0
	}
	return e.qIndex
}

func (e *HighBitDepthMonochromeVideoEncoder) Close() error {
	return nil
}

func (e *HighBitDepthMonochromeVideoEncoder) Recon() SourceFrameMono16 {
	if e == nil {
		return SourceFrameMono16{}
	}
	if e.lastRecon.Y != nil {
		return e.lastRecon
	}
	return e.recon
}

func (e *HighBitDepthMonochromeVideoEncoder) Encode(src SourceFrameMono16, forceKey bool) ([]byte, bool, error) {
	return e.EncodeWithTemporalID(src, forceKey, e.TemporalID())
}

func (e *HighBitDepthMonochromeVideoEncoder) EncodeWithTemporalID(src SourceFrameMono16, forceKey bool, temporalID uint8) ([]byte, bool, error) {
	if e == nil {
		return nil, false, fmt.Errorf("encoder: nil high-bit-depth monochrome video encoder")
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

func (e *HighBitDepthMonochromeVideoEncoder) Prewarm() error {
	if e == nil {
		return nil
	}
	src := SourceFrameMono16{
		Y:        make([]uint16, e.width*e.height),
		YStride:  e.width,
		Width:    e.renderWidth,
		Height:   e.renderHeight,
		BitDepth: e.bitDepth,
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
	e.lastRecon = SourceFrameMono16{}
	return nil
}

func (e *HighBitDepthMonochromeVideoEncoder) validateRenderSource(src SourceFrameMono16) error {
	if src.Width != e.renderWidth || src.Height != e.renderHeight {
		return fmt.Errorf("encoder: frame %dx%d does not match stream %dx%d", src.Width, src.Height, e.renderWidth, e.renderHeight)
	}
	if src.BitDepth != e.bitDepth {
		return fmt.Errorf("encoder: source bit depth %d does not match stream bit depth %d", src.BitDepth, e.bitDepth)
	}
	if src.YStride < src.Width {
		return fmt.Errorf("encoder: monochrome Y stride %d is smaller than width %d", src.YStride, src.Width)
	}
	if src.Height > 0 && src.YStride > (int(^uint(0)>>1)-(src.Width-1))/(src.Height-1) {
		return fmt.Errorf("encoder: monochrome Y plane dimensions overflow int")
	}
	need := (src.Height-1)*src.YStride + src.Width
	if len(src.Y) < need {
		return fmt.Errorf("encoder: monochrome Y plane is too short: got %d samples, need %d", len(src.Y), need)
	}
	maxSample := uint16((1 << src.BitDepth) - 1)
	for y := range src.Height {
		row := src.Y[y*src.YStride : y*src.YStride+src.Width]
		for x, sample := range row {
			if sample > maxSample {
				return fmt.Errorf("encoder: monochrome Y sample (%d,%d)=%d exceeds %d-bit maximum %d", x, y, sample, src.BitDepth, maxSample)
			}
		}
	}
	return nil
}

func (e *HighBitDepthMonochromeVideoEncoder) padSource(src SourceFrameMono16) SourceFrameMono16 {
	if e.padded.Y == nil {
		e.padded = SourceFrameMono16{
			Y:        make([]uint16, e.width*e.height),
			YStride:  e.width,
			Width:    e.width,
			Height:   e.height,
			BitDepth: e.bitDepth,
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

func (e *HighBitDepthMonochromeVideoEncoder) sequenceHeader(width, height int) SequenceHeader {
	seq := lossyHighBitDepthMonochromeKeyframeSequence(width, height, e.bitDepth)
	if e != nil && e.screenContentSelectable {
		seq.SeqForceScreenContentTools = SequenceSelectScreenContentTools
		seq.SeqForceIntegerMV = SequenceSelectIntegerMV
	}
	return seq
}

func (e *HighBitDepthMonochromeVideoEncoder) applyContentHintToFrameHeader(prefix *FrameHeaderPrefix) {
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

func (e *HighBitDepthMonochromeVideoEncoder) encodeKeyWithSequenceMax(src SourceFrameMono16, maxWidth, maxHeight int) ([]byte, error) {
	keyQ := e.keyframeQIndex()
	seq := e.sequenceHeader(maxWidth, maxHeight)
	header := lossyMonochromeKeyframeHeaderForSequence(seq, src.Width, src.Height, keyQ)
	e.applyContentHintToFrameHeader(&header.Prefix)
	if e.renderWidth != e.width || e.renderHeight != e.height {
		header.Size.RenderWidth = uint32(e.renderWidth)
		header.Size.RenderHeight = uint32(e.renderHeight)
		header.Size.HaveRenderSize = true
	}
	allocMono16Frame(&e.keyRecon, src)
	tilePayload, err := e.pc.encodeHighBitDepthMonochromeKeyframeTileWithOptions(src, &e.keyRecon, keyQ, 0, uint16(src.Width/4), header.Prefix.AllowScreenContentTools)
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

func (e *HighBitDepthMonochromeVideoEncoder) encodePReusing(src SourceFrameMono16, temporalID uint8) ([]byte, error) {
	droppable := temporalID > 0
	isT1 := e.temporalLayers == 3 && temporalID == 1
	afterT1 := e.temporalLayers == 3 && temporalID == 2 && e.lastTemporalID == 1
	var out *SourceFrameMono16
	switch {
	case !droppable:
		out = &e.reconBufs[e.reconIdx]
	case e.temporalLayers == 3 && temporalID == 2:
		out = &e.t2Recon
	default:
		out = &e.t1Recon
	}
	allocMono16Frame(out, src)

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
	tilePayload, err := e.pc.encodeHighBitDepthMonochromePFrameTile(src, ref, out, effQ, header.Prefix.ForceIntegerMV, header.Prefix.AllowScreenContentTools, 0, uint16(src.Width/4))
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

func (e *HighBitDepthMonochromeVideoEncoder) encodeReferencePFrameWithSequenceMax(src SourceFrameMono16, ref SourceFrameMono16, codedRefBuffer uint8, settings FrameEncodeSettings, maxWidth, maxHeight int) ([]byte, error) {
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
	allocMono16Frame(out, src)

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

	tilePayload, err := e.pc.encodeHighBitDepthMonochromePFrameTile(src, ref, out, effQ, header.Prefix.ForceIntegerMV, header.Prefix.AllowScreenContentTools, 0, uint16(src.Width/4))
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

func (e *HighBitDepthMonochromeVideoEncoder) assembleKeyTU(seq SequenceHeader, header IntraFrameHeaderParams, tilePayload []byte) ([]byte, error) {
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

func (e *HighBitDepthMonochromeVideoEncoder) rcUpdate(frameBits int) {
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

func (e *HighBitDepthMonochromeVideoEncoder) rcSurplusFrameLimit() int {
	if e.temporalLayers == 2 {
		return 2
	}
	return 8
}

func (e *HighBitDepthMonochromeVideoEncoder) keyframeQIndex() uint8 {
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

func (e *HighBitDepthMonochromeVideoEncoder) layerQIndex(temporalID uint8) uint8 {
	q := int(e.qIndex) + int(temporalID)*layerQIndexOffset
	if q > 255 {
		q = 255
	}
	return uint8(q)
}

func allocMono16Frame(dst *SourceFrameMono16, src SourceFrameMono16) {
	need := (src.Height-1)*src.YStride + src.Width
	if len(dst.Y) != need {
		dst.Y = make([]uint16, need)
	}
	dst.YStride = src.YStride
	dst.Width = src.Width
	dst.Height = src.Height
	dst.BitDepth = src.BitDepth
}

func copyMono16FrameInto(dst *SourceFrameMono16, src SourceFrameMono16) {
	allocMono16Frame(dst, src)
	for y := 0; y < src.Height; y++ {
		copy(dst.Y[y*dst.YStride:y*dst.YStride+src.Width], src.Y[y*src.YStride:y*src.YStride+src.Width])
	}
}

func scaleSourceFrameMono16Nearest(dst *SourceFrameMono16, src SourceFrameMono16, width, height int) (SourceFrameMono16, error) {
	if width <= 0 || height <= 0 {
		return SourceFrameMono16{}, ErrInvalidFrame
	}
	if len(dst.Y) != width*height {
		dst.Y = make([]uint16, width*height)
	}
	dst.YStride = width
	dst.Width = width
	dst.Height = height
	dst.BitDepth = src.BitDepth
	scalePlaneNearest16(dst.Y, width, width, height, src.Y, src.YStride, src.Width, src.Height)
	return *dst, nil
}

func scalePlaneNearest16(dst []uint16, dstStride, dstWidth, dstHeight int, src []uint16, srcStride, srcWidth, srcHeight int) {
	for y := 0; y < dstHeight; y++ {
		sy := y * srcHeight / dstHeight
		drow := dst[y*dstStride : y*dstStride+dstWidth]
		srow := src[sy*srcStride:]
		for x := 0; x < dstWidth; x++ {
			drow[x] = srow[x*srcWidth/dstWidth]
		}
	}
}
