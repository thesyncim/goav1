package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/obu"
)

// OBU describes one encoder-emitted low-overhead OBU. Payload aliases caller
// memory and is copied into the output by AppendLowOverheadOBU.
type OBU struct {
	Type       obu.Type
	TemporalID uint8
	SpatialID  uint8
	Payload    []byte
}

type webRTCScalabilityMetadataDimensions struct {
	present bool
	width   [WebRTCMaxSpatialLayers]uint16
	height  [WebRTCMaxSpatialLayers]uint16
}

// LowOverheadOBUSize returns the exact number of bytes required to emit unit as
// one low-overhead OBU with obu_has_size_field set.
func LowOverheadOBUSize(unit OBU) (int, error) {
	if err := validateOBU(unit); err != nil {
		return 0, err
	}
	return lowOverheadOBUSizeUnchecked(unit), nil
}

// AppendLowOverheadOBU appends unit as one low-overhead OBU with
// obu_has_size_field set. The function never grows dst: callers must reserve
// capacity first or use LowOverheadOBUSize to size a reusable buffer.
func AppendLowOverheadOBU(dst []byte, unit OBU) ([]byte, error) {
	if err := validateOBU(unit); err != nil {
		return dst, err
	}
	size := lowOverheadOBUSizeUnchecked(unit)
	if cap(dst)-len(dst) < size {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+size]
	n, err := obu.PutHeader(out[off:], obu.Header{
		Type:         unit.Type,
		Extension:    unit.TemporalID != 0 || unit.SpatialID != 0,
		HasSizeField: true,
		TemporalID:   unit.TemporalID,
		SpatialID:    unit.SpatialID,
	})
	if err != nil {
		return dst, err
	}
	off += n
	n, err = bitstream.PutLEB128(out[off:], uint32(len(unit.Payload)))
	if err != nil {
		return dst, err
	}
	off += n
	copy(out[off:], unit.Payload)
	return out, nil
}

func webRTCScalabilityMetadataPayloadSize(mode ScalabilityMode, dims webRTCScalabilityMetadataDimensions) (size int, ok bool, err error) {
	if !mode.Valid() || mode == ScalabilityModeL1T1 {
		return 0, false, nil
	}
	if _, ok := WebRTCScalabilityModeIDC(mode); ok {
		return bitstream.LEB128Len(uint32(obu.MetadataTypeScalability)) + 1 + 1, true, nil
	}
	structureSize, ok := webRTCScalabilityStructureSize(mode, dims)
	if !ok {
		return 0, false, nil
	}
	return bitstream.LEB128Len(uint32(obu.MetadataTypeScalability)) + 1 + structureSize + 1, true, nil
}

func appendWebRTCScalabilityMetadataPayload(dst []byte, mode ScalabilityMode, dims webRTCScalabilityMetadataDimensions) ([]byte, bool, error) {
	idc, ok := WebRTCScalabilityModeIDC(mode)
	if !ok {
		if !webRTCScalabilityModeUsesExplicitSS(mode) {
			return dst, false, nil
		}
		idc = obu.ScalabilityModeSS
	}
	size, ok, err := webRTCScalabilityMetadataPayloadSize(mode, dims)
	if err != nil || !ok {
		return dst, ok, err
	}
	if cap(dst)-len(dst) < size {
		return dst, true, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+size]
	n, err := bitstream.PutLEB128(out[off:], uint32(obu.MetadataTypeScalability))
	if err != nil {
		return dst, true, err
	}
	off += n
	out[off] = idc
	off++
	if idc == obu.ScalabilityModeSS {
		written, err := appendWebRTCScalabilityStructurePayload(out[off:], mode, dims)
		if err != nil {
			return dst, true, err
		}
		off += written
	}
	out[off] = 0x80
	return out, true, nil
}

func webRTCScalabilityModeUsesExplicitSS(mode ScalabilityMode) bool {
	if !mode.Valid() || mode == ScalabilityModeL1T1 {
		return false
	}
	_, ok := WebRTCScalabilityModeIDC(mode)
	return !ok
}

func webRTCScalabilityStructureSize(mode ScalabilityMode, dims webRTCScalabilityMetadataDimensions) (int, bool) {
	spatial, temporal, _, ok := mode.Layers()
	if !ok || spatial == 0 || spatial > WebRTCMaxSpatialLayers || temporal == 0 || temporal > 3 {
		return 0, false
	}
	groupSize, ok := webRTCScalabilityTemporalGroupSize(temporal)
	if !ok {
		return 0, false
	}
	size := 1 + int(spatial) + 1 + int(groupSize)*2
	if dims.present {
		size += int(spatial) * 4
	}
	return size, true
}

func appendWebRTCScalabilityStructurePayload(dst []byte, mode ScalabilityMode, dims webRTCScalabilityMetadataDimensions) (int, error) {
	spatial, temporal, _, ok := mode.Layers()
	if !ok || spatial == 0 || spatial > WebRTCMaxSpatialLayers {
		return 0, ErrInvalidFrame
	}
	groupSize, ok := webRTCScalabilityTemporalGroupSize(temporal)
	if !ok {
		return 0, ErrInvalidFrame
	}
	size, ok := webRTCScalabilityStructureSize(mode, dims)
	if !ok {
		return 0, ErrInvalidFrame
	}
	if len(dst) < size {
		return 0, bitstream.ErrShortBuffer
	}

	off := 0
	flags := byte((spatial - 1) << 6)
	if dims.present {
		flags |= 1 << 5
	}
	flags |= 1 << 4 // spatial_layer_description_present_flag
	flags |= 1 << 3 // temporal_group_description_present_flag
	dst[off] = flags
	off++
	if dims.present {
		for i := uint8(0); i < spatial; i++ {
			w := dims.width[i]
			h := dims.height[i]
			dst[off] = byte(w >> 8)
			dst[off+1] = byte(w)
			dst[off+2] = byte(h >> 8)
			dst[off+3] = byte(h)
			off += 4
		}
	}
	for i := uint8(0); i < spatial; i++ {
		dst[off] = webRTCScalabilitySpatialLayerRefID(mode, i)
		off++
	}
	dst[off] = groupSize
	off++
	for i := uint8(0); i < groupSize; i++ {
		entry := webRTCScalabilityTemporalGroupEntry(temporal, i)
		b := byte(entry.temporalID<<5) | 1
		if entry.temporalSwitchingUp {
			b |= 1 << 4
		}
		if entry.spatialSwitchingUp {
			b |= 1 << 3
		}
		dst[off] = b
		dst[off+1] = entry.refPicDiff
		off += 2
	}
	return off, nil
}

func webRTCScalabilitySpatialLayerRefID(mode ScalabilityMode, spatialID uint8) uint8 {
	if spatialID == 0 || mode.IsSimulcast() {
		return 0xff
	}
	return spatialID - 1
}

func webRTCScalabilityTemporalGroupSize(temporal uint8) (uint8, bool) {
	switch temporal {
	case 1:
		return 1, true
	case 2:
		return 2, true
	case 3:
		return 4, true
	default:
		return 0, false
	}
}

type webRTCScalabilityTemporalGroup struct {
	temporalID          uint8
	temporalSwitchingUp bool
	spatialSwitchingUp  bool
	refPicDiff          uint8
}

func webRTCScalabilityTemporalGroupEntry(temporal uint8, index uint8) webRTCScalabilityTemporalGroup {
	switch temporal {
	case 1:
		return webRTCScalabilityTemporalGroup{temporalID: 0, temporalSwitchingUp: true, refPicDiff: 1}
	case 2:
		if index == 0 {
			return webRTCScalabilityTemporalGroup{temporalID: 0, temporalSwitchingUp: true, refPicDiff: 2}
		}
		return webRTCScalabilityTemporalGroup{temporalID: 1, temporalSwitchingUp: true, refPicDiff: 1}
	default:
		switch index {
		case 0:
			return webRTCScalabilityTemporalGroup{temporalID: 0, temporalSwitchingUp: true, refPicDiff: 4}
		case 1:
			return webRTCScalabilityTemporalGroup{temporalID: 2, temporalSwitchingUp: true, refPicDiff: 1}
		case 2:
			return webRTCScalabilityTemporalGroup{temporalID: 1, refPicDiff: 2}
		default:
			return webRTCScalabilityTemporalGroup{temporalID: 2, temporalSwitchingUp: true, refPicDiff: 1}
		}
	}
}

func webRTCScalabilityMetadataDimensionsForFrames(frames []FrameEncodeSettings) webRTCScalabilityMetadataDimensions {
	var dims webRTCScalabilityMetadataDimensions
	if len(frames) == 0 || len(frames) > WebRTCMaxSpatialLayers {
		return dims
	}
	for i := range frames {
		res := frames[i].Resolution
		if res.Width <= 0 || res.Height <= 0 || res.Width > 0xffff || res.Height > 0xffff {
			return webRTCScalabilityMetadataDimensions{}
		}
		dims.width[i] = uint16(res.Width)
		dims.height[i] = uint16(res.Height)
	}
	dims.present = true
	return dims
}

func webRTCScalabilityMetadataDimensionsForConfig(config Config) webRTCScalabilityMetadataDimensions {
	var frames [WebRTCMaxSpatialLayers]FrameEncodeSettings
	if config.SpatialLayerCount == 0 || config.SpatialLayerCount > WebRTCMaxSpatialLayers {
		return webRTCScalabilityMetadataDimensions{}
	}
	for i := uint8(0); i < config.SpatialLayerCount; i++ {
		frames[i].Resolution = config.SpatialLayers[i].Resolution
	}
	return webRTCScalabilityMetadataDimensionsForFrames(frames[:config.SpatialLayerCount])
}

func lowOverheadWebRTCScalabilityMetadataOBUSize(mode ScalabilityMode, dims webRTCScalabilityMetadataDimensions) (size int, ok bool, err error) {
	payloadSize, ok, err := webRTCScalabilityMetadataPayloadSize(mode, dims)
	if err != nil || !ok {
		return 0, ok, err
	}
	return 1 + bitstream.LEB128Len(uint32(payloadSize)) + payloadSize, true, nil
}

func appendLowOverheadWebRTCScalabilityMetadataOBU(dst []byte, mode ScalabilityMode, dims webRTCScalabilityMetadataDimensions) ([]byte, bool, error) {
	payloadSize, ok, err := webRTCScalabilityMetadataPayloadSize(mode, dims)
	if err != nil || !ok {
		return dst, ok, err
	}
	obuSize := 1 + bitstream.LEB128Len(uint32(payloadSize)) + payloadSize
	if cap(dst)-len(dst) < obuSize {
		return dst, true, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+obuSize]
	n, err := obu.PutHeader(out[off:], obu.Header{
		Type:         obu.TypeMetadata,
		HasSizeField: true,
	})
	if err != nil {
		return dst, true, err
	}
	off += n
	n, err = bitstream.PutLEB128(out[off:], uint32(payloadSize))
	if err != nil {
		return dst, true, err
	}
	off += n
	payload, _, err := appendWebRTCScalabilityMetadataPayload(out[:off], mode, dims)
	if err != nil {
		return dst, true, err
	}
	if len(payload) != len(out) {
		return dst, true, ErrInvalidFrame
	}
	return out, true, nil
}

func validateOBU(unit OBU) error {
	if unit.Type > 15 || unit.Type.Reserved() {
		return ErrInvalidFrame
	}
	if unit.TemporalID > 7 || unit.SpatialID > 3 {
		return ErrInvalidFrame
	}
	if uint64(len(unit.Payload)) > bitstream.MaxLEB128Value {
		return bitstream.ErrLEB128Overflow
	}
	return nil
}

func lowOverheadOBUSizeUnchecked(unit OBU) int {
	headerLen := 1
	if unit.TemporalID != 0 || unit.SpatialID != 0 {
		headerLen = 2
	}
	return headerLen + bitstream.LEB128Len(uint32(len(unit.Payload))) + len(unit.Payload)
}

// AppendLowOverheadMetadataOBU encodes meta as one AV1 low-overhead
// OBU_METADATA (spec section 5.8 / 6.7) and appends it to dst, growing dst if
// needed. Supports HDR_CLL, HDR_MDCV, ITUT_T35, and TIMECODE.
func AppendLowOverheadMetadataOBU(dst []byte, meta obu.Metadata) ([]byte, error) {
	var payload [128]byte
	var p []byte
	var leb [8]byte
	ln, err := bitstream.PutLEB128(leb[:], uint32(meta.Type))
	if err != nil {
		return dst, err
	}
	switch meta.Type {
	case obu.MetadataTypeHDRCLL:
		p = payload[:0]
		p = append(p, leb[:ln]...)
		p = append(p,
			byte(meta.HDRCLL.MaxCLL>>8), byte(meta.HDRCLL.MaxCLL),
			byte(meta.HDRCLL.MaxFALL>>8), byte(meta.HDRCLL.MaxFALL),
			0x80,
		)
	case obu.MetadataTypeHDRMDCV:
		p = payload[:0]
		p = append(p, leb[:ln]...)
		for i := range 3 {
			p = append(p,
				byte(meta.HDRMDCV.PrimaryChromaticityX[i]>>8), byte(meta.HDRMDCV.PrimaryChromaticityX[i]),
				byte(meta.HDRMDCV.PrimaryChromaticityY[i]>>8), byte(meta.HDRMDCV.PrimaryChromaticityY[i]),
			)
		}
		p = append(p,
			byte(meta.HDRMDCV.WhitePointChromaticityX>>8), byte(meta.HDRMDCV.WhitePointChromaticityX),
			byte(meta.HDRMDCV.WhitePointChromaticityY>>8), byte(meta.HDRMDCV.WhitePointChromaticityY),
			byte(meta.HDRMDCV.LuminanceMax>>24), byte(meta.HDRMDCV.LuminanceMax>>16),
			byte(meta.HDRMDCV.LuminanceMax>>8), byte(meta.HDRMDCV.LuminanceMax),
			byte(meta.HDRMDCV.LuminanceMin>>24), byte(meta.HDRMDCV.LuminanceMin>>16),
			byte(meta.HDRMDCV.LuminanceMin>>8), byte(meta.HDRMDCV.LuminanceMin),
			0x80,
		)
	case obu.MetadataTypeITUTT35:
		need := ln + 1 + len(meta.ITUTT35.Payload) + 1
		if meta.ITUTT35.CountryCode == 0xff {
			need++
		}
		if need <= len(payload) {
			p = payload[:0]
		} else {
			p = make([]byte, 0, need)
		}
		p = append(p, leb[:ln]...)
		p = append(p, meta.ITUTT35.CountryCode)
		if meta.ITUTT35.CountryCode == 0xff {
			p = append(p, meta.ITUTT35.CountryCodeExtension)
		}
		p = append(p, meta.ITUTT35.Payload...)
		p = append(p, 0x80)
	case obu.MetadataTypeTimecode:
		tc := meta.Timecode
		if tc.CountingType >= 1<<5 || tc.NFrames >= 1<<9 || tc.TimeOffsetLength >= 1<<5 {
			return dst, ErrInvalidFrame
		}
		copy(payload[:ln], leb[:ln])
		bw := newBitWriter(payload[ln:])
		if err := bw.writeBits(uint64(tc.CountingType), 5); err != nil {
			return dst, err
		}
		if err := bw.writeBool(tc.FullTimestamp); err != nil {
			return dst, err
		}
		if err := bw.writeBool(tc.Discontinuity); err != nil {
			return dst, err
		}
		if err := bw.writeBool(tc.CountDropped); err != nil {
			return dst, err
		}
		if err := bw.writeBits(uint64(tc.NFrames), 9); err != nil {
			return dst, err
		}
		if tc.FullTimestamp {
			if err := bw.writeBits(uint64(tc.Seconds&0x3f), 6); err != nil {
				return dst, err
			}
			if err := bw.writeBits(uint64(tc.Minutes&0x3f), 6); err != nil {
				return dst, err
			}
			if err := bw.writeBits(uint64(tc.Hours&0x1f), 5); err != nil {
				return dst, err
			}
		} else {
			hasSec := tc.HasSeconds || tc.Seconds != 0 || tc.HasMinutes || tc.Minutes != 0 || tc.HasHours || tc.Hours != 0
			if err := bw.writeBool(hasSec); err != nil {
				return dst, err
			}
			if hasSec {
				if err := bw.writeBits(uint64(tc.Seconds&0x3f), 6); err != nil {
					return dst, err
				}
				hasMin := tc.HasMinutes || tc.Minutes != 0 || tc.HasHours || tc.Hours != 0
				if err := bw.writeBool(hasMin); err != nil {
					return dst, err
				}
				if hasMin {
					if err := bw.writeBits(uint64(tc.Minutes&0x3f), 6); err != nil {
						return dst, err
					}
					hasHour := tc.HasHours || tc.Hours != 0
					if err := bw.writeBool(hasHour); err != nil {
						return dst, err
					}
					if hasHour {
						if err := bw.writeBits(uint64(tc.Hours&0x1f), 5); err != nil {
							return dst, err
						}
					}
				}
			}
		}
		if err := bw.writeBits(uint64(tc.TimeOffsetLength), 5); err != nil {
			return dst, err
		}
		if tc.TimeOffsetLength > 0 {
			if err := bw.writeBits(uint64(tc.TimeOffsetValue), tc.TimeOffsetLength); err != nil {
				return dst, err
			}
		}
		if err := bw.writeTrailingBits(); err != nil {
			return dst, err
		}
		p = payload[:ln+bw.bytesWritten()]
	default:
		return dst, ErrInvalidFrame
	}

	unit := OBU{Type: obu.TypeMetadata, Payload: p}
	need := lowOverheadOBUSizeUnchecked(unit)
	if cap(dst)-len(dst) < need {
		grown := make([]byte, len(dst), max(len(dst)+need, cap(dst)*2))
		copy(grown, dst)
		dst = grown
	}
	return AppendLowOverheadOBU(dst, unit)
}

// InjectLowOverheadMetadataOBUs inserts metas as OBU_METADATA units right after
// any leading OBU_TEMPORAL_DELIMITER and OBU_SEQUENCE_HEADER in tu, reusing
// *scratch. When len(metas) == 0, tu is returned unchanged with zero overhead.
func InjectLowOverheadMetadataOBUs(tu []byte, metas []obu.Metadata, scratch *[]byte) ([]byte, error) {
	if len(metas) == 0 {
		return tu, nil
	}
	insertAt := 0
	it := obu.NewLowOverheadIterator(tu)
	for {
		u, ok, err := it.Next()
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		if u.Header.Type == obu.TypeTemporalDelimiter || u.Header.Type == obu.TypeSequenceHeader {
			insertAt += len(u.Raw)
			continue
		}
		break
	}
	out := (*scratch)[:0]
	out = append(out, tu[:insertAt]...)
	for _, m := range metas {
		var err error
		out, err = AppendLowOverheadMetadataOBU(out, m)
		if err != nil {
			return nil, err
		}
	}
	out = append(out, tu[insertAt:]...)
	*scratch = out
	return out, nil
}
