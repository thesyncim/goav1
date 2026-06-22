package goav1

import (
	"encoding/binary"
	"errors"
)

const (
	// RTCPPSFBLayerRefreshRequestFMT is the RTCP payload-specific feedback
	// FMT value registered for Layer Refresh Request feedback.
	RTCPPSFBLayerRefreshRequestFMT = 10

	// AV1RTCPLayerRefreshLayerIndexSize is the size of one AV1 LRR layer
	// index field: temporal_id plus spatial_id with reserved bits.
	AV1RTCPLayerRefreshLayerIndexSize = 2

	// AV1RTCPLayerRefreshRequestEntrySize is the size of one LRR FCI entry.
	AV1RTCPLayerRefreshRequestEntrySize = 12

	// AV1RTCPLayerRefreshMaxTemporalID is the maximum temporal_id that fits
	// the AV1 LRR layer-index field.
	AV1RTCPLayerRefreshMaxTemporalID = 7

	// AV1RTCPLayerRefreshMaxSpatialID is the maximum spatial_id that fits the
	// AV1 LRR layer-index field.
	AV1RTCPLayerRefreshMaxSpatialID = 3
)

var (
	// ErrRTCPShortBuffer is returned when an RTCP helper's caller-owned buffer
	// is too small for the requested AV1 feedback field.
	ErrRTCPShortBuffer = errors.New("goav1: short RTCP buffer")
	// ErrRTCPInvalidLayerRefreshRequest is returned when an AV1 LRR entry
	// carries out-of-range layer IDs, payload type, or upgrade semantics.
	ErrRTCPInvalidLayerRefreshRequest = errors.New("goav1: invalid RTCP layer refresh request")
)

// AV1RTCPLayerRefreshLayerIndex is one AV1 layer index in an LRR feedback
// entry. TemporalID maps to AV1 temporal_id; SpatialID maps to AV1 spatial_id.
type AV1RTCPLayerRefreshLayerIndex struct {
	TemporalID uint8
	SpatialID  uint8
}

// Validate rejects AV1 LRR layer indices outside the AV1 RTP payload range.
func (l AV1RTCPLayerRefreshLayerIndex) Validate() error {
	if l.TemporalID > AV1RTCPLayerRefreshMaxTemporalID ||
		l.SpatialID > AV1RTCPLayerRefreshMaxSpatialID {
		return ErrRTCPInvalidLayerRefreshRequest
	}
	return nil
}

// AV1RTCPLayerRefreshRequestEntry is one Feedback Control Information entry
// inside an RTCP payload-specific feedback packet with FMT=10. RTP/RTCP
// packet headers and command-source SSRC handling remain caller-owned.
type AV1RTCPLayerRefreshRequestEntry struct {
	// SSRC is the media sender requested to send a layer refresh point.
	SSRC uint32
	// SequenceNumber is the command sequence number for the target sender.
	SequenceNumber uint8
	// PayloadType is the 7-bit RTP payload type whose layer IDs are being used.
	PayloadType uint8
	// Target is the requested layer to refresh.
	Target AV1RTCPLayerRefreshLayerIndex
	// CurrentPresent reports whether Current is present in the FCI entry.
	CurrentPresent bool
	// Current is the highest layer the receiver can already decode.
	Current AV1RTCPLayerRefreshLayerIndex
}

// Validate rejects malformed LRR entries. When CurrentPresent is true, Target
// must be a strict temporal or spatial upgrade from Current.
func (e AV1RTCPLayerRefreshRequestEntry) Validate() error {
	if e.PayloadType > 0x7f {
		return ErrRTCPInvalidLayerRefreshRequest
	}
	if err := e.Target.Validate(); err != nil {
		return err
	}
	if !e.CurrentPresent {
		if e.Current != (AV1RTCPLayerRefreshLayerIndex{}) {
			return ErrRTCPInvalidLayerRefreshRequest
		}
		return nil
	}
	if err := e.Current.Validate(); err != nil {
		return err
	}
	if e.Target.TemporalID < e.Current.TemporalID ||
		e.Target.SpatialID < e.Current.SpatialID ||
		e.Target == e.Current {
		return ErrRTCPInvalidLayerRefreshRequest
	}
	return nil
}

// PutAV1RTCPLayerRefreshLayerIndex serializes one AV1 LRR layer index into
// dst. Reserved bits are written as zero.
func PutAV1RTCPLayerRefreshLayerIndex(dst []byte, index AV1RTCPLayerRefreshLayerIndex) (int, error) {
	if len(dst) < AV1RTCPLayerRefreshLayerIndexSize {
		return 0, ErrRTCPShortBuffer
	}
	if err := index.Validate(); err != nil {
		return 0, err
	}
	dst[0] = index.TemporalID & 0x07
	dst[1] = index.SpatialID & 0x03
	return AV1RTCPLayerRefreshLayerIndexSize, nil
}

// ParseAV1RTCPLayerRefreshLayerIndex parses one AV1 LRR layer index. Reserved
// RES bits are ignored on reception; a set VP9 SID high bit is rejected because
// AV1 spatial_id is only two bits in this field.
func ParseAV1RTCPLayerRefreshLayerIndex(src []byte) (AV1RTCPLayerRefreshLayerIndex, int, error) {
	if len(src) < AV1RTCPLayerRefreshLayerIndexSize {
		return AV1RTCPLayerRefreshLayerIndex{}, 0, ErrRTCPShortBuffer
	}
	if src[1]&0x04 != 0 {
		return AV1RTCPLayerRefreshLayerIndex{}, 0, ErrRTCPInvalidLayerRefreshRequest
	}
	index := AV1RTCPLayerRefreshLayerIndex{
		TemporalID: src[0] & 0x07,
		SpatialID:  src[1] & 0x03,
	}
	if err := index.Validate(); err != nil {
		return AV1RTCPLayerRefreshLayerIndex{}, 0, err
	}
	return index, AV1RTCPLayerRefreshLayerIndexSize, nil
}

// PutAV1RTCPLayerRefreshRequestEntry serializes one AV1 LRR FCI entry into
// dst. The caller owns the surrounding RTCP PSFB packet.
func PutAV1RTCPLayerRefreshRequestEntry(dst []byte, entry AV1RTCPLayerRefreshRequestEntry) (int, error) {
	if len(dst) < AV1RTCPLayerRefreshRequestEntrySize {
		return 0, ErrRTCPShortBuffer
	}
	if err := entry.Validate(); err != nil {
		return 0, err
	}
	binary.BigEndian.PutUint32(dst[0:4], entry.SSRC)
	dst[4] = entry.SequenceNumber
	dst[5] = entry.PayloadType & 0x7f
	if entry.CurrentPresent {
		dst[5] |= 0x80
	}
	dst[6] = 0
	dst[7] = 0
	if _, err := PutAV1RTCPLayerRefreshLayerIndex(dst[8:10], entry.Target); err != nil {
		return 0, err
	}
	if entry.CurrentPresent {
		if _, err := PutAV1RTCPLayerRefreshLayerIndex(dst[10:12], entry.Current); err != nil {
			return 0, err
		}
	} else {
		dst[10] = 0
		dst[11] = 0
	}
	return AV1RTCPLayerRefreshRequestEntrySize, nil
}

// ParseAV1RTCPLayerRefreshRequestEntry parses one AV1 LRR FCI entry. Reserved
// fields are ignored on reception.
func ParseAV1RTCPLayerRefreshRequestEntry(src []byte) (AV1RTCPLayerRefreshRequestEntry, int, error) {
	if len(src) < AV1RTCPLayerRefreshRequestEntrySize {
		return AV1RTCPLayerRefreshRequestEntry{}, 0, ErrRTCPShortBuffer
	}
	target, _, err := ParseAV1RTCPLayerRefreshLayerIndex(src[8:10])
	if err != nil {
		return AV1RTCPLayerRefreshRequestEntry{}, 0, err
	}
	entry := AV1RTCPLayerRefreshRequestEntry{
		SSRC:           binary.BigEndian.Uint32(src[0:4]),
		SequenceNumber: src[4],
		CurrentPresent: src[5]&0x80 != 0,
		PayloadType:    src[5] & 0x7f,
		Target:         target,
	}
	if entry.CurrentPresent {
		current, _, err := ParseAV1RTCPLayerRefreshLayerIndex(src[10:12])
		if err != nil {
			return AV1RTCPLayerRefreshRequestEntry{}, 0, err
		}
		entry.Current = current
	}
	if err := entry.Validate(); err != nil {
		return AV1RTCPLayerRefreshRequestEntry{}, 0, err
	}
	return entry, AV1RTCPLayerRefreshRequestEntrySize, nil
}

// EncoderWebRTCValidateLayerRefreshRequest validates that entry's target and
// current AV1 LRR layer indices fit config's WebRTC scalability grid.
func EncoderWebRTCValidateLayerRefreshRequest(config EncoderConfig, entry AV1RTCPLayerRefreshRequestEntry) error {
	if err := entry.Validate(); err != nil {
		return err
	}
	normalized, err := SetWebRTCEncoderSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return err
	}
	if entry.Target.TemporalID >= normalized.TemporalLayerCount ||
		entry.Target.SpatialID >= normalized.SpatialLayerCount {
		return ErrRTCPInvalidLayerRefreshRequest
	}
	if entry.CurrentPresent &&
		(entry.Current.TemporalID >= normalized.TemporalLayerCount ||
			entry.Current.SpatialID >= normalized.SpatialLayerCount) {
		return ErrRTCPInvalidLayerRefreshRequest
	}
	return nil
}
