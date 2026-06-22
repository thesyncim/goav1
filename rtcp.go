package goav1

import (
	"encoding/binary"
	"errors"
)

const (
	// RTCPRTPFBGenericNACKFMT is the RTCP transport-layer feedback FMT value
	// for generic negative acknowledgements.
	RTCPRTPFBGenericNACKFMT = 1

	// RTCPRTPFBTransportFeedbackFMT is the RTCP transport-layer feedback FMT
	// value for transport-wide congestion-control feedback.
	RTCPRTPFBTransportFeedbackFMT = 15

	// RTCPPSFBPictureLossIndicationFMT is the RTCP payload-specific feedback
	// FMT value for Picture Loss Indication feedback.
	RTCPPSFBPictureLossIndicationFMT = 1

	// RTCPPSFBFullIntraRequestFMT is the RTCP payload-specific feedback FMT
	// value for Full Intra Request feedback.
	RTCPPSFBFullIntraRequestFMT = 4

	// RTCPPSFBLayerRefreshRequestFMT is the RTCP payload-specific feedback
	// FMT value registered for Layer Refresh Request feedback.
	RTCPPSFBLayerRefreshRequestFMT = 10

	// RTCPPSFBApplicationLayerFeedbackFMT is the RTCP payload-specific
	// feedback FMT value for application-layer feedback such as REMB.
	RTCPPSFBApplicationLayerFeedbackFMT = 15

	// RTCPGenericNACKPairSize is the size of one generic NACK FCI PID/BLP
	// pair.
	RTCPGenericNACKPairSize = 4

	// RTCPTransportFeedbackFCIHeaderSize is the size of the transport-wide
	// congestion-control feedback FCI header before packet status chunks.
	RTCPTransportFeedbackFCIHeaderSize = 8

	// RTCPTransportFeedbackChunkSize is the size of one packet status chunk.
	RTCPTransportFeedbackChunkSize = 2

	// RTCPTransportFeedbackFCIMinSize is the smallest non-empty transport-wide
	// congestion-control feedback FCI: header plus one status chunk.
	RTCPTransportFeedbackFCIMinSize = RTCPTransportFeedbackFCIHeaderSize + RTCPTransportFeedbackChunkSize

	// RTCPTransportFeedbackDeltaTickMicros is the receive-delta tick duration
	// used by transport-wide congestion-control feedback.
	RTCPTransportFeedbackDeltaTickMicros = 250

	// RTCPTransportFeedbackBaseTimeTickMicros is the reference-time tick
	// duration used by transport-wide congestion-control feedback.
	RTCPTransportFeedbackBaseTimeTickMicros = RTCPTransportFeedbackDeltaTickMicros * (1 << 8)

	// RTCPTransportFeedbackMaxPackets is the maximum packet status count that
	// fits the transport-wide congestion-control feedback FCI field.
	RTCPTransportFeedbackMaxPackets = 0xffff

	// RTCPTransportFeedbackMaxReferenceTimeTicks is the maximum 24-bit
	// reference-time value that fits in transport-wide feedback.
	RTCPTransportFeedbackMaxReferenceTimeTicks = 0xffffff

	// RTCPPictureLossIndicationFCISize is the size of a PLI FCI payload. PLI
	// carries no FCI bytes; the RTCP PSFB packet header identifies the media
	// sender being reported.
	RTCPPictureLossIndicationFCISize = 0

	// RTCPFullIntraRequestEntrySize is the size of one FIR FCI entry.
	RTCPFullIntraRequestEntrySize = 8

	// RTCPReceiverEstimatedMaximumBitrateUniqueIdentifier is the REMB FCI
	// unique identifier "REMB".
	RTCPReceiverEstimatedMaximumBitrateUniqueIdentifier = 0x52454D42

	// RTCPReceiverEstimatedMaximumBitrateFCIMinSize is the size of a REMB FCI
	// payload with no SSRC feedback entries.
	RTCPReceiverEstimatedMaximumBitrateFCIMinSize = 8

	// RTCPReceiverEstimatedMaximumBitrateSSRCSize is the size of one REMB SSRC
	// feedback entry.
	RTCPReceiverEstimatedMaximumBitrateSSRCSize = 4

	// RTCPReceiverEstimatedMaximumBitrateMaxSSRCs is the maximum SSRC count
	// that fits the REMB FCI count field.
	RTCPReceiverEstimatedMaximumBitrateMaxSSRCs = 0xff

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
	// is too small for the requested feedback field.
	ErrRTCPShortBuffer = errors.New("goav1: short RTCP buffer")
	// ErrRTCPInvalidFeedback is returned when a generic RTCP feedback helper
	// receives malformed FCI data.
	ErrRTCPInvalidFeedback = errors.New("goav1: invalid RTCP feedback")
	// ErrRTCPInvalidLayerRefreshRequest is returned when an AV1 LRR entry
	// carries out-of-range layer IDs, payload type, or upgrade semantics.
	ErrRTCPInvalidLayerRefreshRequest = errors.New("goav1: invalid RTCP layer refresh request")
)

// RTCPGenericNACKPair is one generic NACK Feedback Control Information pair.
// PacketID is the first lost RTP sequence number; LostPacketBitmask marks the
// following 16 sequence numbers.
type RTCPGenericNACKPair struct {
	PacketID          uint16
	LostPacketBitmask uint16
}

// RTCPGenericNACKPairsSize returns the FCI byte count needed to serialize
// pairs.
func RTCPGenericNACKPairsSize(pairs []RTCPGenericNACKPair) (int, error) {
	maxInt := int(^uint(0) >> 1)
	if len(pairs) > maxInt/RTCPGenericNACKPairSize {
		return 0, ErrRTCPInvalidFeedback
	}
	return len(pairs) * RTCPGenericNACKPairSize, nil
}

// PutRTCPGenericNACKPair serializes one generic NACK FCI pair into dst.
func PutRTCPGenericNACKPair(dst []byte, pair RTCPGenericNACKPair) (int, error) {
	if len(dst) < RTCPGenericNACKPairSize {
		return 0, ErrRTCPShortBuffer
	}
	binary.BigEndian.PutUint16(dst[0:2], pair.PacketID)
	binary.BigEndian.PutUint16(dst[2:4], pair.LostPacketBitmask)
	return RTCPGenericNACKPairSize, nil
}

// PutRTCPGenericNACKPairs serializes a complete generic NACK FCI pair list.
// The caller owns the surrounding RTCP RTPFB packet.
func PutRTCPGenericNACKPairs(dst []byte, pairs []RTCPGenericNACKPair) (int, error) {
	size, err := RTCPGenericNACKPairsSize(pairs)
	if err != nil {
		return 0, err
	}
	if len(dst) < size {
		return 0, ErrRTCPShortBuffer
	}
	for i := range pairs {
		off := i * RTCPGenericNACKPairSize
		if _, err := PutRTCPGenericNACKPair(dst[off:off+RTCPGenericNACKPairSize], pairs[i]); err != nil {
			return 0, err
		}
	}
	return size, nil
}

// AppendRTCPGenericNACKPairs appends a complete generic NACK FCI pair list to
// dst without growing beyond dst's existing capacity.
func AppendRTCPGenericNACKPairs(dst []byte, pairs []RTCPGenericNACKPair) ([]byte, error) {
	size, err := RTCPGenericNACKPairsSize(pairs)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < size {
		return dst, ErrRTCPShortBuffer
	}
	off := len(dst)
	out := dst[:off+size]
	if _, err := PutRTCPGenericNACKPairs(out[off:], pairs); err != nil {
		return dst, err
	}
	return out, nil
}

// ParseRTCPGenericNACKPair parses one generic NACK FCI pair.
func ParseRTCPGenericNACKPair(src []byte) (RTCPGenericNACKPair, int, error) {
	if len(src) < RTCPGenericNACKPairSize {
		return RTCPGenericNACKPair{}, 0, ErrRTCPShortBuffer
	}
	return RTCPGenericNACKPair{
		PacketID:          binary.BigEndian.Uint16(src[0:2]),
		LostPacketBitmask: binary.BigEndian.Uint16(src[2:4]),
	}, RTCPGenericNACKPairSize, nil
}

// ParseRTCPGenericNACKPairs parses a complete generic NACK FCI pair list into
// dst without growing beyond dst's existing capacity.
func ParseRTCPGenericNACKPairs(src []byte, dst []RTCPGenericNACKPair) ([]RTCPGenericNACKPair, error) {
	if len(src)%RTCPGenericNACKPairSize != 0 {
		return dst, ErrRTCPInvalidFeedback
	}
	count := len(src) / RTCPGenericNACKPairSize
	if cap(dst)-len(dst) < count {
		return dst, ErrRTCPShortBuffer
	}
	off := len(dst)
	out := dst[:off+count]
	for i := 0; i < count; i++ {
		start := i * RTCPGenericNACKPairSize
		pair, _, err := ParseRTCPGenericNACKPair(src[start : start+RTCPGenericNACKPairSize])
		if err != nil {
			return dst, err
		}
		out[off+i] = pair
	}
	return out, nil
}

// RTCPTransportFeedbackPacketStatus is one transport-wide congestion-control
// packet status symbol.
type RTCPTransportFeedbackPacketStatus uint8

const (
	// RTCPTransportFeedbackStatusNotReceived marks a packet that was not
	// received.
	RTCPTransportFeedbackStatusNotReceived RTCPTransportFeedbackPacketStatus = 0
	// RTCPTransportFeedbackStatusSmallDelta marks a received packet whose delta
	// fits in one unsigned byte.
	RTCPTransportFeedbackStatusSmallDelta RTCPTransportFeedbackPacketStatus = 1
	// RTCPTransportFeedbackStatusLargeOrNegativeDelta marks a received packet
	// whose delta is negative or needs a two-byte signed value.
	RTCPTransportFeedbackStatusLargeOrNegativeDelta RTCPTransportFeedbackPacketStatus = 2
)

const rtcpTransportFeedbackMaxRunLength = 0x1fff

// RTCPTransportFeedbackPacket is one packet status in a transport-wide
// congestion-control FCI. SequenceNumber must be contiguous from the feedback's
// BaseSequenceNumber when serializing.
type RTCPTransportFeedbackPacket struct {
	SequenceNumber uint16
	Received       bool
	DeltaTicks     int16
}

// RTCPTransportFeedback is one transport-wide congestion-control feedback FCI.
// Packets aliases the caller-owned destination slice passed to Parse.
type RTCPTransportFeedback struct {
	BaseSequenceNumber  uint16
	ReferenceTimeTicks  uint32
	FeedbackPacketCount uint8
	DeltasPresent       bool
	Packets             []RTCPTransportFeedbackPacket
}

// Validate rejects feedback that cannot be represented in a transport-wide
// congestion-control FCI.
func (f RTCPTransportFeedback) Validate() error {
	if len(f.Packets) == 0 || len(f.Packets) > RTCPTransportFeedbackMaxPackets {
		return ErrRTCPInvalidFeedback
	}
	if f.ReferenceTimeTicks > RTCPTransportFeedbackMaxReferenceTimeTicks {
		return ErrRTCPInvalidFeedback
	}
	for i := range f.Packets {
		expected := f.BaseSequenceNumber + uint16(i)
		packet := f.Packets[i]
		if packet.SequenceNumber != expected {
			return ErrRTCPInvalidFeedback
		}
		if !packet.Received {
			if packet.DeltaTicks != 0 {
				return ErrRTCPInvalidFeedback
			}
			continue
		}
		if !f.DeltasPresent && packet.DeltaTicks != 0 {
			return ErrRTCPInvalidFeedback
		}
	}
	return nil
}

// RTCPTransportFeedbackFCISize returns the byte count needed to serialize f as
// a transport-wide congestion-control FCI. The caller owns the surrounding RTCP
// RTPFB packet.
func RTCPTransportFeedbackFCISize(f RTCPTransportFeedback) (int, error) {
	chunks, deltaBytes, err := rtcpTransportFeedbackEncodedChunkStats(f, nil)
	if err != nil {
		return 0, err
	}
	return RTCPTransportFeedbackFCIHeaderSize + chunks*RTCPTransportFeedbackChunkSize + deltaBytes, nil
}

// PutRTCPTransportFeedbackFCI serializes one transport-wide
// congestion-control FCI into dst.
func PutRTCPTransportFeedbackFCI(dst []byte, f RTCPTransportFeedback) (int, error) {
	size, err := RTCPTransportFeedbackFCISize(f)
	if err != nil {
		return 0, err
	}
	if len(dst) < size {
		return 0, ErrRTCPShortBuffer
	}

	binary.BigEndian.PutUint16(dst[0:2], f.BaseSequenceNumber)
	binary.BigEndian.PutUint16(dst[2:4], uint16(len(f.Packets)))
	rtcpPutUint24(dst[4:7], f.ReferenceTimeTicks)
	dst[7] = f.FeedbackPacketCount

	off := RTCPTransportFeedbackFCIHeaderSize
	_, _, err = rtcpTransportFeedbackEncodedChunkStats(f, func(chunk uint16) error {
		binary.BigEndian.PutUint16(dst[off:off+RTCPTransportFeedbackChunkSize], chunk)
		off += RTCPTransportFeedbackChunkSize
		return nil
	})
	if err != nil {
		return 0, err
	}
	if f.DeltasPresent {
		for i := range f.Packets {
			packet := f.Packets[i]
			if !packet.Received {
				continue
			}
			if packet.DeltaTicks >= 0 && packet.DeltaTicks <= 0xff {
				dst[off] = byte(packet.DeltaTicks)
				off++
				continue
			}
			binary.BigEndian.PutUint16(dst[off:off+2], uint16(packet.DeltaTicks))
			off += 2
		}
	}
	return size, nil
}

// AppendRTCPTransportFeedbackFCI appends one transport-wide
// congestion-control FCI to dst without growing beyond dst's existing capacity.
func AppendRTCPTransportFeedbackFCI(dst []byte, f RTCPTransportFeedback) ([]byte, error) {
	size, err := RTCPTransportFeedbackFCISize(f)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < size {
		return dst, ErrRTCPShortBuffer
	}
	off := len(dst)
	out := dst[:off+size]
	if _, err := PutRTCPTransportFeedbackFCI(out[off:], f); err != nil {
		return dst, err
	}
	return out, nil
}

// ParseRTCPTransportFeedbackFCI parses one transport-wide congestion-control
// FCI into dst without growing beyond dst's existing capacity.
func ParseRTCPTransportFeedbackFCI(src []byte, dst []RTCPTransportFeedbackPacket) (RTCPTransportFeedback, error) {
	if len(src) < RTCPTransportFeedbackFCIHeaderSize {
		return RTCPTransportFeedback{}, ErrRTCPShortBuffer
	}
	statusCount := int(binary.BigEndian.Uint16(src[2:4]))
	if statusCount == 0 {
		return RTCPTransportFeedback{}, ErrRTCPInvalidFeedback
	}
	if cap(dst)-len(dst) < statusCount {
		return RTCPTransportFeedback{}, ErrRTCPShortBuffer
	}

	baseSequence := binary.BigEndian.Uint16(src[0:2])
	off := len(dst)
	out := dst[:off+statusCount]
	feedback := RTCPTransportFeedback{
		BaseSequenceNumber:  baseSequence,
		ReferenceTimeTicks:  rtcpUint24(src[4:7]),
		FeedbackPacketCount: src[7],
		Packets:             out[off:],
	}

	index := RTCPTransportFeedbackFCIHeaderSize
	statusIndex := 0
	deltaBytes := 0
	for statusIndex < statusCount {
		if index+RTCPTransportFeedbackChunkSize > len(src) {
			return RTCPTransportFeedback{}, ErrRTCPShortBuffer
		}
		chunk := binary.BigEndian.Uint16(src[index : index+RTCPTransportFeedbackChunkSize])
		index += RTCPTransportFeedbackChunkSize
		var err error
		statusIndex, deltaBytes, err = rtcpTransportFeedbackDecodeChunk(
			chunk, baseSequence, statusCount, statusIndex, deltaBytes, out[off:],
		)
		if err != nil {
			return RTCPTransportFeedback{}, err
		}
	}

	remaining := len(src) - index
	if remaining >= deltaBytes {
		feedback.DeltasPresent = true
		if err := rtcpTransportFeedbackParseDeltas(src[index:index+deltaBytes], feedback.Packets); err != nil {
			return RTCPTransportFeedback{}, err
		}
		if !rtcpTransportFeedbackIsPadding(src[index+deltaBytes:]) {
			return RTCPTransportFeedback{}, ErrRTCPInvalidFeedback
		}
		return feedback, nil
	}
	if !rtcpTransportFeedbackIsPadding(src[index:]) {
		return RTCPTransportFeedback{}, ErrRTCPInvalidFeedback
	}
	for i := range feedback.Packets {
		if feedback.Packets[i].Received {
			feedback.Packets[i].DeltaTicks = 0
		}
	}
	return feedback, nil
}

func rtcpTransportFeedbackEncodedChunkStats(
	f RTCPTransportFeedback, emit func(uint16) error,
) (int, int, error) {
	if err := f.Validate(); err != nil {
		return 0, 0, err
	}
	deltaBytes := 0
	for i := range f.Packets {
		_, deltaSize, err := rtcpTransportFeedbackPacketStatus(f.Packets[i], f.DeltasPresent)
		if err != nil {
			return 0, 0, err
		}
		deltaBytes += deltaSize
	}

	chunks := 0
	for i := 0; i < len(f.Packets); {
		status := rtcpTransportFeedbackPacketStatusMust(f.Packets[i], f.DeltasPresent)
		run := 1
		for i+run < len(f.Packets) &&
			run < rtcpTransportFeedbackMaxRunLength &&
			rtcpTransportFeedbackPacketStatusMust(f.Packets[i+run], f.DeltasPresent) == status {
			run++
		}
		if run >= 7 || run == len(f.Packets)-i {
			chunk := uint16(status)<<13 | uint16(run)
			if emit != nil {
				if err := emit(chunk); err != nil {
					return 0, 0, err
				}
			}
			chunks++
			i += run
			continue
		}

		oneBitLength := rtcpTransportFeedbackOneBitChunkLength(f, i)
		if oneBitLength > 0 {
			chunk := uint16(0x8000)
			for j := 0; j < oneBitLength; j++ {
				if rtcpTransportFeedbackPacketStatusMust(f.Packets[i+j], f.DeltasPresent) ==
					RTCPTransportFeedbackStatusSmallDelta {
					chunk |= 1 << (13 - j)
				}
			}
			if emit != nil {
				if err := emit(chunk); err != nil {
					return 0, 0, err
				}
			}
			chunks++
			i += oneBitLength
			continue
		}

		twoBitLength := len(f.Packets) - i
		if twoBitLength > 7 {
			twoBitLength = 7
		}
		chunk := uint16(0xc000)
		for j := 0; j < twoBitLength; j++ {
			status := rtcpTransportFeedbackPacketStatusMust(f.Packets[i+j], f.DeltasPresent)
			chunk |= uint16(status) << (2 * (6 - j))
		}
		if emit != nil {
			if err := emit(chunk); err != nil {
				return 0, 0, err
			}
		}
		chunks++
		i += twoBitLength
	}
	return chunks, deltaBytes, nil
}

func rtcpTransportFeedbackOneBitChunkLength(f RTCPTransportFeedback, start int) int {
	remaining := len(f.Packets) - start
	limit := remaining
	if limit > 14 {
		limit = 14
	}
	for i := 0; i < limit; i++ {
		if rtcpTransportFeedbackPacketStatusMust(f.Packets[start+i], f.DeltasPresent) ==
			RTCPTransportFeedbackStatusLargeOrNegativeDelta {
			return 0
		}
	}
	return limit
}

func rtcpTransportFeedbackPacketStatus(
	packet RTCPTransportFeedbackPacket, deltasPresent bool,
) (RTCPTransportFeedbackPacketStatus, int, error) {
	if !packet.Received {
		if packet.DeltaTicks != 0 {
			return 0, 0, ErrRTCPInvalidFeedback
		}
		return RTCPTransportFeedbackStatusNotReceived, 0, nil
	}
	if !deltasPresent {
		if packet.DeltaTicks != 0 {
			return 0, 0, ErrRTCPInvalidFeedback
		}
		return RTCPTransportFeedbackStatusSmallDelta, 0, nil
	}
	if packet.DeltaTicks >= 0 && packet.DeltaTicks <= 0xff {
		return RTCPTransportFeedbackStatusSmallDelta, 1, nil
	}
	return RTCPTransportFeedbackStatusLargeOrNegativeDelta, 2, nil
}

func rtcpTransportFeedbackPacketStatusMust(
	packet RTCPTransportFeedbackPacket, deltasPresent bool,
) RTCPTransportFeedbackPacketStatus {
	status, _, _ := rtcpTransportFeedbackPacketStatus(packet, deltasPresent)
	return status
}

func rtcpTransportFeedbackDecodeChunk(
	chunk uint16,
	baseSequence uint16,
	statusCount int,
	statusIndex int,
	deltaBytes int,
	packets []RTCPTransportFeedbackPacket,
) (int, int, error) {
	appendStatus := func(status RTCPTransportFeedbackPacketStatus) error {
		if status == 3 {
			return ErrRTCPInvalidFeedback
		}
		packet := &packets[statusIndex]
		packet.SequenceNumber = baseSequence + uint16(statusIndex)
		packet.Received = status != RTCPTransportFeedbackStatusNotReceived
		packet.DeltaTicks = int16(status)
		switch status {
		case RTCPTransportFeedbackStatusNotReceived:
		case RTCPTransportFeedbackStatusSmallDelta:
			deltaBytes++
		case RTCPTransportFeedbackStatusLargeOrNegativeDelta:
			deltaBytes += 2
		default:
			return ErrRTCPInvalidFeedback
		}
		statusIndex++
		return nil
	}

	remaining := statusCount - statusIndex
	if chunk&0x8000 == 0 {
		run := int(chunk & 0x1fff)
		status := RTCPTransportFeedbackPacketStatus((chunk >> 13) & 0x03)
		if run == 0 || status == 3 {
			return statusIndex, deltaBytes, ErrRTCPInvalidFeedback
		}
		if run > remaining {
			run = remaining
		}
		for i := 0; i < run; i++ {
			if err := appendStatus(status); err != nil {
				return statusIndex, deltaBytes, err
			}
		}
		return statusIndex, deltaBytes, nil
	}

	if chunk&0x4000 == 0 {
		count := remaining
		if count > 14 {
			count = 14
		}
		for i := 0; i < count; i++ {
			status := RTCPTransportFeedbackPacketStatus((chunk >> (13 - i)) & 0x01)
			if err := appendStatus(status); err != nil {
				return statusIndex, deltaBytes, err
			}
		}
		return statusIndex, deltaBytes, nil
	}

	count := remaining
	if count > 7 {
		count = 7
	}
	for i := 0; i < count; i++ {
		status := RTCPTransportFeedbackPacketStatus((chunk >> (2 * (6 - i))) & 0x03)
		if err := appendStatus(status); err != nil {
			return statusIndex, deltaBytes, err
		}
	}
	return statusIndex, deltaBytes, nil
}

func rtcpTransportFeedbackParseDeltas(src []byte, packets []RTCPTransportFeedbackPacket) error {
	index := 0
	for i := range packets {
		status := RTCPTransportFeedbackPacketStatus(packets[i].DeltaTicks)
		switch status {
		case RTCPTransportFeedbackStatusNotReceived:
			packets[i].DeltaTicks = 0
		case RTCPTransportFeedbackStatusSmallDelta:
			if index >= len(src) {
				return ErrRTCPShortBuffer
			}
			packets[i].DeltaTicks = int16(src[index])
			index++
		case RTCPTransportFeedbackStatusLargeOrNegativeDelta:
			if index+2 > len(src) {
				return ErrRTCPShortBuffer
			}
			packets[i].DeltaTicks = int16(binary.BigEndian.Uint16(src[index : index+2]))
			index += 2
		default:
			return ErrRTCPInvalidFeedback
		}
	}
	if index != len(src) {
		return ErrRTCPInvalidFeedback
	}
	return nil
}

func rtcpTransportFeedbackIsPadding(src []byte) bool {
	if len(src) > 3 {
		return false
	}
	for i := range src {
		if src[i] != 0 {
			return false
		}
	}
	return true
}

func rtcpUint24(src []byte) uint32 {
	return uint32(src[0])<<16 | uint32(src[1])<<8 | uint32(src[2])
}

func rtcpPutUint24(dst []byte, value uint32) {
	dst[0] = byte(value >> 16)
	dst[1] = byte(value >> 8)
	dst[2] = byte(value)
}

// PutRTCPPictureLossIndicationFCI serializes a Picture Loss Indication FCI
// payload. PLI FCI is empty, so dst is left unchanged and zero bytes are
// reported.
func PutRTCPPictureLossIndicationFCI(dst []byte) (int, error) {
	return RTCPPictureLossIndicationFCISize, nil
}

// AppendRTCPPictureLossIndicationFCI appends a Picture Loss Indication FCI
// payload. PLI FCI is empty, so the returned slice aliases dst unchanged.
func AppendRTCPPictureLossIndicationFCI(dst []byte) ([]byte, error) {
	return dst, nil
}

// ParseRTCPPictureLossIndicationFCI validates a Picture Loss Indication FCI
// payload. PLI FCI must be empty.
func ParseRTCPPictureLossIndicationFCI(src []byte) error {
	if len(src) != RTCPPictureLossIndicationFCISize {
		return ErrRTCPInvalidFeedback
	}
	return nil
}

const (
	rtcpReceiverEstimatedMaximumBitrateMaxMantissa   uint64 = 0x3ffff
	rtcpReceiverEstimatedMaximumBitrateMaxBitrateBps uint64 = 1<<63 - 1
)

// RTCPReceiverEstimatedMaximumBitrate is one legacy WebRTC REMB feedback FCI.
// SSRCs aliases the caller-owned destination slice passed to Parse.
type RTCPReceiverEstimatedMaximumBitrate struct {
	BitrateBps uint64
	SSRCs      []uint32
}

// RTCPReceiverEstimatedMaximumBitrateFCISize returns the REMB FCI byte count
// needed to serialize ssrcs. The caller owns the surrounding RTCP PSFB packet,
// including sender SSRC and the unused media SSRC.
func RTCPReceiverEstimatedMaximumBitrateFCISize(ssrcs []uint32) (int, error) {
	if len(ssrcs) > RTCPReceiverEstimatedMaximumBitrateMaxSSRCs {
		return 0, ErrRTCPInvalidFeedback
	}
	return RTCPReceiverEstimatedMaximumBitrateFCIMinSize +
		len(ssrcs)*RTCPReceiverEstimatedMaximumBitrateSSRCSize, nil
}

// PutRTCPReceiverEstimatedMaximumBitrateFCI serializes one REMB FCI into dst.
// The bitrate is encoded with WebRTC's 6-bit exponent and 18-bit mantissa
// representation.
func PutRTCPReceiverEstimatedMaximumBitrateFCI(dst []byte, bitrateBps uint64, ssrcs []uint32) (int, error) {
	size, err := RTCPReceiverEstimatedMaximumBitrateFCISize(ssrcs)
	if err != nil {
		return 0, err
	}
	if len(dst) < size {
		return 0, ErrRTCPShortBuffer
	}
	exponent, mantissa, err := rtcpReceiverEstimatedMaximumBitrateExponentMantissa(bitrateBps)
	if err != nil {
		return 0, err
	}

	binary.BigEndian.PutUint32(dst[0:4], RTCPReceiverEstimatedMaximumBitrateUniqueIdentifier)
	dst[4] = uint8(len(ssrcs))
	dst[5] = (exponent << 2) | uint8(mantissa>>16)
	binary.BigEndian.PutUint16(dst[6:8], uint16(mantissa&0xffff))
	for i, ssrc := range ssrcs {
		off := RTCPReceiverEstimatedMaximumBitrateFCIMinSize +
			i*RTCPReceiverEstimatedMaximumBitrateSSRCSize
		binary.BigEndian.PutUint32(dst[off:off+RTCPReceiverEstimatedMaximumBitrateSSRCSize], ssrc)
	}
	return size, nil
}

// AppendRTCPReceiverEstimatedMaximumBitrateFCI appends one REMB FCI to dst
// without growing beyond dst's existing capacity.
func AppendRTCPReceiverEstimatedMaximumBitrateFCI(dst []byte, bitrateBps uint64, ssrcs []uint32) ([]byte, error) {
	size, err := RTCPReceiverEstimatedMaximumBitrateFCISize(ssrcs)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < size {
		return dst, ErrRTCPShortBuffer
	}
	off := len(dst)
	out := dst[:off+size]
	if _, err := PutRTCPReceiverEstimatedMaximumBitrateFCI(out[off:], bitrateBps, ssrcs); err != nil {
		return dst, err
	}
	return out, nil
}

// ParseRTCPReceiverEstimatedMaximumBitrateFCI parses one complete REMB FCI
// into dst without growing beyond dst's existing capacity.
func ParseRTCPReceiverEstimatedMaximumBitrateFCI(
	src []byte, dst []uint32,
) (RTCPReceiverEstimatedMaximumBitrate, error) {
	if len(src) < RTCPReceiverEstimatedMaximumBitrateFCIMinSize {
		return RTCPReceiverEstimatedMaximumBitrate{}, ErrRTCPShortBuffer
	}
	if binary.BigEndian.Uint32(src[0:4]) != RTCPReceiverEstimatedMaximumBitrateUniqueIdentifier {
		return RTCPReceiverEstimatedMaximumBitrate{}, ErrRTCPInvalidFeedback
	}
	ssrcCount := int(src[4])
	size := RTCPReceiverEstimatedMaximumBitrateFCIMinSize +
		ssrcCount*RTCPReceiverEstimatedMaximumBitrateSSRCSize
	if len(src) != size {
		return RTCPReceiverEstimatedMaximumBitrate{}, ErrRTCPInvalidFeedback
	}

	exponent := src[5] >> 2
	mantissa := (uint64(src[5]&0x03) << 16) | uint64(binary.BigEndian.Uint16(src[6:8]))
	if mantissa > (rtcpReceiverEstimatedMaximumBitrateMaxBitrateBps >> exponent) {
		return RTCPReceiverEstimatedMaximumBitrate{}, ErrRTCPInvalidFeedback
	}
	if cap(dst)-len(dst) < ssrcCount {
		return RTCPReceiverEstimatedMaximumBitrate{}, ErrRTCPShortBuffer
	}
	off := len(dst)
	out := dst[:off+ssrcCount]
	for i := 0; i < ssrcCount; i++ {
		start := RTCPReceiverEstimatedMaximumBitrateFCIMinSize +
			i*RTCPReceiverEstimatedMaximumBitrateSSRCSize
		out[off+i] = binary.BigEndian.Uint32(src[start : start+RTCPReceiverEstimatedMaximumBitrateSSRCSize])
	}
	return RTCPReceiverEstimatedMaximumBitrate{
		BitrateBps: mantissa << exponent,
		SSRCs:      out[off:],
	}, nil
}

func rtcpReceiverEstimatedMaximumBitrateExponentMantissa(bitrateBps uint64) (uint8, uint32, error) {
	if bitrateBps > rtcpReceiverEstimatedMaximumBitrateMaxBitrateBps {
		return 0, 0, ErrRTCPInvalidFeedback
	}
	mantissa := bitrateBps
	var exponent uint8
	for mantissa > rtcpReceiverEstimatedMaximumBitrateMaxMantissa {
		mantissa >>= 1
		exponent++
	}
	return exponent, uint32(mantissa), nil
}

// RTCPFullIntraRequestEntry is one Full Intra Request Feedback Control
// Information entry. SSRC is the media sender requested to send an intra
// picture; SequenceNumber is the command sequence number for that sender.
type RTCPFullIntraRequestEntry struct {
	SSRC           uint32
	SequenceNumber uint8
}

// RTCPFullIntraRequestEntriesSize returns the FCI byte count needed to
// serialize entries.
func RTCPFullIntraRequestEntriesSize(entries []RTCPFullIntraRequestEntry) (int, error) {
	maxInt := int(^uint(0) >> 1)
	if len(entries) > maxInt/RTCPFullIntraRequestEntrySize {
		return 0, ErrRTCPInvalidFeedback
	}
	return len(entries) * RTCPFullIntraRequestEntrySize, nil
}

// PutRTCPFullIntraRequestEntry serializes one FIR FCI entry into dst.
// Reserved bytes are written as zero.
func PutRTCPFullIntraRequestEntry(dst []byte, entry RTCPFullIntraRequestEntry) (int, error) {
	if len(dst) < RTCPFullIntraRequestEntrySize {
		return 0, ErrRTCPShortBuffer
	}
	binary.BigEndian.PutUint32(dst[0:4], entry.SSRC)
	dst[4] = entry.SequenceNumber
	dst[5] = 0
	dst[6] = 0
	dst[7] = 0
	return RTCPFullIntraRequestEntrySize, nil
}

// PutRTCPFullIntraRequestEntries serializes a complete FIR FCI entry list.
// The caller owns the surrounding RTCP PSFB packet.
func PutRTCPFullIntraRequestEntries(dst []byte, entries []RTCPFullIntraRequestEntry) (int, error) {
	size, err := RTCPFullIntraRequestEntriesSize(entries)
	if err != nil {
		return 0, err
	}
	if len(dst) < size {
		return 0, ErrRTCPShortBuffer
	}
	for i := range entries {
		off := i * RTCPFullIntraRequestEntrySize
		if _, err := PutRTCPFullIntraRequestEntry(dst[off:off+RTCPFullIntraRequestEntrySize], entries[i]); err != nil {
			return 0, err
		}
	}
	return size, nil
}

// AppendRTCPFullIntraRequestEntries appends a complete FIR FCI entry list to
// dst without growing beyond dst's existing capacity.
func AppendRTCPFullIntraRequestEntries(dst []byte, entries []RTCPFullIntraRequestEntry) ([]byte, error) {
	size, err := RTCPFullIntraRequestEntriesSize(entries)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < size {
		return dst, ErrRTCPShortBuffer
	}
	off := len(dst)
	out := dst[:off+size]
	if _, err := PutRTCPFullIntraRequestEntries(out[off:], entries); err != nil {
		return dst, err
	}
	return out, nil
}

// ParseRTCPFullIntraRequestEntry parses one FIR FCI entry. Reserved bytes are
// ignored on reception.
func ParseRTCPFullIntraRequestEntry(src []byte) (RTCPFullIntraRequestEntry, int, error) {
	if len(src) < RTCPFullIntraRequestEntrySize {
		return RTCPFullIntraRequestEntry{}, 0, ErrRTCPShortBuffer
	}
	return RTCPFullIntraRequestEntry{
		SSRC:           binary.BigEndian.Uint32(src[0:4]),
		SequenceNumber: src[4],
	}, RTCPFullIntraRequestEntrySize, nil
}

// ParseRTCPFullIntraRequestEntries parses a complete FIR FCI entry list into
// dst without growing beyond dst's existing capacity.
func ParseRTCPFullIntraRequestEntries(src []byte, dst []RTCPFullIntraRequestEntry) ([]RTCPFullIntraRequestEntry, error) {
	if len(src)%RTCPFullIntraRequestEntrySize != 0 {
		return dst, ErrRTCPInvalidFeedback
	}
	count := len(src) / RTCPFullIntraRequestEntrySize
	if cap(dst)-len(dst) < count {
		return dst, ErrRTCPShortBuffer
	}
	off := len(dst)
	out := dst[:off+count]
	for i := 0; i < count; i++ {
		start := i * RTCPFullIntraRequestEntrySize
		entry, _, err := ParseRTCPFullIntraRequestEntry(src[start : start+RTCPFullIntraRequestEntrySize])
		if err != nil {
			return dst, err
		}
		out[off+i] = entry
	}
	return out, nil
}

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

// AV1RTCPLayerRefreshRequestEntriesSize returns the FCI byte count needed to
// serialize entries and validates every entry before reporting success.
func AV1RTCPLayerRefreshRequestEntriesSize(entries []AV1RTCPLayerRefreshRequestEntry) (int, error) {
	maxInt := int(^uint(0) >> 1)
	if len(entries) > maxInt/AV1RTCPLayerRefreshRequestEntrySize {
		return 0, ErrRTCPInvalidLayerRefreshRequest
	}
	for i := range entries {
		if err := entries[i].Validate(); err != nil {
			return 0, err
		}
	}
	return len(entries) * AV1RTCPLayerRefreshRequestEntrySize, nil
}

// PutAV1RTCPLayerRefreshRequestEntries serializes a complete AV1 LRR FCI
// entry list into dst. The caller owns the surrounding RTCP PSFB packet.
func PutAV1RTCPLayerRefreshRequestEntries(dst []byte, entries []AV1RTCPLayerRefreshRequestEntry) (int, error) {
	size, err := AV1RTCPLayerRefreshRequestEntriesSize(entries)
	if err != nil {
		return 0, err
	}
	if len(dst) < size {
		return 0, ErrRTCPShortBuffer
	}
	for i := range entries {
		off := i * AV1RTCPLayerRefreshRequestEntrySize
		if _, err := PutAV1RTCPLayerRefreshRequestEntry(dst[off:off+AV1RTCPLayerRefreshRequestEntrySize], entries[i]); err != nil {
			return 0, err
		}
	}
	return size, nil
}

// AppendAV1RTCPLayerRefreshRequestEntries appends a complete AV1 LRR FCI entry
// list to dst without growing beyond dst's existing capacity.
func AppendAV1RTCPLayerRefreshRequestEntries(dst []byte, entries []AV1RTCPLayerRefreshRequestEntry) ([]byte, error) {
	size, err := AV1RTCPLayerRefreshRequestEntriesSize(entries)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < size {
		return dst, ErrRTCPShortBuffer
	}
	off := len(dst)
	out := dst[:off+size]
	if _, err := PutAV1RTCPLayerRefreshRequestEntries(out[off:], entries); err != nil {
		return dst, err
	}
	return out, nil
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

// ParseAV1RTCPLayerRefreshRequestEntries parses a complete AV1 LRR FCI entry
// list into dst without growing beyond dst's existing capacity.
func ParseAV1RTCPLayerRefreshRequestEntries(
	src []byte, dst []AV1RTCPLayerRefreshRequestEntry,
) ([]AV1RTCPLayerRefreshRequestEntry, error) {
	if len(src)%AV1RTCPLayerRefreshRequestEntrySize != 0 {
		return dst, ErrRTCPInvalidLayerRefreshRequest
	}
	count := len(src) / AV1RTCPLayerRefreshRequestEntrySize
	if cap(dst)-len(dst) < count {
		return dst, ErrRTCPShortBuffer
	}
	off := len(dst)
	out := dst[:off+count]
	for i := 0; i < count; i++ {
		start := i * AV1RTCPLayerRefreshRequestEntrySize
		entry, _, err := ParseAV1RTCPLayerRefreshRequestEntry(src[start : start+AV1RTCPLayerRefreshRequestEntrySize])
		if err != nil {
			return dst, err
		}
		out[off+i] = entry
	}
	return out, nil
}

// EncoderWebRTCValidateLayerRefreshRequest validates that entry's target and
// current AV1 LRR layer indices fit config's WebRTC scalability grid.
func EncoderWebRTCValidateLayerRefreshRequest(config EncoderConfig, entry AV1RTCPLayerRefreshRequestEntry) error {
	normalized, err := SetWebRTCEncoderSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return err
	}
	return encoderWebRTCValidateLayerRefreshRequestForConfig(normalized, entry)
}

// EncoderWebRTCValidateLayerRefreshRequests validates that every entry's
// target and current AV1 LRR layer indices fit config's WebRTC scalability grid.
func EncoderWebRTCValidateLayerRefreshRequests(config EncoderConfig, entries []AV1RTCPLayerRefreshRequestEntry) error {
	normalized, err := SetWebRTCEncoderSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return err
	}
	for i := range entries {
		if err := encoderWebRTCValidateLayerRefreshRequestForConfig(normalized, entries[i]); err != nil {
			return err
		}
	}
	return nil
}

func encoderWebRTCValidateLayerRefreshRequestForConfig(
	normalized EncoderConfig, entry AV1RTCPLayerRefreshRequestEntry,
) error {
	if err := entry.Validate(); err != nil {
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
