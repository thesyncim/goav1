package goav1

import (
	"encoding/binary"
	"errors"
)

const (
	// RTCPVersion is the RTCP version written and accepted by these helpers.
	RTCPVersion = 2

	// RTCPHeaderSize is the size of the common RTCP packet header.
	RTCPHeaderSize = 4

	// RTCPMaxPacketSize is the largest packet size addressable by the 16-bit
	// RTCP length field, in bytes.
	RTCPMaxPacketSize = (0xffff + 1) * 4

	// RTCPPacketCountMax is the largest value that fits the RTCP header count,
	// source-count, reception-report-count, or feedback-message-type field.
	RTCPPacketCountMax = 31

	// RTCPPacketPayloadMaxSize is the largest payload after the common RTCP
	// header that can fit in one packet.
	RTCPPacketPayloadMaxSize = RTCPMaxPacketSize - RTCPHeaderSize

	// RTCPSenderReportPacketType is the RTCP packet type for sender reports.
	RTCPSenderReportPacketType = 200

	// RTCPReceiverReportPacketType is the RTCP packet type for receiver
	// reports.
	RTCPReceiverReportPacketType = 201

	// RTCPSDESPacketType is the RTCP packet type for source-description
	// packets.
	RTCPSDESPacketType = 202

	// RTCPByePacketType is the RTCP packet type for BYE packets.
	RTCPByePacketType = 203

	// RTCPReportMaxBlocks is the largest report-block count that fits the RTCP
	// reception report count field.
	RTCPReportMaxBlocks = 31

	// RTCPByeMaxSources is the largest source count that fits the RTCP BYE
	// source-count field.
	RTCPByeMaxSources = 31

	// RTCPByeReasonMaxTextLen is the largest BYE reason text field.
	RTCPByeReasonMaxTextLen = 0xff

	// RTCPSDESMaxChunks is the largest source-description chunk count that
	// fits the RTCP source-count field.
	RTCPSDESMaxChunks = 31

	// RTCPSDESItemEnd terminates one source-description chunk.
	RTCPSDESItemEnd = 0

	// RTCPSDESItemCNAME is the canonical-name source-description item.
	RTCPSDESItemCNAME = 1

	// RTCPSDESItemName is the user-name source-description item.
	RTCPSDESItemName = 2

	// RTCPSDESItemEmail is the e-mail source-description item.
	RTCPSDESItemEmail = 3

	// RTCPSDESItemPhone is the phone-number source-description item.
	RTCPSDESItemPhone = 4

	// RTCPSDESItemLocation is the geographic-location source-description item.
	RTCPSDESItemLocation = 5

	// RTCPSDESItemTool is the application/tool-name source-description item.
	RTCPSDESItemTool = 6

	// RTCPSDESItemNote is the notice/status source-description item.
	RTCPSDESItemNote = 7

	// RTCPSDESItemPrivate is the private-extension source-description item.
	RTCPSDESItemPrivate = 8

	// RTCPSDESItemMaxTextLen is the largest source-description item text field.
	RTCPSDESItemMaxTextLen = 0xff

	// RTCPReportBlockSize is the size of one RTCP reception report block.
	RTCPReportBlockSize = 24

	// RTCPReportCumulativeLostMin is the minimum signed 24-bit cumulative-lost
	// value that can fit in a report block.
	RTCPReportCumulativeLostMin = -0x800000

	// RTCPReportCumulativeLostMax is the maximum signed 24-bit cumulative-lost
	// value that can fit in a report block.
	RTCPReportCumulativeLostMax = 0x7fffff

	// RTCPSenderReportSenderInfoSize is the SSRC plus sender-info block size
	// before any report blocks.
	RTCPSenderReportSenderInfoSize = 24

	// RTCPSenderReportPacketMinSize is the smallest complete sender report
	// packet: common header plus sender info and no report blocks.
	RTCPSenderReportPacketMinSize = RTCPHeaderSize + RTCPSenderReportSenderInfoSize

	// RTCPReceiverReportPacketMinSize is the smallest complete receiver report
	// packet: common header plus reporter SSRC and no report blocks.
	RTCPReceiverReportPacketMinSize = RTCPHeaderSize + 4

	// RTCPFeedbackCommonSize is the sender/media SSRC block shared by RTPFB
	// and PSFB packets.
	RTCPFeedbackCommonSize = 8

	// RTCPFeedbackPacketHeaderSize is the common RTCP header plus the RTPFB or
	// PSFB sender/media SSRC block.
	RTCPFeedbackPacketHeaderSize = RTCPHeaderSize + RTCPFeedbackCommonSize

	// RTCPFeedbackMaxFCISize is the largest feedback-control-information
	// payload that can fit in one RTPFB or PSFB packet.
	RTCPFeedbackMaxFCISize = RTCPMaxPacketSize - RTCPFeedbackPacketHeaderSize

	// RTCPRTPFBPacketType is the RTCP packet type for transport-layer feedback.
	RTCPRTPFBPacketType = 205

	// RTCPPSFBPacketType is the RTCP packet type for payload-specific feedback.
	RTCPPSFBPacketType = 206

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
	// ErrRTCPInvalidPacket is returned when a generic RTCP packet helper
	// receives malformed packet data.
	ErrRTCPInvalidPacket = errors.New("goav1: invalid RTCP packet")
	// ErrRTCPInvalidFeedback is returned when a generic RTCP feedback helper
	// receives malformed FCI data.
	ErrRTCPInvalidFeedback = errors.New("goav1: invalid RTCP feedback")
	// ErrRTCPInvalidLayerRefreshRequest is returned when an AV1 LRR entry
	// carries out-of-range layer IDs, payload type, or upgrade semantics.
	ErrRTCPInvalidLayerRefreshRequest = errors.New("goav1: invalid RTCP layer refresh request")
)

// RTCPPacket is one generic RTCP packet. Count is the low five-bit header field:
// reception report count, source count, or feedback message type depending on
// PacketType. Payload aliases the caller-owned payload passed to Put or the
// source packet passed to Parse. Padding is ignored by Put and filled by Parse.
type RTCPPacket struct {
	PacketType uint8
	Count      uint8
	Payload    []byte
	Padding    int
}

// RTCPReportBlock is one sender/receiver report block.
type RTCPReportBlock struct {
	SSRC                          uint32
	FractionLost                  uint8
	CumulativePacketsLost         int32
	ExtendedHighestSequenceNumber uint32
	InterarrivalJitter            uint32
	LastSenderReport              uint32
	DelaySinceLastSenderReport    uint32
}

// RTCPSenderReport is one complete RTCP sender report packet. Reports aliases
// the caller-owned destination slice passed to Parse.
type RTCPSenderReport struct {
	SenderSSRC        uint32
	NTPSeconds        uint32
	NTPFraction       uint32
	RTPTimestamp      uint32
	SenderPacketCount uint32
	SenderOctetCount  uint32
	Reports           []RTCPReportBlock
}

// RTCPReceiverReport is one complete RTCP receiver report packet. Reports
// aliases the caller-owned destination slice passed to Parse.
type RTCPReceiverReport struct {
	SenderSSRC uint32
	Reports    []RTCPReportBlock
}

// RTCPSDESItem is one RTCP source-description item. Text is copied by Put and
// aliases the source packet when returned by Parse.
type RTCPSDESItem struct {
	Type uint8
	Text []byte
}

// RTCPSDESChunk is one source-description chunk for one SSRC/CSRC. Items is
// copied by Put and aliases the caller-owned item destination passed to Parse.
type RTCPSDESChunk struct {
	Source uint32
	Items  []RTCPSDESItem
}

// RTCPSDESPacket is one complete RTCP source-description packet. Chunks aliases
// the caller-owned chunk destination passed to Parse.
type RTCPSDESPacket struct {
	Chunks []RTCPSDESChunk
}

// RTCPByePacket is one complete RTCP BYE packet. Sources is copied by Put and
// aliases the caller-owned destination slice passed to Parse. Reason is copied
// by Put and aliases src when returned by Parse.
type RTCPByePacket struct {
	Sources []uint32
	Reason  []byte
}

// RTCPPacketSize returns the complete packet size for a generic RTCP payload,
// including any padding needed to align the packet to a 32-bit boundary.
func RTCPPacketSize(payloadLen int) (int, error) {
	if payloadLen < 0 || payloadLen > RTCPPacketPayloadMaxSize {
		return 0, ErrRTCPInvalidPacket
	}
	size := RTCPHeaderSize + payloadLen
	size += rtcpPaddingLength(size)
	if size > RTCPMaxPacketSize {
		return 0, ErrRTCPInvalidPacket
	}
	return size, nil
}

// PutRTCPPacket serializes one generic RTCP packet into dst.
func PutRTCPPacket(dst []byte, packet RTCPPacket) (int, error) {
	if packet.Count > RTCPPacketCountMax {
		return 0, ErrRTCPInvalidPacket
	}
	size, err := RTCPPacketSize(len(packet.Payload))
	if err != nil {
		return 0, err
	}
	if len(dst) < size {
		return 0, ErrRTCPShortBuffer
	}
	padding := rtcpPaddingLength(RTCPHeaderSize + len(packet.Payload))
	first := byte(RTCPVersion<<6) | (packet.Count & 0x1f)
	if padding != 0 {
		first |= 0x20
	}
	dst[0] = first
	dst[1] = packet.PacketType
	binary.BigEndian.PutUint16(dst[2:4], uint16(size/4-1))
	copy(dst[RTCPHeaderSize:RTCPHeaderSize+len(packet.Payload)], packet.Payload)
	if padding != 0 {
		paddingStart := RTCPHeaderSize + len(packet.Payload)
		for i := 0; i < padding-1; i++ {
			dst[paddingStart+i] = 0
		}
		dst[size-1] = byte(padding)
	}
	return size, nil
}

// AppendRTCPPacket appends one generic RTCP packet to dst without growing
// beyond dst's existing capacity.
func AppendRTCPPacket(dst []byte, packet RTCPPacket) ([]byte, error) {
	if packet.Count > RTCPPacketCountMax {
		return dst, ErrRTCPInvalidPacket
	}
	size, err := RTCPPacketSize(len(packet.Payload))
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < size {
		return dst, ErrRTCPShortBuffer
	}
	off := len(dst)
	out := dst[:off+size]
	if _, err := PutRTCPPacket(out[off:], packet); err != nil {
		return dst, err
	}
	return out, nil
}

// ParseRTCPPacket parses one generic RTCP packet from src. Payload aliases src
// and excludes the common header plus any packet padding. n is the number of
// bytes consumed for the first RTCP packet.
func ParseRTCPPacket(src []byte) (packet RTCPPacket, n int, err error) {
	count, packetLen, err := parseRTCPFixedPacketHeader(src, 0, RTCPHeaderSize)
	if err != nil {
		return RTCPPacket{}, 0, err
	}
	padding := 0
	if src[0]&0x20 != 0 {
		padding = int(src[packetLen-1])
		if padding == 0 || padding > packetLen-RTCPHeaderSize {
			return RTCPPacket{}, 0, ErrRTCPInvalidPacket
		}
	}
	payloadEnd := packetLen - padding
	return RTCPPacket{
		PacketType: src[1],
		Count:      count,
		Payload:    src[RTCPHeaderSize:payloadEnd],
		Padding:    padding,
	}, packetLen, nil
}

// ParseRTCPCompoundPackets parses every RTCP packet in src. Packets are
// appended to dst without allocation and each returned Payload aliases src.
func ParseRTCPCompoundPackets(src []byte, dst []RTCPPacket) ([]RTCPPacket, error) {
	if len(src) == 0 {
		return nil, ErrRTCPShortBuffer
	}
	off := len(dst)
	for len(src) > 0 {
		if cap(dst)-len(dst) < 1 {
			return nil, ErrRTCPShortBuffer
		}
		packet, n, err := ParseRTCPPacket(src)
		if err != nil {
			return nil, err
		}
		dst = dst[:len(dst)+1]
		dst[len(dst)-1] = packet
		src = src[n:]
	}
	return dst[off:], nil
}

// RTCPSenderReportPacketSize returns the complete sender report packet size.
func RTCPSenderReportPacketSize(reportCount int) (int, error) {
	if reportCount < 0 || reportCount > RTCPReportMaxBlocks {
		return 0, ErrRTCPInvalidPacket
	}
	return RTCPSenderReportPacketMinSize + reportCount*RTCPReportBlockSize, nil
}

// RTCPReceiverReportPacketSize returns the complete receiver report packet
// size.
func RTCPReceiverReportPacketSize(reportCount int) (int, error) {
	if reportCount < 0 || reportCount > RTCPReportMaxBlocks {
		return 0, ErrRTCPInvalidPacket
	}
	return RTCPReceiverReportPacketMinSize + reportCount*RTCPReportBlockSize, nil
}

// RTCPSDESPacketSize returns the complete source-description packet size.
func RTCPSDESPacketSize(chunks []RTCPSDESChunk) (int, error) {
	size, err := rtcpSDESPacketPayloadSize(chunks)
	if err != nil {
		return 0, err
	}
	return RTCPHeaderSize + size, nil
}

// RTCPByePacketSize returns the complete BYE packet size.
func RTCPByePacketSize(sources []uint32, reason []byte) (int, error) {
	if len(sources) > RTCPByeMaxSources || len(reason) > RTCPByeReasonMaxTextLen {
		return 0, ErrRTCPInvalidPacket
	}
	size := RTCPHeaderSize + len(sources)*4
	if len(reason) > 0 {
		size += 1 + len(reason)
		size += rtcpPaddingLength(size)
	}
	return size, nil
}

// PutRTCPByePacket serializes one complete BYE packet into dst.
func PutRTCPByePacket(dst []byte, packet RTCPByePacket) (int, error) {
	size, err := RTCPByePacketSize(packet.Sources, packet.Reason)
	if err != nil {
		return 0, err
	}
	if len(dst) < size {
		return 0, ErrRTCPShortBuffer
	}
	if err := putRTCPPacketHeader(dst, RTCPByePacketType, len(packet.Sources), size); err != nil {
		return 0, err
	}
	off := RTCPHeaderSize
	for _, source := range packet.Sources {
		binary.BigEndian.PutUint32(dst[off:off+4], source)
		off += 4
	}
	if len(packet.Reason) > 0 {
		dst[off] = byte(len(packet.Reason))
		off++
		copy(dst[off:off+len(packet.Reason)], packet.Reason)
		off += len(packet.Reason)
		padding := rtcpPaddingLength(off)
		for i := 0; i < padding; i++ {
			dst[off+i] = 0
		}
		off += padding
	}
	return size, nil
}

// AppendRTCPByePacket appends one complete BYE packet to dst without growing
// beyond dst's existing capacity.
func AppendRTCPByePacket(dst []byte, packet RTCPByePacket) ([]byte, error) {
	size, err := RTCPByePacketSize(packet.Sources, packet.Reason)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < size {
		return dst, ErrRTCPShortBuffer
	}
	off := len(dst)
	out := dst[:off+size]
	if _, err := PutRTCPByePacket(out[off:], packet); err != nil {
		return dst, err
	}
	return out, nil
}

// ParseRTCPByePacket parses one complete BYE packet from src. Sources are
// appended to dst without allocation and alias dst. Reason aliases src. n is
// the number of bytes consumed for the first RTCP packet.
func ParseRTCPByePacket(src []byte, dst []uint32) (packet RTCPByePacket, n int, err error) {
	count, packetLen, err := parseRTCPFixedPacketHeader(src, RTCPByePacketType, RTCPHeaderSize)
	if err != nil {
		return RTCPByePacket{}, 0, err
	}
	minNoPadding := RTCPHeaderSize + int(count)*4
	payloadEnd, err := rtcpPacketPayloadEnd(src, packetLen, minNoPadding)
	if err != nil {
		return RTCPByePacket{}, 0, err
	}
	if payloadEnd < minNoPadding {
		return RTCPByePacket{}, 0, ErrRTCPInvalidPacket
	}
	off := len(dst)
	if cap(dst)-off < int(count) {
		return RTCPByePacket{}, 0, ErrRTCPShortBuffer
	}
	sources := dst[:off+int(count)]
	pos := RTCPHeaderSize
	for i := 0; i < int(count); i++ {
		sources[off+i] = binary.BigEndian.Uint32(src[pos : pos+4])
		pos += 4
	}
	var reason []byte
	if pos < payloadEnd {
		reasonLen := int(src[pos])
		pos++
		if payloadEnd-pos < reasonLen {
			return RTCPByePacket{}, 0, ErrRTCPInvalidPacket
		}
		reason = src[pos : pos+reasonLen]
		pos += reasonLen
		for pos < payloadEnd {
			if src[pos] != 0 {
				return RTCPByePacket{}, 0, ErrRTCPInvalidPacket
			}
			pos++
		}
	}
	if pos != payloadEnd {
		return RTCPByePacket{}, 0, ErrRTCPInvalidPacket
	}
	return RTCPByePacket{Sources: sources[off:], Reason: reason}, packetLen, nil
}

// PutRTCPSDESPacket serializes one complete source-description packet into dst.
func PutRTCPSDESPacket(dst []byte, packet RTCPSDESPacket) (int, error) {
	size, err := RTCPSDESPacketSize(packet.Chunks)
	if err != nil {
		return 0, err
	}
	if len(dst) < size {
		return 0, ErrRTCPShortBuffer
	}
	if err := putRTCPPacketHeader(dst, RTCPSDESPacketType, len(packet.Chunks), size); err != nil {
		return 0, err
	}
	off := RTCPHeaderSize
	for _, chunk := range packet.Chunks {
		binary.BigEndian.PutUint32(dst[off:off+4], chunk.Source)
		off += 4
		for _, item := range chunk.Items {
			itemSize, err := validateRTCPSDESItem(item)
			if err != nil {
				return 0, err
			}
			dst[off] = item.Type
			dst[off+1] = byte(itemSize - 2)
			copy(dst[off+2:off+itemSize], item.Text)
			off += itemSize
		}
		dst[off] = RTCPSDESItemEnd
		off++
		padding := rtcpPaddingLength(off)
		for i := 0; i < padding; i++ {
			dst[off+i] = 0
		}
		off += padding
	}
	return size, nil
}

// AppendRTCPSDESPacket appends one complete source-description packet to dst
// without growing beyond dst's existing capacity.
func AppendRTCPSDESPacket(dst []byte, packet RTCPSDESPacket) ([]byte, error) {
	size, err := RTCPSDESPacketSize(packet.Chunks)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < size {
		return dst, ErrRTCPShortBuffer
	}
	off := len(dst)
	out := dst[:off+size]
	if _, err := PutRTCPSDESPacket(out[off:], packet); err != nil {
		return dst, err
	}
	return out, nil
}

// ParseRTCPSDESPacket parses one complete source-description packet from src.
// Chunks are appended to chunkDst, items are appended to itemDst, and item text
// aliases src. n is the number of bytes consumed for the first RTCP packet.
func ParseRTCPSDESPacket(
	src []byte,
	chunkDst []RTCPSDESChunk,
	itemDst []RTCPSDESItem,
) (packet RTCPSDESPacket, n int, err error) {
	count, packetLen, err := parseRTCPFixedPacketHeader(src, RTCPSDESPacketType, RTCPHeaderSize)
	if err != nil {
		return RTCPSDESPacket{}, 0, err
	}
	payloadEnd, err := rtcpPacketPayloadEnd(src, packetLen, RTCPHeaderSize)
	if err != nil {
		return RTCPSDESPacket{}, 0, err
	}
	if payloadEnd&3 != 0 {
		return RTCPSDESPacket{}, 0, ErrRTCPInvalidPacket
	}
	chunkOff := len(chunkDst)
	if cap(chunkDst)-chunkOff < int(count) {
		return RTCPSDESPacket{}, 0, ErrRTCPShortBuffer
	}
	chunks := chunkDst[:chunkOff+int(count)]
	items := itemDst
	pos := RTCPHeaderSize
	for i := 0; i < int(count); i++ {
		if payloadEnd-pos < 4 {
			return RTCPSDESPacket{}, 0, ErrRTCPInvalidPacket
		}
		source := binary.BigEndian.Uint32(src[pos : pos+4])
		pos += 4
		itemStart := len(items)
		for {
			if pos >= payloadEnd {
				return RTCPSDESPacket{}, 0, ErrRTCPInvalidPacket
			}
			itemType := src[pos]
			pos++
			if itemType == RTCPSDESItemEnd {
				break
			}
			if pos >= payloadEnd {
				return RTCPSDESPacket{}, 0, ErrRTCPInvalidPacket
			}
			itemLen := int(src[pos])
			pos++
			if payloadEnd-pos < itemLen {
				return RTCPSDESPacket{}, 0, ErrRTCPInvalidPacket
			}
			if cap(items)-len(items) < 1 {
				return RTCPSDESPacket{}, 0, ErrRTCPShortBuffer
			}
			itemText := src[pos : pos+itemLen]
			items = items[:len(items)+1]
			items[len(items)-1] = RTCPSDESItem{Type: itemType, Text: itemText}
			pos += itemLen
		}
		for pos&3 != 0 {
			if pos >= payloadEnd || src[pos] != 0 {
				return RTCPSDESPacket{}, 0, ErrRTCPInvalidPacket
			}
			pos++
		}
		chunks[chunkOff+i] = RTCPSDESChunk{
			Source: source,
			Items:  items[itemStart:len(items)],
		}
	}
	if pos != payloadEnd {
		return RTCPSDESPacket{}, 0, ErrRTCPInvalidPacket
	}
	return RTCPSDESPacket{Chunks: chunks[chunkOff:]}, packetLen, nil
}

// PutRTCPSenderReportPacket serializes one complete sender report packet into
// dst.
func PutRTCPSenderReportPacket(dst []byte, report RTCPSenderReport) (int, error) {
	size, err := RTCPSenderReportPacketSize(len(report.Reports))
	if err != nil {
		return 0, err
	}
	if len(dst) < size {
		return 0, ErrRTCPShortBuffer
	}
	if err := putRTCPPacketHeader(dst, RTCPSenderReportPacketType, len(report.Reports), size); err != nil {
		return 0, err
	}
	binary.BigEndian.PutUint32(dst[4:8], report.SenderSSRC)
	binary.BigEndian.PutUint32(dst[8:12], report.NTPSeconds)
	binary.BigEndian.PutUint32(dst[12:16], report.NTPFraction)
	binary.BigEndian.PutUint32(dst[16:20], report.RTPTimestamp)
	binary.BigEndian.PutUint32(dst[20:24], report.SenderPacketCount)
	binary.BigEndian.PutUint32(dst[24:28], report.SenderOctetCount)
	if err := putRTCPReportBlocks(dst[RTCPSenderReportPacketMinSize:size], report.Reports); err != nil {
		return 0, err
	}
	return size, nil
}

// AppendRTCPSenderReportPacket appends one complete sender report packet to dst
// without growing beyond dst's existing capacity.
func AppendRTCPSenderReportPacket(dst []byte, report RTCPSenderReport) ([]byte, error) {
	size, err := RTCPSenderReportPacketSize(len(report.Reports))
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < size {
		return dst, ErrRTCPShortBuffer
	}
	off := len(dst)
	out := dst[:off+size]
	if _, err := PutRTCPSenderReportPacket(out[off:], report); err != nil {
		return dst, err
	}
	return out, nil
}

// ParseRTCPSenderReportPacket parses one complete sender report packet from
// src. Reports are appended to dst without allocation and alias dst. n is the
// number of bytes consumed for the first RTCP packet.
func ParseRTCPSenderReportPacket(src []byte, dst []RTCPReportBlock) (report RTCPSenderReport, n int, err error) {
	count, packetLen, err := parseRTCPFixedPacketHeader(src, RTCPSenderReportPacketType, RTCPSenderReportPacketMinSize)
	if err != nil {
		return RTCPSenderReport{}, 0, err
	}
	expected := RTCPSenderReportPacketMinSize + int(count)*RTCPReportBlockSize
	if err := validateRTCPFixedPacketPadding(src, packetLen, expected); err != nil {
		return RTCPSenderReport{}, 0, err
	}
	reports, err := parseRTCPReportBlocks(src[RTCPSenderReportPacketMinSize:expected], int(count), dst)
	if err != nil {
		return RTCPSenderReport{}, 0, err
	}
	return RTCPSenderReport{
		SenderSSRC:        binary.BigEndian.Uint32(src[4:8]),
		NTPSeconds:        binary.BigEndian.Uint32(src[8:12]),
		NTPFraction:       binary.BigEndian.Uint32(src[12:16]),
		RTPTimestamp:      binary.BigEndian.Uint32(src[16:20]),
		SenderPacketCount: binary.BigEndian.Uint32(src[20:24]),
		SenderOctetCount:  binary.BigEndian.Uint32(src[24:28]),
		Reports:           reports,
	}, packetLen, nil
}

// PutRTCPReceiverReportPacket serializes one complete receiver report packet
// into dst.
func PutRTCPReceiverReportPacket(dst []byte, report RTCPReceiverReport) (int, error) {
	size, err := RTCPReceiverReportPacketSize(len(report.Reports))
	if err != nil {
		return 0, err
	}
	if len(dst) < size {
		return 0, ErrRTCPShortBuffer
	}
	if err := putRTCPPacketHeader(dst, RTCPReceiverReportPacketType, len(report.Reports), size); err != nil {
		return 0, err
	}
	binary.BigEndian.PutUint32(dst[4:8], report.SenderSSRC)
	if err := putRTCPReportBlocks(dst[RTCPReceiverReportPacketMinSize:size], report.Reports); err != nil {
		return 0, err
	}
	return size, nil
}

// AppendRTCPReceiverReportPacket appends one complete receiver report packet to
// dst without growing beyond dst's existing capacity.
func AppendRTCPReceiverReportPacket(dst []byte, report RTCPReceiverReport) ([]byte, error) {
	size, err := RTCPReceiverReportPacketSize(len(report.Reports))
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < size {
		return dst, ErrRTCPShortBuffer
	}
	off := len(dst)
	out := dst[:off+size]
	if _, err := PutRTCPReceiverReportPacket(out[off:], report); err != nil {
		return dst, err
	}
	return out, nil
}

// ParseRTCPReceiverReportPacket parses one complete receiver report packet from
// src. Reports are appended to dst without allocation and alias dst. n is the
// number of bytes consumed for the first RTCP packet.
func ParseRTCPReceiverReportPacket(src []byte, dst []RTCPReportBlock) (report RTCPReceiverReport, n int, err error) {
	count, packetLen, err := parseRTCPFixedPacketHeader(src, RTCPReceiverReportPacketType, RTCPReceiverReportPacketMinSize)
	if err != nil {
		return RTCPReceiverReport{}, 0, err
	}
	expected := RTCPReceiverReportPacketMinSize + int(count)*RTCPReportBlockSize
	if err := validateRTCPFixedPacketPadding(src, packetLen, expected); err != nil {
		return RTCPReceiverReport{}, 0, err
	}
	reports, err := parseRTCPReportBlocks(src[RTCPReceiverReportPacketMinSize:expected], int(count), dst)
	if err != nil {
		return RTCPReceiverReport{}, 0, err
	}
	return RTCPReceiverReport{
		SenderSSRC: binary.BigEndian.Uint32(src[4:8]),
		Reports:    reports,
	}, packetLen, nil
}

// RTCPFeedbackPacket is one complete RTCP RTPFB or PSFB packet. FCI aliases
// the caller-owned payload passed to Put or the source slice passed to Parse.
type RTCPFeedbackPacket struct {
	PacketType uint8
	FMT        uint8
	SenderSSRC uint32
	MediaSSRC  uint32
	FCI        []byte
}

// RTCPFeedbackPacketSize returns the complete RTCP feedback packet byte count,
// including any padding needed to align the packet to a 32-bit boundary.
func RTCPFeedbackPacketSize(fciLen int) (int, error) {
	if fciLen < 0 || fciLen > RTCPFeedbackMaxFCISize {
		return 0, ErrRTCPInvalidFeedback
	}
	size := RTCPFeedbackPacketHeaderSize + fciLen
	padding := rtcpPaddingLength(size)
	size += padding
	if size > RTCPMaxPacketSize {
		return 0, ErrRTCPInvalidFeedback
	}
	return size, nil
}

// PutRTCPFeedbackPacket serializes one complete RTPFB or PSFB packet into dst.
func PutRTCPFeedbackPacket(dst []byte, packet RTCPFeedbackPacket) (int, error) {
	size, err := RTCPFeedbackPacketSize(len(packet.FCI))
	if err != nil {
		return 0, err
	}
	if len(dst) < size {
		return 0, ErrRTCPShortBuffer
	}
	if packet.FMT > 0x1f ||
		(packet.PacketType != RTCPRTPFBPacketType && packet.PacketType != RTCPPSFBPacketType) {
		return 0, ErrRTCPInvalidFeedback
	}

	padding := rtcpPaddingLength(RTCPFeedbackPacketHeaderSize + len(packet.FCI))
	first := byte(RTCPVersion<<6) | (packet.FMT & 0x1f)
	if padding != 0 {
		first |= 0x20
	}
	dst[0] = first
	dst[1] = packet.PacketType
	binary.BigEndian.PutUint16(dst[2:4], uint16(size/4-1))
	binary.BigEndian.PutUint32(dst[4:8], packet.SenderSSRC)
	binary.BigEndian.PutUint32(dst[8:12], packet.MediaSSRC)
	copy(dst[12:12+len(packet.FCI)], packet.FCI)
	if padding != 0 {
		paddingStart := RTCPFeedbackPacketHeaderSize + len(packet.FCI)
		for i := 0; i < padding-1; i++ {
			dst[paddingStart+i] = 0
		}
		dst[size-1] = byte(padding)
	}
	return size, nil
}

// AppendRTCPFeedbackPacket appends one complete RTPFB or PSFB packet to dst
// without growing beyond dst's existing capacity.
func AppendRTCPFeedbackPacket(dst []byte, packet RTCPFeedbackPacket) ([]byte, error) {
	size, err := RTCPFeedbackPacketSize(len(packet.FCI))
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < size {
		return dst, ErrRTCPShortBuffer
	}
	off := len(dst)
	out := dst[:off+size]
	if _, err := PutRTCPFeedbackPacket(out[off:], packet); err != nil {
		return dst, err
	}
	return out, nil
}

// ParseRTCPFeedbackPacket parses one complete RTPFB or PSFB packet from src.
// The returned FCI aliases src. n is the number of bytes consumed for the first
// RTCP packet.
func ParseRTCPFeedbackPacket(src []byte) (packet RTCPFeedbackPacket, n int, err error) {
	if len(src) < RTCPFeedbackPacketHeaderSize {
		return RTCPFeedbackPacket{}, 0, ErrRTCPShortBuffer
	}
	if src[0]>>6 != RTCPVersion {
		return RTCPFeedbackPacket{}, 0, ErrRTCPInvalidFeedback
	}
	packetType := src[1]
	if packetType != RTCPRTPFBPacketType && packetType != RTCPPSFBPacketType {
		return RTCPFeedbackPacket{}, 0, ErrRTCPInvalidFeedback
	}
	packetLen := (int(binary.BigEndian.Uint16(src[2:4])) + 1) * 4
	if packetLen < RTCPFeedbackPacketHeaderSize {
		return RTCPFeedbackPacket{}, 0, ErrRTCPInvalidFeedback
	}
	if len(src) < packetLen {
		return RTCPFeedbackPacket{}, 0, ErrRTCPShortBuffer
	}
	padding := 0
	if src[0]&0x20 != 0 {
		padding = int(src[packetLen-1])
		if padding == 0 || padding > packetLen-RTCPFeedbackPacketHeaderSize {
			return RTCPFeedbackPacket{}, 0, ErrRTCPInvalidFeedback
		}
	}
	fciEnd := packetLen - padding
	return RTCPFeedbackPacket{
		PacketType: packetType,
		FMT:        src[0] & 0x1f,
		SenderSSRC: binary.BigEndian.Uint32(src[4:8]),
		MediaSSRC:  binary.BigEndian.Uint32(src[8:12]),
		FCI:        src[RTCPFeedbackPacketHeaderSize:fciEnd],
	}, packetLen, nil
}

func rtcpPaddingLength(size int) int {
	return (4 - (size & 3)) & 3
}

func putRTCPPacketHeader(dst []byte, packetType uint8, count int, size int) error {
	if count < 0 || count > RTCPReportMaxBlocks || size < RTCPHeaderSize || size&3 != 0 || size > RTCPMaxPacketSize {
		return ErrRTCPInvalidPacket
	}
	dst[0] = byte(RTCPVersion<<6) | byte(count)
	dst[1] = packetType
	binary.BigEndian.PutUint16(dst[2:4], uint16(size/4-1))
	return nil
}

func parseRTCPFixedPacketHeader(src []byte, packetType uint8, minSize int) (count uint8, packetLen int, err error) {
	if len(src) < minSize {
		return 0, 0, ErrRTCPShortBuffer
	}
	if src[0]>>6 != RTCPVersion || (packetType != 0 && src[1] != packetType) {
		return 0, 0, ErrRTCPInvalidPacket
	}
	packetLen = (int(binary.BigEndian.Uint16(src[2:4])) + 1) * 4
	if packetLen < minSize {
		return 0, 0, ErrRTCPInvalidPacket
	}
	if len(src) < packetLen {
		return 0, 0, ErrRTCPShortBuffer
	}
	return src[0] & 0x1f, packetLen, nil
}

func validateRTCPFixedPacketPadding(src []byte, packetLen int, expectedNoPadding int) error {
	if expectedNoPadding > packetLen {
		return ErrRTCPInvalidPacket
	}
	if src[0]&0x20 == 0 {
		if packetLen != expectedNoPadding {
			return ErrRTCPInvalidPacket
		}
		return nil
	}
	padding := int(src[packetLen-1])
	if padding == 0 || padding > packetLen-expectedNoPadding || packetLen-padding != expectedNoPadding {
		return ErrRTCPInvalidPacket
	}
	return nil
}

func rtcpPacketPayloadEnd(src []byte, packetLen int, minNoPadding int) (int, error) {
	if packetLen < minNoPadding {
		return 0, ErrRTCPInvalidPacket
	}
	if src[0]&0x20 == 0 {
		return packetLen, nil
	}
	padding := int(src[packetLen-1])
	if padding == 0 || padding > packetLen-minNoPadding {
		return 0, ErrRTCPInvalidPacket
	}
	return packetLen - padding, nil
}

func rtcpSDESPacketPayloadSize(chunks []RTCPSDESChunk) (int, error) {
	if len(chunks) > RTCPSDESMaxChunks {
		return 0, ErrRTCPInvalidPacket
	}
	total := 0
	for _, chunk := range chunks {
		size := 4
		for _, item := range chunk.Items {
			itemSize, err := validateRTCPSDESItem(item)
			if err != nil {
				return 0, err
			}
			size += itemSize
		}
		size++ // END item.
		size += rtcpPaddingLength(size)
		total += size
		if RTCPHeaderSize+total > RTCPMaxPacketSize {
			return 0, ErrRTCPInvalidPacket
		}
	}
	return total, nil
}

func validateRTCPSDESItem(item RTCPSDESItem) (int, error) {
	if item.Type == RTCPSDESItemEnd || len(item.Text) > RTCPSDESItemMaxTextLen {
		return 0, ErrRTCPInvalidPacket
	}
	return 2 + len(item.Text), nil
}

func putRTCPReportBlocks(dst []byte, reports []RTCPReportBlock) error {
	if len(dst) < len(reports)*RTCPReportBlockSize {
		return ErrRTCPShortBuffer
	}
	for i := range reports {
		report := reports[i]
		if report.CumulativePacketsLost < RTCPReportCumulativeLostMin ||
			report.CumulativePacketsLost > RTCPReportCumulativeLostMax {
			return ErrRTCPInvalidPacket
		}
		off := i * RTCPReportBlockSize
		binary.BigEndian.PutUint32(dst[off:off+4], report.SSRC)
		lost := uint32(report.CumulativePacketsLost) & 0x00ffffff
		dst[off+4] = report.FractionLost
		dst[off+5] = byte(lost >> 16)
		dst[off+6] = byte(lost >> 8)
		dst[off+7] = byte(lost)
		binary.BigEndian.PutUint32(dst[off+8:off+12], report.ExtendedHighestSequenceNumber)
		binary.BigEndian.PutUint32(dst[off+12:off+16], report.InterarrivalJitter)
		binary.BigEndian.PutUint32(dst[off+16:off+20], report.LastSenderReport)
		binary.BigEndian.PutUint32(dst[off+20:off+24], report.DelaySinceLastSenderReport)
	}
	return nil
}

func parseRTCPReportBlocks(src []byte, count int, dst []RTCPReportBlock) ([]RTCPReportBlock, error) {
	size := count * RTCPReportBlockSize
	if len(src) < size {
		return nil, ErrRTCPShortBuffer
	}
	off := len(dst)
	if cap(dst)-off < count {
		return nil, ErrRTCPShortBuffer
	}
	out := dst[:off+count]
	for i := 0; i < count; i++ {
		srcOff := i * RTCPReportBlockSize
		lost := int32(uint32(src[srcOff+5])<<16 | uint32(src[srcOff+6])<<8 | uint32(src[srcOff+7]))
		if lost&0x00800000 != 0 {
			lost |= ^int32(0x00ffffff)
		}
		out[off+i] = RTCPReportBlock{
			SSRC:                          binary.BigEndian.Uint32(src[srcOff : srcOff+4]),
			FractionLost:                  src[srcOff+4],
			CumulativePacketsLost:         lost,
			ExtendedHighestSequenceNumber: binary.BigEndian.Uint32(src[srcOff+8 : srcOff+12]),
			InterarrivalJitter:            binary.BigEndian.Uint32(src[srcOff+12 : srcOff+16]),
			LastSenderReport:              binary.BigEndian.Uint32(src[srcOff+16 : srcOff+20]),
			DelaySinceLastSenderReport:    binary.BigEndian.Uint32(src[srcOff+20 : srcOff+24]),
		}
	}
	return out[off:], nil
}

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

// EncoderWebRTCLayerRefreshRequestTarget validates entries against config and
// returns the inclusive spatial/temporal cap that covers all requested LRR
// target layers. The returned cap can be passed to
// RTCPicture.ActiveDecodeTargetsRTPOptions after the forced refresh picture is
// encoded. ok is false when entries is empty.
func EncoderWebRTCLayerRefreshRequestTarget(
	config EncoderConfig,
	entries []AV1RTCPLayerRefreshRequestEntry,
) (target AV1RTCPLayerRefreshLayerIndex, ok bool, err error) {
	normalized, err := SetWebRTCEncoderSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return AV1RTCPLayerRefreshLayerIndex{}, false, err
	}
	for i := range entries {
		if err := encoderWebRTCValidateLayerRefreshRequestForConfig(normalized, entries[i]); err != nil {
			return AV1RTCPLayerRefreshLayerIndex{}, false, err
		}
		if !ok || entries[i].Target.SpatialID > target.SpatialID {
			target.SpatialID = entries[i].Target.SpatialID
		}
		if !ok || entries[i].Target.TemporalID > target.TemporalID {
			target.TemporalID = entries[i].Target.TemporalID
		}
		ok = true
	}
	return target, ok, nil
}

// EncoderWebRTCRTCPFeedbackLayerRefreshTarget parses a single RTCP feedback
// packet and returns the inclusive active decode-target cap requested by valid
// AV1 LRR feedback. Non-LRR RTPFB/PSFB feedback returns ok=false.
func EncoderWebRTCRTCPFeedbackLayerRefreshTarget(
	config EncoderConfig,
	packet RTCPFeedbackPacket,
	lrrScratch []AV1RTCPLayerRefreshRequestEntry,
) (target AV1RTCPLayerRefreshLayerIndex, ok bool, err error) {
	switch packet.PacketType {
	case RTCPRTPFBPacketType:
		return AV1RTCPLayerRefreshLayerIndex{}, false, nil
	case RTCPPSFBPacketType:
	default:
		return AV1RTCPLayerRefreshLayerIndex{}, false, ErrRTCPInvalidFeedback
	}
	if packet.FMT != RTCPPSFBLayerRefreshRequestFMT {
		return AV1RTCPLayerRefreshLayerIndex{}, false, nil
	}
	start := len(lrrScratch)
	entries, err := ParseAV1RTCPLayerRefreshRequestEntries(packet.FCI, lrrScratch)
	if err != nil {
		return AV1RTCPLayerRefreshLayerIndex{}, false, err
	}
	return EncoderWebRTCLayerRefreshRequestTarget(config, entries[start:])
}

// EncoderWebRTCRTCPPacketsLayerRefreshTarget scans parsed generic RTCP packets
// and returns the inclusive active decode-target cap requested by AV1 LRR
// feedback. Non-feedback RTCP packets are ignored.
func EncoderWebRTCRTCPPacketsLayerRefreshTarget(
	config EncoderConfig,
	packets []RTCPPacket,
	lrrScratch []AV1RTCPLayerRefreshRequestEntry,
) (target AV1RTCPLayerRefreshLayerIndex, ok bool, err error) {
	for i := range packets {
		packet := packets[i]
		if packet.PacketType != RTCPRTPFBPacketType && packet.PacketType != RTCPPSFBPacketType {
			continue
		}
		if len(packet.Payload) < RTCPFeedbackCommonSize {
			return AV1RTCPLayerRefreshLayerIndex{}, false, ErrRTCPInvalidFeedback
		}
		current, currentOK, err := EncoderWebRTCRTCPFeedbackLayerRefreshTarget(config, RTCPFeedbackPacket{
			PacketType: packet.PacketType,
			FMT:        packet.Count,
			SenderSSRC: binary.BigEndian.Uint32(packet.Payload[0:4]),
			MediaSSRC:  binary.BigEndian.Uint32(packet.Payload[4:8]),
			FCI:        packet.Payload[RTCPFeedbackCommonSize:],
		}, lrrScratch)
		if err != nil {
			return AV1RTCPLayerRefreshLayerIndex{}, false, err
		}
		if !currentOK {
			continue
		}
		if !ok || current.SpatialID > target.SpatialID {
			target.SpatialID = current.SpatialID
		}
		if !ok || current.TemporalID > target.TemporalID {
			target.TemporalID = current.TemporalID
		}
		ok = true
	}
	return target, ok, nil
}

// EncoderWebRTCRTCPCompoundPacketsLayerRefreshTarget parses a raw compound RTCP
// packet stream and returns the inclusive active decode-target cap requested by
// AV1 LRR feedback. Parsed packets are appended into packetScratch without
// allocation and returned as the newly parsed suffix.
func EncoderWebRTCRTCPCompoundPacketsLayerRefreshTarget(
	config EncoderConfig,
	compound []byte,
	packetScratch []RTCPPacket,
	lrrScratch []AV1RTCPLayerRefreshRequestEntry,
) (target AV1RTCPLayerRefreshLayerIndex, ok bool, packets []RTCPPacket, err error) {
	packets, err = ParseRTCPCompoundPackets(compound, packetScratch)
	if err != nil {
		return AV1RTCPLayerRefreshLayerIndex{}, false, packets, err
	}
	target, ok, err = EncoderWebRTCRTCPPacketsLayerRefreshTarget(config, packets, lrrScratch)
	return target, ok, packets, err
}

// EncoderWebRTCRTCPFeedbackRequiresKeyFrame reports whether a parsed RTCP
// feedback packet should be satisfied by forcing the next WebRTC encoder
// picture to be a key picture. PLI and non-empty FIR feedback require a key
// picture. Valid AV1 LRR feedback also returns true because the current encoder
// satisfies layer-refresh requests by refreshing the full configured picture.
// Transport feedback, NACK, REMB, and unknown PSFB feedback return false after
// any packet-type-specific FCI validation performed here.
func EncoderWebRTCRTCPFeedbackRequiresKeyFrame(
	config EncoderConfig,
	packet RTCPFeedbackPacket,
	firScratch []RTCPFullIntraRequestEntry,
	lrrScratch []AV1RTCPLayerRefreshRequestEntry,
) (bool, error) {
	switch packet.PacketType {
	case RTCPRTPFBPacketType:
		return false, nil
	case RTCPPSFBPacketType:
	default:
		return false, ErrRTCPInvalidFeedback
	}

	switch packet.FMT {
	case RTCPPSFBPictureLossIndicationFMT:
		if err := ParseRTCPPictureLossIndicationFCI(packet.FCI); err != nil {
			return false, err
		}
		return true, nil
	case RTCPPSFBFullIntraRequestFMT:
		entries, err := ParseRTCPFullIntraRequestEntries(packet.FCI, firScratch)
		if err != nil {
			return false, err
		}
		return len(entries) > len(firScratch), nil
	case RTCPPSFBLayerRefreshRequestFMT:
		entries, err := ParseAV1RTCPLayerRefreshRequestEntries(packet.FCI, lrrScratch)
		if err != nil {
			return false, err
		}
		if err := EncoderWebRTCValidateLayerRefreshRequests(config, entries[len(lrrScratch):]); err != nil {
			return false, err
		}
		return len(entries) > len(lrrScratch), nil
	default:
		return false, nil
	}
}

// EncoderWebRTCRTCPPacketsRequireKeyFrame scans parsed generic RTCP packets and
// reports whether any RTPFB/PSFB feedback packet should force the next WebRTC
// encoder picture to be a key picture. Non-feedback RTCP packets are ignored.
func EncoderWebRTCRTCPPacketsRequireKeyFrame(
	config EncoderConfig,
	packets []RTCPPacket,
	firScratch []RTCPFullIntraRequestEntry,
	lrrScratch []AV1RTCPLayerRefreshRequestEntry,
) (bool, error) {
	for i := range packets {
		packet := packets[i]
		if packet.PacketType != RTCPRTPFBPacketType && packet.PacketType != RTCPPSFBPacketType {
			continue
		}
		if len(packet.Payload) < RTCPFeedbackCommonSize {
			return false, ErrRTCPInvalidFeedback
		}
		force, err := EncoderWebRTCRTCPFeedbackRequiresKeyFrame(config, RTCPFeedbackPacket{
			PacketType: packet.PacketType,
			FMT:        packet.Count,
			SenderSSRC: binary.BigEndian.Uint32(packet.Payload[0:4]),
			MediaSSRC:  binary.BigEndian.Uint32(packet.Payload[4:8]),
			FCI:        packet.Payload[RTCPFeedbackCommonSize:],
		}, firScratch, lrrScratch)
		if err != nil || force {
			return force, err
		}
	}
	return false, nil
}

// EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame parses a raw compound RTCP
// packet stream and reports whether any feedback packet should force the next
// WebRTC encoder picture to be a key picture. Parsed packets are appended into
// packetScratch without allocation and returned as the newly parsed suffix.
// Non-feedback RTCP packets are ignored. PLI, non-empty FIR, and valid AV1 LRR
// feedback can force a key picture; transport feedback, NACK, REMB, and
// unknown PSFB feedback do not.
func EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame(
	config EncoderConfig,
	compound []byte,
	packetScratch []RTCPPacket,
	firScratch []RTCPFullIntraRequestEntry,
	lrrScratch []AV1RTCPLayerRefreshRequestEntry,
) (bool, []RTCPPacket, error) {
	packets, err := ParseRTCPCompoundPackets(compound, packetScratch)
	if err != nil {
		return false, packets, err
	}
	force, err := EncoderWebRTCRTCPPacketsRequireKeyFrame(config, packets, firScratch, lrrScratch)
	return force, packets, err
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
