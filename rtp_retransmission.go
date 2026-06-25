package goav1

import (
	"encoding/binary"
	"errors"
)

// ErrRTPInvalidRetransmissionCache is returned when an RTP retransmission cache
// is unbound or bound to an invalid slot window.
var ErrRTPInvalidRetransmissionCache = errors.New("goav1: invalid rtp retransmission cache")

// RTPRetransmissionCacheSlot is caller-owned storage for one cached RTP packet.
// Packet's capacity is the maximum RTP packet size this slot can retain.
type RTPRetransmissionCacheSlot struct {
	Packet []byte

	sequenceNumber uint16
	occupied       bool
}

// RTPRetransmissionPacketSpan describes one cached packet appended to a
// retransmit buffer.
type RTPRetransmissionPacketSpan struct {
	SequenceNumber uint16
	Offset         int
	Length         int
}

// RTPRTXPacketConfig describes the RTP retransmission packet header fields for
// RFC 4588 RTX wrapping. PayloadType is the negotiated RTX payload type,
// SequenceNumber is the retransmission stream sequence number, and SSRC is the
// retransmission stream SSRC.
//
// By default the original packet's RTP header extension is copied into the RTX
// packet, matching RFC 4588. Set ExtensionOverride when an integration needs to
// clear or replace packet-specific extension values such as a transport sequence
// number before sending the retransmission.
type RTPRTXPacketConfig struct {
	PayloadType    uint8
	SequenceNumber uint16
	SSRC           uint32

	ExtensionProfile  uint16
	ExtensionPayload  []byte
	ExtensionOverride bool
}

// RTPRTXPacket is a parsed RFC 4588 retransmission packet. OriginalPayload
// aliases the RTX packet input and excludes the two-octet OSN header.
type RTPRTXPacket struct {
	Header                 RTPHeader
	OriginalSequenceNumber uint16
	OriginalPayload        []byte
}

// RTPRTXPacketSpan describes one cached packet appended as an RTX packet.
type RTPRTXPacketSpan struct {
	OriginalSequenceNumber uint16
	RTXSequenceNumber      uint16
	Offset                 int
	Length                 int
}

// RTPRetransmissionCache keeps a bounded raw RTP packet history for sender-side
// Generic NACK repair. It owns no transport policy: callers decide cache size,
// RTCP parsing, pacing, RTX wrapping, and whether unrepaired NACKs should force
// a keyframe.
//
// A cache is not safe for concurrent use; serialize calls per RTP stream.
type RTPRetransmissionCache struct {
	slots []RTPRetransmissionCacheSlot
}

// BindRTPRetransmissionCache binds a retransmission cache to caller-owned slot
// storage. The slot window must hold at least one packet and at most 32768
// packets, the maximum unambiguous RTP sequence-number distance.
func BindRTPRetransmissionCache(slots []RTPRetransmissionCacheSlot) (RTPRetransmissionCache, error) {
	if len(slots) == 0 || len(slots) > 1<<15 {
		return RTPRetransmissionCache{}, ErrRTPInvalidRetransmissionCache
	}
	for i := range slots {
		slots[i].Packet = slots[i].Packet[:0]
		slots[i].sequenceNumber = 0
		slots[i].occupied = false
	}
	return RTPRetransmissionCache{slots: slots}, nil
}

// Reset clears cached packet state while preserving each slot's packet buffer.
func (c *RTPRetransmissionCache) Reset() {
	if c == nil {
		return
	}
	for i := range c.slots {
		c.slots[i].Packet = c.slots[i].Packet[:0]
		c.slots[i].sequenceNumber = 0
		c.slots[i].occupied = false
	}
}

// Store parses packet, indexes it by RTP sequence number, and copies the raw
// packet bytes into the matching cache slot.
func (c *RTPRetransmissionCache) Store(packet []byte) error {
	if c == nil || len(c.slots) == 0 {
		return ErrRTPInvalidRetransmissionCache
	}
	parsed, err := ParseRTPPacket(packet)
	if err != nil {
		return err
	}
	slot := &c.slots[int(parsed.Header.SequenceNumber%uint16(len(c.slots)))]
	if cap(slot.Packet) < len(packet) {
		return ErrRTPShortBuffer
	}
	slot.Packet = slot.Packet[:len(packet)]
	copy(slot.Packet, packet)
	slot.sequenceNumber = parsed.Header.SequenceNumber
	slot.occupied = true
	return nil
}

// Contains reports whether sequenceNumber is still present in the cache.
func (c *RTPRetransmissionCache) Contains(sequenceNumber uint16) bool {
	if c == nil || len(c.slots) == 0 {
		return false
	}
	slot := &c.slots[int(sequenceNumber%uint16(len(c.slots)))]
	return slot.occupied && slot.sequenceNumber == sequenceNumber
}

// AppendPacket appends a cached raw RTP packet to dst without growing beyond
// dst's existing capacity. The bool reports whether the packet was found.
func (c *RTPRetransmissionCache) AppendPacket(dst []byte, sequenceNumber uint16) ([]byte, bool, error) {
	if c == nil || len(c.slots) == 0 {
		return dst, false, ErrRTPInvalidRetransmissionCache
	}
	slot := &c.slots[int(sequenceNumber%uint16(len(c.slots)))]
	if !slot.occupied || slot.sequenceNumber != sequenceNumber {
		return dst, false, nil
	}
	if cap(dst)-len(dst) < len(slot.Packet) {
		return dst, true, ErrRTPShortBuffer
	}
	off := len(dst)
	out := dst[:off+len(slot.Packet)]
	copy(out[off:], slot.Packet)
	return out, true, nil
}

// AppendPacketsForRTCPGenericNACKPairs appends cached packets covered by pairs
// to dst and fills spans with the packet locations. Missing packets are skipped
// so callers can decide whether to wait, send a keyframe, or rely on later NACKs.
func (c *RTPRetransmissionCache) AppendPacketsForRTCPGenericNACKPairs(
	dst []byte, spans []RTPRetransmissionPacketSpan, pairs []RTCPGenericNACKPair,
) ([]byte, int, error) {
	if c == nil || len(c.slots) == 0 {
		return dst, 0, ErrRTPInvalidRetransmissionCache
	}
	out := dst
	count := 0
	for _, pair := range pairs {
		var err error
		out, count, err = c.appendRetransmissionPacketForSequence(out, spans, count, pair.PacketID)
		if err != nil {
			return dst, 0, err
		}
		for bit := uint16(0); bit < 16; bit++ {
			if pair.LostPacketBitmask&(1<<bit) == 0 {
				continue
			}
			out, count, err = c.appendRetransmissionPacketForSequence(out, spans, count, pair.PacketID+bit+1)
			if err != nil {
				return dst, 0, err
			}
		}
	}
	return out, count, nil
}

func (c *RTPRetransmissionCache) appendRetransmissionPacketForSequence(
	dst []byte, spans []RTPRetransmissionPacketSpan, count int, sequenceNumber uint16,
) ([]byte, int, error) {
	if !c.Contains(sequenceNumber) {
		return dst, count, nil
	}
	if count >= len(spans) {
		return dst, count, ErrRTPShortBuffer
	}
	before := len(dst)
	out, found, err := c.AppendPacket(dst, sequenceNumber)
	if err != nil {
		return dst, count, err
	}
	if !found {
		return out, count, nil
	}
	spans[count] = RTPRetransmissionPacketSpan{
		SequenceNumber: sequenceNumber,
		Offset:         before,
		Length:         len(out) - before,
	}
	return out, count + 1, nil
}

// RTPRTXPacketSize reports the bytes needed to wrap original as an RFC 4588 RTX
// packet under config. RTP padding on the original packet is not copied into the
// RTX payload.
func RTPRTXPacketSize(original []byte, config RTPRTXPacketConfig) (int, error) {
	parsed, err := ParseRTPPacket(original)
	if err != nil {
		return 0, err
	}
	header := rtpRTXHeader(parsed.Header, config)
	headerSize, err := RTPHeaderSize(header)
	if err != nil {
		return 0, err
	}
	return headerSize + 2 + len(parsed.Payload), nil
}

// PutRTPRTXPacket writes an RFC 4588 retransmission packet for original into
// dst. The RTX payload is the two-octet original sequence number followed by the
// original RTP payload, including any original payload-format headers.
func PutRTPRTXPacket(dst []byte, original []byte, config RTPRTXPacketConfig) (int, error) {
	parsed, err := ParseRTPPacket(original)
	if err != nil {
		return 0, err
	}
	header := rtpRTXHeader(parsed.Header, config)
	headerSize, err := RTPHeaderSize(header)
	if err != nil {
		return 0, err
	}
	size := headerSize + 2 + len(parsed.Payload)
	if len(dst) < size {
		return 0, ErrRTPShortBuffer
	}
	n, err := PutRTPHeader(dst, header)
	if err != nil {
		return 0, err
	}
	binary.BigEndian.PutUint16(dst[n:n+2], parsed.Header.SequenceNumber)
	n += 2
	n += copy(dst[n:], parsed.Payload)
	return n, nil
}

// AppendRTPRTXPacket appends an RFC 4588 retransmission packet for original to
// dst without growing beyond dst's existing capacity.
func AppendRTPRTXPacket(dst []byte, original []byte, config RTPRTXPacketConfig) ([]byte, error) {
	size, err := RTPRTXPacketSize(original, config)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < size {
		return dst, ErrRTPShortBuffer
	}
	off := len(dst)
	out := dst[:off+size]
	n, err := PutRTPRTXPacket(out[off:], original, config)
	if err != nil {
		return dst, err
	}
	return out[:off+n], nil
}

// ParseRTPRTXPacket parses one RFC 4588 RTX packet. The returned header is the
// RTX packet header; OriginalSequenceNumber is the OSN payload header field.
func ParseRTPRTXPacket(src []byte) (RTPRTXPacket, error) {
	parsed, err := ParseRTPPacket(src)
	if err != nil {
		return RTPRTXPacket{}, err
	}
	if len(parsed.Payload) < 2 {
		return RTPRTXPacket{}, ErrRTPShortPayload
	}
	return RTPRTXPacket{
		Header:                 parsed.Header,
		OriginalSequenceNumber: binary.BigEndian.Uint16(parsed.Payload[:2]),
		OriginalPayload:        parsed.Payload[2:],
	}, nil
}

// RTPPacketFromRTXSize reports the bytes needed to restore an original RTP
// packet from an RFC 4588 RTX packet using the negotiated original payload type
// and SSRC.
func RTPPacketFromRTXSize(rtx []byte, payloadType uint8, ssrc uint32) (int, error) {
	parsed, err := ParseRTPRTXPacket(rtx)
	if err != nil {
		return 0, err
	}
	header := rtpOriginalHeaderFromRTX(parsed, payloadType, ssrc)
	return RTPPacketSize(header, parsed.OriginalPayload, 0)
}

// PutRTPPacketFromRTX restores the original RTP packet shape from an RFC 4588
// RTX packet. The restored packet uses the OSN sequence number, the RTX packet's
// original media timestamp, marker bit, CSRC list, and RTP header extension, plus
// the caller-supplied original payload type and SSRC.
func PutRTPPacketFromRTX(dst []byte, rtx []byte, payloadType uint8, ssrc uint32) (int, error) {
	parsed, err := ParseRTPRTXPacket(rtx)
	if err != nil {
		return 0, err
	}
	header := rtpOriginalHeaderFromRTX(parsed, payloadType, ssrc)
	return PutRTPPacket(dst, header, parsed.OriginalPayload, 0)
}

// AppendRTPPacketFromRTX appends a restored original RTP packet from an RFC 4588
// RTX packet without growing beyond dst's existing capacity.
func AppendRTPPacketFromRTX(dst []byte, rtx []byte, payloadType uint8, ssrc uint32) ([]byte, error) {
	size, err := RTPPacketFromRTXSize(rtx, payloadType, ssrc)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < size {
		return dst, ErrRTPShortBuffer
	}
	off := len(dst)
	out := dst[:off+size]
	n, err := PutRTPPacketFromRTX(out[off:], rtx, payloadType, ssrc)
	if err != nil {
		return dst, err
	}
	return out[:off+n], nil
}

// AppendRTXPacketsForRTCPGenericNACKPairs appends cached packets requested by
// pairs as RFC 4588 RTX packets. Missing originals are skipped. The returned
// config has SequenceNumber advanced once per emitted RTX packet so callers can
// reuse it as the next retransmission-stream sequence state.
func (c *RTPRetransmissionCache) AppendRTXPacketsForRTCPGenericNACKPairs(
	dst []byte, spans []RTPRTXPacketSpan, pairs []RTCPGenericNACKPair, config RTPRTXPacketConfig,
) ([]byte, int, RTPRTXPacketConfig, error) {
	if c == nil || len(c.slots) == 0 {
		return dst, 0, config, ErrRTPInvalidRetransmissionCache
	}
	out := dst
	count := 0
	for _, pair := range pairs {
		var err error
		out, count, config, err = c.appendRTXPacketForSequence(out, spans, count, pair.PacketID, config)
		if err != nil {
			return dst, 0, config, err
		}
		for bit := uint16(0); bit < 16; bit++ {
			if pair.LostPacketBitmask&(1<<bit) == 0 {
				continue
			}
			out, count, config, err = c.appendRTXPacketForSequence(out, spans, count, pair.PacketID+bit+1, config)
			if err != nil {
				return dst, 0, config, err
			}
		}
	}
	return out, count, config, nil
}

func (c *RTPRetransmissionCache) appendRTXPacketForSequence(
	dst []byte, spans []RTPRTXPacketSpan, count int, sequenceNumber uint16, config RTPRTXPacketConfig,
) ([]byte, int, RTPRTXPacketConfig, error) {
	slot := c.cachedSlot(sequenceNumber)
	if slot == nil {
		return dst, count, config, nil
	}
	if count >= len(spans) {
		return dst, count, config, ErrRTPShortBuffer
	}
	before := len(dst)
	out, err := AppendRTPRTXPacket(dst, slot.Packet, config)
	if err != nil {
		return dst, count, config, err
	}
	spans[count] = RTPRTXPacketSpan{
		OriginalSequenceNumber: sequenceNumber,
		RTXSequenceNumber:      config.SequenceNumber,
		Offset:                 before,
		Length:                 len(out) - before,
	}
	config.SequenceNumber++
	return out, count + 1, config, nil
}

func (c *RTPRetransmissionCache) cachedSlot(sequenceNumber uint16) *RTPRetransmissionCacheSlot {
	if c == nil || len(c.slots) == 0 {
		return nil
	}
	slot := &c.slots[int(sequenceNumber%uint16(len(c.slots)))]
	if !slot.occupied || slot.sequenceNumber != sequenceNumber {
		return nil
	}
	return slot
}

func rtpRTXHeader(original RTPHeader, config RTPRTXPacketConfig) RTPHeader {
	header := RTPHeader{
		Marker:         original.Marker,
		PayloadType:    config.PayloadType,
		SequenceNumber: config.SequenceNumber,
		Timestamp:      original.Timestamp,
		SSRC:           config.SSRC,
		CSRC:           original.CSRC,
		CSRCCount:      original.CSRCCount,
	}
	if config.ExtensionOverride {
		header.ExtensionProfile = config.ExtensionProfile
		header.ExtensionPayload = config.ExtensionPayload
	} else {
		header.ExtensionProfile = original.ExtensionProfile
		header.ExtensionPayload = original.ExtensionPayload
	}
	return header
}

func rtpOriginalHeaderFromRTX(rtx RTPRTXPacket, payloadType uint8, ssrc uint32) RTPHeader {
	return RTPHeader{
		Marker:           rtx.Header.Marker,
		PayloadType:      payloadType,
		SequenceNumber:   rtx.OriginalSequenceNumber,
		Timestamp:        rtx.Header.Timestamp,
		SSRC:             ssrc,
		CSRC:             rtx.Header.CSRC,
		CSRCCount:        rtx.Header.CSRCCount,
		ExtensionProfile: rtx.Header.ExtensionProfile,
		ExtensionPayload: rtx.Header.ExtensionPayload,
	}
}
