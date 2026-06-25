package goav1

import "errors"

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
