package goav1

import "errors"

// ErrRTPInvalidPacketSequencer is returned when an RTP packet sequencer is
// unbound or bound to an invalid slot window.
var ErrRTPInvalidPacketSequencer = errors.New("goav1: invalid rtp packet sequencer")

// RTPSequencedPacket is one complete RTP packet released by RTPPacketSequencer
// in sequence-number order. RawPacket and Packet.Payload alias the packet bytes
// passed to Push. AfterLoss is set on the first packet released after the
// caller skips a missing sequence-number gap or after a packet arrives beyond
// the configured reorder window.
type RTPSequencedPacket struct {
	RawPacket []byte
	Packet    RTPPacket
	AfterLoss bool
}

// RTPPacketSequencerSlot is caller-owned storage for RTPPacketSequencer.
type RTPPacketSequencerSlot struct {
	rawPacket      []byte
	packet         RTPPacket
	sequenceNumber uint16
	occupied       bool
}

// RTPPacketSequencer reorders complete RTP packets by sequence number, drops
// duplicates and late packets, and marks the first released packet after a
// skipped gap. It is intentionally policy-light: callers still decide when a
// missing packet has timed out, whether to NACK, and when to call SkipMissing.
//
// A sequencer is not safe for concurrent use; serialize calls per RTP stream.
type RTPPacketSequencer struct {
	slots []RTPPacketSequencerSlot

	nextSequenceNumber uint16
	initialized        bool
	afterLoss          bool
	buffered           int
}

// BindRTPPacketSequencer binds a packet sequencer to caller-owned slot storage.
// The slot window must hold at least one packet and at most 32768 packets, the
// maximum unambiguous RTP sequence-number distance.
func BindRTPPacketSequencer(slots []RTPPacketSequencerSlot) (RTPPacketSequencer, error) {
	if len(slots) == 0 || len(slots) > 1<<15 {
		return RTPPacketSequencer{}, ErrRTPInvalidPacketSequencer
	}
	clear(slots)
	return RTPPacketSequencer{slots: slots}, nil
}

// Reset clears all buffered packets and sequence-number state.
func (s *RTPPacketSequencer) Reset() {
	if s == nil {
		return
	}
	clear(s.slots)
	s.nextSequenceNumber = 0
	s.initialized = false
	s.afterLoss = false
	s.buffered = 0
}

// Initialized reports whether the sequencer has accepted at least one packet.
func (s *RTPPacketSequencer) Initialized() bool {
	return s != nil && s.initialized
}

// Buffered reports the number of packets currently waiting for missing earlier
// sequence numbers.
func (s *RTPPacketSequencer) Buffered() int {
	if s == nil {
		return 0
	}
	return s.buffered
}

// ExpectedSequenceNumber returns the next sequence number that would be
// released immediately.
func (s *RTPPacketSequencer) ExpectedSequenceNumber() (uint16, bool) {
	if s == nil || !s.initialized {
		return 0, false
	}
	return s.nextSequenceNumber, true
}

// AppendMissingSequenceNumbers appends missing RTP sequence numbers between the
// next expected packet and the furthest buffered packet. It does not advance
// the sequencer or decide whether those packets have timed out.
func (s *RTPPacketSequencer) AppendMissingSequenceNumbers(dst []uint16) ([]uint16, error) {
	summary := s.missingSummary()
	if summary.missingCount == 0 {
		return dst, nil
	}
	if cap(dst)-len(dst) < summary.missingCount {
		return dst, ErrRTPShortBuffer
	}
	off := len(dst)
	out := dst[:off+summary.missingCount]
	write := off
	for distance := uint16(0); distance < summary.latestDistance; distance++ {
		seq := s.nextSequenceNumber + distance
		if s.hasBufferedSequence(seq) {
			continue
		}
		out[write] = seq
		write++
	}
	return out, nil
}

// AppendMissingRTCPGenericNACKPairs appends Generic NACK PID/BLP pairs for the
// currently missing sequence numbers reported by AppendMissingSequenceNumbers.
// The caller owns the retransmission policy and surrounding RTCP feedback
// packet.
func (s *RTPPacketSequencer) AppendMissingRTCPGenericNACKPairs(dst []RTCPGenericNACKPair) ([]RTCPGenericNACKPair, error) {
	summary := s.missingSummary()
	if summary.nackPairCount == 0 {
		return dst, nil
	}
	if cap(dst)-len(dst) < summary.nackPairCount {
		return dst, ErrRTCPShortBuffer
	}
	off := len(dst)
	out := dst[:off+summary.nackPairCount]
	pairIndex := off
	havePair := false
	for distance := uint16(0); distance < summary.latestDistance; distance++ {
		seq := s.nextSequenceNumber + distance
		if s.hasBufferedSequence(seq) {
			continue
		}
		if !havePair {
			out[pairIndex] = RTCPGenericNACKPair{PacketID: seq}
			havePair = true
			continue
		}
		pair := &out[pairIndex]
		pairDistance := uint16(seq - pair.PacketID)
		if pairDistance <= 16 {
			pair.LostPacketBitmask |= 1 << (pairDistance - 1)
			continue
		}
		pairIndex++
		out[pairIndex] = RTCPGenericNACKPair{PacketID: seq}
	}
	return out, nil
}

// Push parses packet, stores it if it arrived ahead of a gap, and appends any
// now-contiguous packets to out. Packets older than the next expected sequence
// number are treated as duplicates or late arrivals and ignored.
func (s *RTPPacketSequencer) Push(packet []byte, out []RTPSequencedPacket) ([]RTPSequencedPacket, error) {
	if s == nil || len(s.slots) == 0 {
		return out, ErrRTPInvalidPacketSequencer
	}
	parsed, err := ParseRTPPacket(packet)
	if err != nil {
		return out, err
	}
	seq := parsed.Header.SequenceNumber
	if !s.initialized {
		s.initialized = true
		s.nextSequenceNumber = seq
	}
	if seq != s.nextSequenceNumber && !rtpSequenceNumberAhead(seq, s.nextSequenceNumber) {
		return out, nil
	}
	if distance := uint16(seq - s.nextSequenceNumber); distance >= uint16(len(s.slots)) {
		s.clearBuffered()
		s.nextSequenceNumber = seq
		s.afterLoss = true
	}
	s.store(seq, packet, parsed)
	return s.drain(out), nil
}

// SkipMissing declares the current sequence-number gap lost and releases the
// earliest buffered packet, followed by any contiguous buffered packets. The
// first released packet is marked AfterLoss so callers can route it through
// DecodeRTPPacketAfterLoss or DecodeRTPSequencedPacket.
func (s *RTPPacketSequencer) SkipMissing(out []RTPSequencedPacket) []RTPSequencedPacket {
	if s == nil || !s.initialized || s.buffered == 0 {
		return out
	}
	out = s.drain(out)
	if s.buffered == 0 {
		return out
	}
	bestSeq, ok := s.earliestBufferedSequence()
	if !ok {
		return out
	}
	s.nextSequenceNumber = bestSeq
	s.afterLoss = true
	return s.drain(out)
}

// Flush skips gaps until all currently buffered packets have been released.
func (s *RTPPacketSequencer) Flush(out []RTPSequencedPacket) []RTPSequencedPacket {
	for s != nil && s.buffered != 0 {
		beforeBuffered := s.buffered
		beforeOut := len(out)
		out = s.SkipMissing(out)
		if s.buffered == beforeBuffered && len(out) == beforeOut {
			break
		}
	}
	return out
}

func (s *RTPPacketSequencer) store(seq uint16, raw []byte, packet RTPPacket) {
	slot := &s.slots[int(seq%uint16(len(s.slots)))]
	if slot.occupied && slot.sequenceNumber == seq {
		return
	}
	if slot.occupied {
		s.buffered--
	}
	*slot = RTPPacketSequencerSlot{
		rawPacket:      raw,
		packet:         packet,
		sequenceNumber: seq,
		occupied:       true,
	}
	s.buffered++
}

func (s *RTPPacketSequencer) drain(out []RTPSequencedPacket) []RTPSequencedPacket {
	for s.initialized {
		slot := &s.slots[int(s.nextSequenceNumber%uint16(len(s.slots)))]
		if !slot.occupied || slot.sequenceNumber != s.nextSequenceNumber {
			return out
		}
		out = append(out, RTPSequencedPacket{
			RawPacket: slot.rawPacket,
			Packet:    slot.packet,
			AfterLoss: s.afterLoss,
		})
		s.afterLoss = false
		*slot = RTPPacketSequencerSlot{}
		s.buffered--
		s.nextSequenceNumber++
	}
	return out
}

func (s *RTPPacketSequencer) earliestBufferedSequence() (uint16, bool) {
	var best uint16
	bestDistance := uint16(0xffff)
	ok := false
	for i := range s.slots {
		slot := &s.slots[i]
		if !slot.occupied {
			continue
		}
		distance := uint16(slot.sequenceNumber - s.nextSequenceNumber)
		if !ok || distance < bestDistance {
			best = slot.sequenceNumber
			bestDistance = distance
			ok = true
		}
	}
	return best, ok
}

type rtpPacketSequencerMissingSummary struct {
	latestDistance uint16
	missingCount   int
	nackPairCount  int
}

func (s *RTPPacketSequencer) missingSummary() rtpPacketSequencerMissingSummary {
	if s == nil || !s.initialized || s.buffered == 0 || len(s.slots) == 0 {
		return rtpPacketSequencerMissingSummary{}
	}
	latestDistance, ok := s.latestBufferedDistance()
	if !ok || latestDistance == 0 {
		return rtpPacketSequencerMissingSummary{}
	}
	var summary rtpPacketSequencerMissingSummary
	summary.latestDistance = latestDistance
	var pairPacketID uint16
	havePair := false
	for distance := uint16(0); distance < latestDistance; distance++ {
		seq := s.nextSequenceNumber + distance
		if s.hasBufferedSequence(seq) {
			continue
		}
		summary.missingCount++
		if !havePair || uint16(seq-pairPacketID) > 16 {
			summary.nackPairCount++
			pairPacketID = seq
			havePair = true
		}
	}
	return summary
}

func (s *RTPPacketSequencer) latestBufferedDistance() (uint16, bool) {
	var latest uint16
	ok := false
	for i := range s.slots {
		slot := &s.slots[i]
		if !slot.occupied {
			continue
		}
		distance := uint16(slot.sequenceNumber - s.nextSequenceNumber)
		if !ok || distance > latest {
			latest = distance
			ok = true
		}
	}
	return latest, ok
}

func (s *RTPPacketSequencer) hasBufferedSequence(seq uint16) bool {
	slot := &s.slots[int(seq%uint16(len(s.slots)))]
	return slot.occupied && slot.sequenceNumber == seq
}

func (s *RTPPacketSequencer) clearBuffered() {
	clear(s.slots)
	s.buffered = 0
}

func rtpSequenceNumberAhead(seq uint16, next uint16) bool {
	return int16(seq-next) > 0
}
