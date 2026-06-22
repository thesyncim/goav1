// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package rtp

import (
	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/obu"
)

const maxObusWithOmittedLastSize = 3

type PayloadSizeLimits struct {
	MaxPayloadLen            int
	FirstPacketReductionLen  int
	LastPacketReductionLen   int
	SinglePacketReductionLen int
}

type PacketizerOBU struct {
	Type obu.Type

	Header      obu.Header
	HeaderSize  int
	PayloadSize int

	headerByte    byte
	extensionByte byte
	hasExtension  bool
	payload       []byte

	Size int
}

type PacketPlan struct {
	FirstOBU       int
	NumOBUElements int
	FirstOBUOffset int
	LastOBUSize    int
	PacketSize     int
}

type PacketDependencyDescriptorFlags struct {
	FirstPacketInFrame bool
	LastPacketInFrame  bool
}

// PacketizerScratchSize reports caller-owned scratch slots needed to packetize
// one low-overhead AV1 frame into RTP payloads.
type PacketizerScratchSize struct {
	OBUs    int
	Packets int
	Work    int
}

type Packetizer struct {
	obus    []PacketizerOBU
	packets []PacketPlan

	packetIndex int

	keyFrame           bool
	lastFrameInPicture bool
}

func NewPacketizer(payload []byte, limits PayloadSizeLimits, keyFrame bool, lastFrameInPicture bool, obuScratch []PacketizerOBU, packetScratch []PacketPlan, workScratch []PacketPlan) (Packetizer, error) {
	obuCount, err := ParsePacketizerOBUs(payload, obuScratch)
	if err != nil {
		return Packetizer{}, err
	}
	if obuCount == 0 {
		return Packetizer{
			keyFrame:           keyFrame,
			lastFrameInPicture: lastFrameInPicture,
		}, nil
	}

	packetCount, err := PacketizeOBUs(obuScratch[:obuCount], limits, packetScratch, workScratch)
	if err != nil {
		return Packetizer{}, err
	}
	return Packetizer{
		obus:               obuScratch[:obuCount],
		packets:            packetScratch[:packetCount],
		keyFrame:           keyFrame,
		lastFrameInPicture: lastFrameInPicture,
	}, nil
}

// PacketizerScratchLen reports scratch needed by NewPacketizer. Callers may
// first pass nil or short obuScratch to learn the OBU count, allocate that many
// PacketizerOBU slots, and call again to learn packet-plan and work-plan counts.
func PacketizerScratchLen(payload []byte, limits PayloadSizeLimits, obuScratch []PacketizerOBU) (PacketizerScratchSize, error) {
	obuCount, err := CountPacketizerOBUs(payload)
	size := PacketizerScratchSize{OBUs: obuCount}
	if err != nil {
		return size, err
	}
	if obuCount == 0 || len(obuScratch) < obuCount {
		return size, nil
	}

	parsed, err := ParsePacketizerOBUs(payload, obuScratch[:obuCount])
	if err != nil {
		return size, err
	}
	packetCount, err := PacketizeOBUCount(obuScratch[:parsed], limits)
	if err != nil {
		return size, err
	}
	size.Packets = packetCount
	if packetCount > 1 {
		size.Work = packetCount
	}
	return size, nil
}

func (p *Packetizer) NumPackets() int {
	return len(p.packets) - p.packetIndex
}

// RemainingPacketPayloadSize reports the exact number of RTP payload bytes that
// the remaining NextPacket calls will write.
func (p *Packetizer) RemainingPacketPayloadSize() int {
	size := 0
	for i := p.packetIndex; i < len(p.packets); i++ {
		size += 1 + p.packets[i].PacketSize
	}
	return size
}

// NextPacketSize reports the exact dst length required by the next NextPacket
// call without advancing the packetizer.
func (p *Packetizer) NextPacketSize() (int, bool) {
	if p.packetIndex >= len(p.packets) {
		return 0, false
	}
	return 1 + p.packets[p.packetIndex].PacketSize, true
}

// NextPacketPlan reports the packet plan that will be used by the next
// NextPacket call without advancing the packetizer.
func (p *Packetizer) NextPacketPlan() (PacketPlan, bool) {
	if p.packetIndex >= len(p.packets) {
		return PacketPlan{}, false
	}
	return p.packets[p.packetIndex], true
}

func (p *Packetizer) NextPacketDependencyDescriptorFlags() (PacketDependencyDescriptorFlags, bool) {
	if p.packetIndex >= len(p.packets) {
		return PacketDependencyDescriptorFlags{}, false
	}
	return PacketDependencyDescriptorFlags{
		FirstPacketInFrame: p.packetIndex == 0,
		LastPacketInFrame:  p.packetIndex == len(p.packets)-1,
	}, true
}

func (p *Packetizer) NextPacket(dst []byte) (n int, marker bool, ok bool, err error) {
	if p.packetIndex >= len(p.packets) {
		return 0, false, false, nil
	}

	packet := p.packets[p.packetIndex]
	if len(dst) < 1+packet.PacketSize {
		return 0, false, false, ErrShortBuffer
	}

	header := p.aggregationHeader(packet)
	b, err := header.Byte()
	if err != nil {
		return 0, false, false, err
	}

	dst[0] = b
	off := 1
	obuOffset := packet.FirstOBUOffset

	for i := 0; i < packet.NumOBUElements-1; i++ {
		current := &p.obus[packet.FirstOBU+i]
		fragmentSize := current.Size - obuOffset
		n, err := bitstream.PutLEB128(dst[off:], uint32(fragmentSize))
		if err != nil {
			return 0, false, false, err
		}
		off += n
		wrote, err := writeOBUFragment(dst[off:], current, obuOffset, fragmentSize)
		if err != nil {
			return 0, false, false, err
		}
		off += wrote
		obuOffset = 0
	}

	last := &p.obus[packet.FirstOBU+packet.NumOBUElements-1]
	fragmentSize := packet.LastOBUSize
	if packet.NumOBUElements > maxObusWithOmittedLastSize {
		n, err := bitstream.PutLEB128(dst[off:], uint32(fragmentSize))
		if err != nil {
			return 0, false, false, err
		}
		off += n
	}
	wrote, err := writeOBUFragment(dst[off:], last, obuOffset, fragmentSize)
	if err != nil {
		return 0, false, false, err
	}
	off += wrote
	if off != 1+packet.PacketSize {
		return 0, false, false, ErrShortBuffer
	}

	p.packetIndex++
	marker = p.packetIndex == len(p.packets) && p.lastFrameInPicture
	return off, marker, true, nil
}

func ParsePacketizerOBUs(payload []byte, out []PacketizerOBU) (int, error) {
	return scanPacketizerOBUs(payload, out, false)
}

// CountPacketizerOBUs reports how many outbound OBU elements ParsePacketizerOBUs
// will write after dropping RTP-ineligible temporal delimiter, tile list, and
// padding OBUs. It validates the same OBU framing without requiring scratch.
func CountPacketizerOBUs(payload []byte) (int, error) {
	return scanPacketizerOBUs(payload, nil, true)
}

func scanPacketizerOBUs(payload []byte, out []PacketizerOBU, countOnly bool) (int, error) {
	count := 0
	for off := 0; off < len(payload); {
		unit, err := obu.ParseElement(payload[off:])
		if err != nil {
			return count, err
		}
		if len(unit.Raw) == 0 {
			return count, ErrShortPayload
		}
		off += len(unit.Raw)

		typ := unit.Header.Type
		if typ == obu.TypeTemporalDelimiter || typ == obu.TypeTileList || typ == obu.TypePadding {
			continue
		}
		if !countOnly && count >= len(out) {
			return count, ErrOBUBufferTooSmall
		}

		header := unit.Header
		header.HasSizeField = false
		headerByte, extensionByte := packetizerHeaderBytes(header)
		if !countOnly {
			out[count] = PacketizerOBU{
				Type:          typ,
				Header:        header,
				HeaderSize:    header.HeaderLen(),
				PayloadSize:   len(unit.Payload),
				headerByte:    headerByte,
				extensionByte: extensionByte,
				hasExtension:  header.Extension,
				payload:       unit.Payload,
				Size:          header.HeaderLen() + len(unit.Payload),
			}
		}
		count++
	}
	return count, nil
}

func PacketizeOBUs(obus []PacketizerOBU, limits PayloadSizeLimits, packets []PacketPlan, work []PacketPlan) (int, error) {
	if len(obus) != 0 && len(packets) == 0 {
		return 0, ErrPacketPlanTooSmall
	}
	count, err := packetizeInternal(obus, limits, packets)
	if err != nil || count <= 1 {
		return count, err
	}

	unused := 0
	for i := range count {
		available := limits.MaxPayloadLen - 1
		if i == 0 {
			available -= limits.FirstPacketReductionLen
		} else if i == count-1 {
			available -= limits.LastPacketReductionLen
		}
		if available >= packets[i].PacketSize {
			unused += available - packets[i].PacketSize
		}
	}

	if unused <= count*10 {
		return count, nil
	}
	if len(work) < count {
		return count, ErrPacketPlanTooSmall
	}

	evenLimits := limits
	reduction := unused / count
	if reduction <= 1 || evenLimits.MaxPayloadLen <= reduction {
		return count, nil
	}
	evenLimits.MaxPayloadLen -= reduction - 1
	if evenLimits.MaxPayloadLen-evenLimits.LastPacketReductionLen < 3 ||
		evenLimits.MaxPayloadLen-evenLimits.FirstPacketReductionLen < 3 {
		return count, nil
	}

	evenCount, err := packetizeInternal(obus, evenLimits, work)
	if err != nil {
		return count, nil
	}
	if evenCount != count {
		return count, nil
	}
	copy(packets[:count], work[:evenCount])
	return count, nil
}

// PacketizeOBUCount reports the exact packet-plan count PacketizeOBUs needs for
// obus and limits. Callers can use it after ParsePacketizerOBUs to size both the
// packet plan scratch and PacketizeOBUs' optional work scratch.
func PacketizeOBUCount(obus []PacketizerOBU, limits PayloadSizeLimits) (int, error) {
	return packetizeInternal(obus, limits, nil)
}

func packetizeInternal(obus []PacketizerOBU, limits PayloadSizeLimits, packets []PacketPlan) (int, error) {
	if len(obus) == 0 {
		return 0, nil
	}
	if limits.MaxPayloadLen-limits.LastPacketReductionLen < 3 ||
		limits.MaxPayloadLen-limits.FirstPacketReductionLen < 3 {
		return 0, ErrInvalidPayloadLimits
	}
	writePlans := packets != nil
	if writePlans && len(packets) == 0 {
		return 0, ErrPacketPlanTooSmall
	}

	maxPayload := limits.MaxPayloadLen - 1
	packetCount := 1
	packet := PacketPlan{FirstOBU: 0}
	remaining := maxPayload - limits.FirstPacketReductionLen
	packetHasLayer := false
	packetTemporalID := uint8(0)
	packetSpatialID := uint8(0)

	for obuIndex := range obus {
		isLastOBU := obuIndex == len(obus)-1
		current := &obus[obuIndex]
		currentHasLayer, currentTemporalID, currentSpatialID := packetizerOBULayer(current)

		// Keep unassociated OBUs at the front of a packet and do not mix layer IDs.
		if packet.NumOBUElements > 0 && packetHasLayer &&
			(!currentHasLayer || currentTemporalID != packetTemporalID || currentSpatialID != packetSpatialID) {
			if err := storePacketPlan(packets, packetCount-1, packet); err != nil {
				return packetCount, err
			}
			if writePlans && packetCount >= len(packets) {
				return packetCount, ErrPacketPlanTooSmall
			}
			packet = PacketPlan{FirstOBU: obuIndex}
			packetCount++
			remaining = maxPayload
			packetHasLayer = false
		}

		previousExtra := additionalBytesForPreviousOBUElement(packet)
		minRequired := 1
		if packet.NumOBUElements >= maxObusWithOmittedLastSize {
			minRequired = 2
		}
		if remaining < previousExtra+minRequired {
			if err := storePacketPlan(packets, packetCount-1, packet); err != nil {
				return packetCount, err
			}
			if writePlans && packetCount >= len(packets) {
				return packetCount, ErrPacketPlanTooSmall
			}
			packet = PacketPlan{FirstOBU: obuIndex}
			packetCount++
			remaining = maxPayload
			previousExtra = 0
			packetHasLayer = false
		}

		packet.PacketSize += previousExtra
		remaining -= previousExtra
		previousPacketHasLayer := packetHasLayer
		previousPacketTemporalID := packetTemporalID
		previousPacketSpatialID := packetSpatialID
		packet.NumOBUElements++
		if currentHasLayer {
			packetHasLayer = true
			packetTemporalID = currentTemporalID
			packetSpatialID = currentSpatialID
		}

		mustWriteSize := packet.NumOBUElements > maxObusWithOmittedLastSize
		required := current.Size
		if mustWriteSize {
			required += bitstream.LEB128Len(uint32(current.Size))
		}

		available := remaining
		if isLastOBU {
			if packetCount == 1 {
				available += limits.FirstPacketReductionLen
				available -= limits.SinglePacketReductionLen
			} else {
				available -= limits.LastPacketReductionLen
			}
		}

		if required <= available {
			packet.LastOBUSize = current.Size
			packet.PacketSize += required
			remaining -= required
			continue
		}

		maxFirstFragment := remaining
		if mustWriteSize {
			maxFirstFragment = maxFragmentSize(remaining)
		}
		firstFragmentSize := min(current.Size-1, maxFirstFragment)

		if firstFragmentSize == 0 {
			packet.NumOBUElements--
			packet.PacketSize -= previousExtra
			packetHasLayer = previousPacketHasLayer
			packetTemporalID = previousPacketTemporalID
			packetSpatialID = previousPacketSpatialID
		} else {
			packet.PacketSize += firstFragmentSize
			if mustWriteSize {
				packet.PacketSize += bitstream.LEB128Len(uint32(firstFragmentSize))
			}
			packet.LastOBUSize = firstFragmentSize
		}

		obuOffset := firstFragmentSize
		for obuOffset+maxPayload < current.Size {
			if err := storePacketPlan(packets, packetCount-1, packet); err != nil {
				return packetCount, err
			}
			if writePlans && packetCount >= len(packets) {
				return packetCount, ErrPacketPlanTooSmall
			}
			packet = PacketPlan{
				FirstOBU:       obuIndex,
				NumOBUElements: 1,
				FirstOBUOffset: obuOffset,
				LastOBUSize:    maxPayload,
				PacketSize:     maxPayload,
			}
			packetCount++
			if currentHasLayer {
				packetHasLayer = true
				packetTemporalID = currentTemporalID
				packetSpatialID = currentSpatialID
			} else {
				packetHasLayer = false
			}
			obuOffset += maxPayload
		}

		lastFragmentSize := current.Size - obuOffset
		if isLastOBU && lastFragmentSize > maxPayload-limits.LastPacketReductionLen {
			secondLastFragmentSize := (lastFragmentSize + limits.LastPacketReductionLen) / 2
			if secondLastFragmentSize >= lastFragmentSize {
				secondLastFragmentSize = lastFragmentSize - 1
			}
			lastFragmentSize -= secondLastFragmentSize

			if err := storePacketPlan(packets, packetCount-1, packet); err != nil {
				return packetCount, err
			}
			if writePlans && packetCount >= len(packets) {
				return packetCount, ErrPacketPlanTooSmall
			}
			packet = PacketPlan{
				FirstOBU:       obuIndex,
				NumOBUElements: 1,
				FirstOBUOffset: obuOffset,
				LastOBUSize:    secondLastFragmentSize,
				PacketSize:     secondLastFragmentSize,
			}
			packetCount++
			if currentHasLayer {
				packetHasLayer = true
				packetTemporalID = currentTemporalID
				packetSpatialID = currentSpatialID
			} else {
				packetHasLayer = false
			}
			obuOffset += secondLastFragmentSize
		}

		if err := storePacketPlan(packets, packetCount-1, packet); err != nil {
			return packetCount, err
		}
		if writePlans && packetCount >= len(packets) {
			return packetCount, ErrPacketPlanTooSmall
		}
		packet = PacketPlan{
			FirstOBU:       obuIndex,
			NumOBUElements: 1,
			FirstOBUOffset: obuOffset,
			LastOBUSize:    lastFragmentSize,
			PacketSize:     lastFragmentSize,
		}
		packetCount++
		remaining = maxPayload - lastFragmentSize
		if currentHasLayer {
			packetHasLayer = true
			packetTemporalID = currentTemporalID
			packetSpatialID = currentSpatialID
		} else {
			packetHasLayer = false
		}
	}

	if err := storePacketPlan(packets, packetCount-1, packet); err != nil {
		return packetCount, err
	}
	return packetCount, nil
}

func packetizerOBULayer(current *PacketizerOBU) (hasLayer bool, temporalID uint8, spatialID uint8) {
	if current.hasExtension {
		return true, current.Header.TemporalID, current.Header.SpatialID
	}
	switch current.Type {
	case obu.TypeFrameHeader, obu.TypeFrame, obu.TypeTileGroup, obu.TypeRedundantFrameHeader:
		return true, 0, 0
	default:
		return false, 0, 0
	}
}

func storePacketPlan(packets []PacketPlan, index int, packet PacketPlan) error {
	if packets == nil {
		return nil
	}
	if index < 0 || index >= len(packets) {
		return ErrPacketPlanTooSmall
	}
	packets[index] = packet
	return nil
}

func additionalBytesForPreviousOBUElement(packet PacketPlan) int {
	if packet.PacketSize == 0 || packet.NumOBUElements > maxObusWithOmittedLastSize {
		return 0
	}
	return bitstream.LEB128Len(uint32(packet.LastOBUSize))
}

func maxFragmentSize(remainingBytes int) int {
	if remainingBytes <= 1 {
		return 0
	}
	for i := 1; i <= bitstream.MaxLEB128Bytes; i++ {
		limit := (uint64(1) << uint(7*i)) + uint64(i)
		if uint64(remainingBytes) < limit {
			return remainingBytes - i
		}
	}
	return remainingBytes - bitstream.MaxLEB128Bytes
}

func (p *Packetizer) aggregationHeader(packet PacketPlan) AggregationHeader {
	lastOBU := &p.obus[packet.FirstOBU+packet.NumOBUElements-1]
	lastOBUOffset := 0
	if packet.NumOBUElements == 1 {
		lastOBUOffset = packet.FirstOBUOffset
	}

	header := AggregationHeader{
		ContinuesPrevious: packet.FirstOBUOffset > 0,
		ContinuesNext:     lastOBUOffset+packet.LastOBUSize < lastOBU.Size,
	}
	if packet.NumOBUElements <= maxObusWithOmittedLastSize {
		header.ElementCount = uint8(packet.NumOBUElements)
	}
	if p.keyFrame && p.packetIndex == 0 && len(p.obus) > 0 && p.obus[0].Type == obu.TypeSequenceHeader {
		header.StartsNewCodedVideoSequence = true
	}
	return header
}

func packetizerHeaderBytes(header obu.Header) (byte, byte) {
	b := byte(header.Type) << 3
	if header.Extension {
		b |= 0x04
	}
	ext := (header.TemporalID << 5) | (header.SpatialID << 3)
	return b, ext
}

func writeOBUFragment(dst []byte, current *PacketizerOBU, offset int, size int) (int, error) {
	if offset < 0 || size < 0 || offset+size > current.Size {
		return 0, ErrShortPayload
	}
	if len(dst) < size {
		return 0, ErrShortBuffer
	}

	off := 0
	pos := offset
	remaining := size
	if pos == 0 && remaining > 0 {
		dst[off] = current.headerByte
		off++
		pos++
		remaining--
	}
	if current.hasExtension && pos == 1 && remaining > 0 {
		dst[off] = current.extensionByte
		off++
		pos++
		remaining--
	}
	if remaining > 0 {
		payloadOffset := pos - 1
		if current.hasExtension {
			payloadOffset--
		}
		copy(dst[off:off+remaining], current.payload[payloadOffset:payloadOffset+remaining])
		off += remaining
	}
	return off, nil
}
