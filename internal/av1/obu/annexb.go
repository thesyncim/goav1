package obu

import "github.com/thesyncim/goav1/internal/av1/bitstream"

// AnnexBUnit is one OBU unit from an AV1 Annex B stream. Raw aliases the
// caller-provided stream bytes.
type AnnexBUnit struct {
	Raw []byte
	OBU Unit

	TemporalUnitIndex uint32
	FrameUnitIndex    uint32
	OBUIndex          uint32
	Offset            uint32
}

// AnnexBIterator iterates the nested Annex B stream structure without
// allocation: temporal_unit_size, frame_unit_size, obu_unit_size, then OBU.
type AnnexBIterator struct {
	src []byte
	off int

	temporalRemain int
	frameRemain    int

	temporalIndex uint32
	frameIndex    uint32
	obuIndex      uint32

	currentTemporal uint32
	currentFrame    uint32
}

func NewAnnexBIterator(src []byte) AnnexBIterator {
	return AnnexBIterator{src: src}
}

// ParseAnnexBElement parses one Annex B size-prefixed OBU unit. The returned
// Unit aliases src after the leb128 obu_unit_size prefix.
func ParseAnnexBElement(src []byte) (Unit, int, error) {
	size, n, err := readAnnexBSize(src)
	if err != nil {
		return Unit{}, 0, err
	}
	if size == 0 {
		return Unit{}, 0, ErrInvalidAnnexB
	}
	start := n
	end := start + size
	if end < start || end > len(src) {
		return Unit{}, 0, ErrShortPayload
	}
	raw := src[start:end]
	unit, err := ParseElement(raw)
	if err != nil {
		return Unit{}, 0, err
	}
	if len(unit.Raw) != len(raw) {
		return Unit{}, 0, ErrSizeMismatch
	}
	return unit, end, nil
}

func (it *AnnexBIterator) Next() (AnnexBUnit, bool, error) {
	if it.off == len(it.src) {
		if it.temporalRemain != 0 || it.frameRemain != 0 {
			return AnnexBUnit{}, false, ErrShortPayload
		}
		return AnnexBUnit{}, false, nil
	}
	if it.off > len(it.src) {
		return AnnexBUnit{}, false, ErrShortPayload
	}

	if it.temporalRemain == 0 {
		size, n, err := readAnnexBSize(it.src[it.off:])
		if err != nil {
			return AnnexBUnit{}, false, err
		}
		if size == 0 {
			return AnnexBUnit{}, false, ErrInvalidAnnexB
		}
		it.off += n
		it.temporalRemain = size
		it.currentTemporal = it.temporalIndex
		it.temporalIndex++
		it.frameIndex = 0
		it.frameRemain = 0
	}

	if it.frameRemain == 0 {
		size, n, err := readAnnexBSize(it.src[it.off:])
		if err != nil {
			return AnnexBUnit{}, false, err
		}
		if size == 0 || n+size > it.temporalRemain {
			return AnnexBUnit{}, false, ErrInvalidAnnexB
		}
		it.off += n
		it.temporalRemain -= n
		it.frameRemain = size
		it.currentFrame = it.frameIndex
		it.frameIndex++
		it.obuIndex = 0
	}

	size, n, err := readAnnexBSize(it.src[it.off:])
	if err != nil {
		return AnnexBUnit{}, false, err
	}
	if size == 0 || n+size > it.frameRemain {
		return AnnexBUnit{}, false, ErrInvalidAnnexB
	}
	it.off += n
	it.temporalRemain -= n
	it.frameRemain -= n

	start := it.off
	end := start + size
	if end < start || end > len(it.src) {
		return AnnexBUnit{}, false, ErrShortPayload
	}
	raw := it.src[start:end]
	unit, err := ParseElement(raw)
	if err != nil {
		return AnnexBUnit{}, false, err
	}
	if len(unit.Raw) != len(raw) {
		return AnnexBUnit{}, false, ErrSizeMismatch
	}
	result := AnnexBUnit{
		Raw:               raw,
		OBU:               unit,
		TemporalUnitIndex: it.currentTemporal,
		FrameUnitIndex:    it.currentFrame,
		OBUIndex:          it.obuIndex,
		Offset:            uint32(start),
	}
	it.off = end
	it.temporalRemain -= size
	it.frameRemain -= size
	it.obuIndex++
	return result, true, nil
}

func readAnnexBSize(src []byte) (int, int, error) {
	value, n, err := bitstream.ReadLEB128(src)
	if err != nil {
		return 0, 0, err
	}
	maxInt := int(^uint(0) >> 1)
	if value > uint32(maxInt) {
		return 0, 0, ErrInvalidAnnexB
	}
	return int(value), n, nil
}
