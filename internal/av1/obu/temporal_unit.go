// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package obu

// TemporalUnit is one Section 5 low-overhead temporal unit. Raw aliases the
// caller-provided stream bytes.
type TemporalUnit struct {
	Raw   []byte
	Index uint32
}

// TemporalUnitIterator groups AV1 low-overhead OBUs into temporal units. This
// follows dav1d's Section 5 reader behavior: each returned unit must start with
// an OBU temporal delimiter and ends immediately before the next delimiter.
type TemporalUnitIterator struct {
	src   []byte
	off   uint32
	index uint32
}

func NewTemporalUnitIterator(src []byte) TemporalUnitIterator {
	return TemporalUnitIterator{src: src}
}

func (it *TemporalUnitIterator) Next() (TemporalUnit, bool, error) {
	srcLen := len(it.src)
	if uint64(srcLen) > uint64(^uint32(0)) {
		return TemporalUnit{}, false, ErrShortPayload
	}
	off := int(it.off)
	if off == srcLen {
		return TemporalUnit{}, false, nil
	}
	if off > srcLen {
		return TemporalUnit{}, false, ErrShortPayload
	}

	start := off
	first := true
	for off < srcLen {
		unit, consumed, err := ParseLowOverhead(it.src[off:])
		if err != nil {
			return TemporalUnit{}, false, err
		}
		if first {
			if unit.Header.Type != TypeTemporalDelimiter {
				return TemporalUnit{}, false, ErrMissingTemporalDelimiter
			}
			first = false
			off += consumed
			it.off = uint32(off)
			continue
		}
		if unit.Header.Type == TypeTemporalDelimiter {
			result := TemporalUnit{Raw: it.src[start:off], Index: it.index}
			it.index++
			return result, true, nil
		}
		off += consumed
		it.off = uint32(off)
	}

	result := TemporalUnit{Raw: it.src[start:off], Index: it.index}
	it.index++
	return result, true, nil
}
