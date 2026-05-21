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
	off   int
	index uint32
}

func NewTemporalUnitIterator(src []byte) TemporalUnitIterator {
	return TemporalUnitIterator{src: src}
}

func (it *TemporalUnitIterator) Next() (TemporalUnit, bool, error) {
	if it.off == len(it.src) {
		return TemporalUnit{}, false, nil
	}
	if it.off > len(it.src) {
		return TemporalUnit{}, false, ErrShortPayload
	}

	start := it.off
	first := true
	for it.off < len(it.src) {
		unit, consumed, err := ParseLowOverhead(it.src[it.off:])
		if err != nil {
			return TemporalUnit{}, false, err
		}
		if first {
			if unit.Header.Type != TypeTemporalDelimiter {
				return TemporalUnit{}, false, ErrMissingTemporalDelimiter
			}
			first = false
			it.off += consumed
			continue
		}
		if unit.Header.Type == TypeTemporalDelimiter {
			result := TemporalUnit{Raw: it.src[start:it.off], Index: it.index}
			it.index++
			return result, true, nil
		}
		it.off += consumed
	}

	result := TemporalUnit{Raw: it.src[start:it.off], Index: it.index}
	it.index++
	return result, true, nil
}
