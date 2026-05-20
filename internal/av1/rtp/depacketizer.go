package rtp

// OBUSpan points at one complete OBU in a caller-owned output buffer.
type OBUSpan struct {
	Offset int
	Length int

	Fragmented  bool
	NewSequence bool
}

// Depacketizer incrementally rebuilds AV1 RTP payloads into complete OBUs.
//
// It owns no packet or OBU memory. Callers pass the current output length and a
// caller-owned span array on every Push call. If Push returns while a fragment is
// open, the caller must pass the returned output length back into the next Push
// call with the same output buffer contents preserved.
type Depacketizer struct {
	inFragment    bool
	fragmentStart int
}

func (d *Depacketizer) Reset() {
	d.inFragment = false
	d.fragmentStart = 0
}

func (d *Depacketizer) InFragment() bool {
	return d.inFragment
}

// Push depacketizes one AV1 RTP payload.
func (d *Depacketizer) Push(dst []byte, used int, spans []OBUSpan, payload []byte) (newUsed int, spanCount int, header AggregationHeader, err error) {
	if used < 0 || used > len(dst) {
		return used, 0, AggregationHeader{}, ErrShortBuffer
	}

	it, err := NewIterator(payload)
	if err != nil {
		return used, 0, AggregationHeader{}, err
	}
	header = it.Header()
	newUsed = used

	for {
		elem, ok, err := it.Next()
		if err != nil {
			return newUsed, spanCount, header, err
		}
		if !ok {
			return newUsed, spanCount, header, nil
		}

		start := newUsed
		if elem.ContinuesPrevious {
			if !d.inFragment {
				return newUsed, spanCount, header, ErrUnexpectedContinuation
			}
			start = d.fragmentStart
		} else if d.inFragment {
			return newUsed, spanCount, header, ErrFragmentInterrupted
		}

		if len(dst)-newUsed < len(elem.Data) {
			return newUsed, spanCount, header, ErrShortBuffer
		}
		newUsed += copy(dst[newUsed:], elem.Data)

		if elem.ContinuesNext {
			if !d.inFragment {
				d.inFragment = true
				d.fragmentStart = start
			}
			continue
		}

		fragmented := d.inFragment || elem.ContinuesPrevious
		if d.inFragment {
			d.inFragment = false
		}
		if spanCount >= len(spans) {
			return newUsed, spanCount, header, ErrShortBuffer
		}
		spans[spanCount] = OBUSpan{
			Offset:      start,
			Length:      newUsed - start,
			Fragmented:  fragmented,
			NewSequence: header.StartsNewCodedVideoSequence && elem.Index == 0,
		}
		spanCount++
	}
}
