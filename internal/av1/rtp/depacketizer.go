// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

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

// PushSize validates one AV1 RTP payload against the current depacketizer state
// and reports the output bytes and OBU span slots Push would need. It does not
// mutate d, so callers can size their dst and spans scratch before calling Push.
func (d Depacketizer) PushSize(used int, payload []byte) (newUsed int, spanCount int, header AggregationHeader, err error) {
	return (&d).push(nil, used, nil, payload, true)
}

// Push depacketizes one AV1 RTP payload.
func (d *Depacketizer) Push(dst []byte, used int, spans []OBUSpan, payload []byte) (newUsed int, spanCount int, header AggregationHeader, err error) {
	return d.push(dst, used, spans, payload, false)
}

func (d *Depacketizer) push(dst []byte, used int, spans []OBUSpan, payload []byte, sizing bool) (newUsed int, spanCount int, header AggregationHeader, err error) {
	if used < 0 || (!sizing && used > len(dst)) {
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

		if !sizing && len(dst)-newUsed < len(elem.Data) {
			return newUsed, spanCount, header, ErrShortBuffer
		}
		if sizing {
			maxInt := int(^uint(0) >> 1)
			if newUsed > maxInt-len(elem.Data) {
				return newUsed, spanCount, header, ErrShortBuffer
			}
			newUsed += len(elem.Data)
		} else {
			newUsed += copy(dst[newUsed:], elem.Data)
		}

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
		if !sizing && spanCount >= len(spans) {
			return newUsed, spanCount, header, ErrShortBuffer
		}
		if !sizing {
			spans[spanCount] = OBUSpan{
				Offset:      start,
				Length:      newUsed - start,
				Fragmented:  fragmented,
				NewSequence: header.StartsNewCodedVideoSequence && elem.Index == 0,
			}
		}
		spanCount++
	}
}
