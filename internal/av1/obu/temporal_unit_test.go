package obu

import (
	"errors"
	"testing"
)

func TestTemporalUnitIterator(t *testing.T) {
	var stream []byte
	firstStart := len(stream)
	stream = appendLowOverheadTestOBU(stream, TypeTemporalDelimiter, nil)
	stream = appendLowOverheadTestOBU(stream, TypeSequenceHeader, []byte{0xaa})
	stream = appendLowOverheadTestOBU(stream, TypeFrameHeader, []byte{0xbb})
	firstEnd := len(stream)
	stream = appendLowOverheadTestOBU(stream, TypeTemporalDelimiter, nil)
	stream = appendLowOverheadTestOBU(stream, TypeFrame, []byte{0xcc})
	secondEnd := len(stream)

	it := NewTemporalUnitIterator(stream)
	first, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("first ok=%v err=%v", ok, err)
	}
	if first.Index != 0 || string(first.Raw) != string(stream[firstStart:firstEnd]) {
		t.Fatalf("first=%+v want=%x", first, stream[firstStart:firstEnd])
	}
	second, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("second ok=%v err=%v", ok, err)
	}
	if second.Index != 1 || string(second.Raw) != string(stream[firstEnd:secondEnd]) {
		t.Fatalf("second=%+v want=%x", second, stream[firstEnd:secondEnd])
	}
	_, ok, err = it.Next()
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("unexpected third temporal unit")
	}
}

func TestTemporalUnitIteratorRequiresDelimiter(t *testing.T) {
	stream := appendLowOverheadTestOBU(nil, TypeSequenceHeader, []byte{0xaa})
	it := NewTemporalUnitIterator(stream)
	_, _, err := it.Next()
	if !errors.Is(err, ErrMissingTemporalDelimiter) {
		t.Fatalf("err=%v want %v", err, ErrMissingTemporalDelimiter)
	}
}

func TestTemporalUnitIteratorAllocs(t *testing.T) {
	stream := appendLowOverheadTestOBU(nil, TypeTemporalDelimiter, nil)
	stream = appendLowOverheadTestOBU(stream, TypeFrame, []byte{0xaa})
	allocs := testing.AllocsPerRun(1000, func() {
		it := NewTemporalUnitIterator(stream)
		unit, ok, err := it.Next()
		if err != nil || !ok || len(unit.Raw) == 0 {
			t.Fatalf("unit=%+v ok=%v err=%v", unit, ok, err)
		}
	})
	if allocs != 0 {
		t.Fatalf("TemporalUnitIterator allocated: %f", allocs)
	}
}

func FuzzTemporalUnitIterator(f *testing.F) {
	stream := appendLowOverheadTestOBU(nil, TypeTemporalDelimiter, nil)
	stream = appendLowOverheadTestOBU(stream, TypeFrame, []byte{0xaa})
	f.Add(stream)
	f.Fuzz(func(t *testing.T, src []byte) {
		if len(src) > 4096 {
			src = src[:4096]
		}
		it := NewTemporalUnitIterator(src)
		for i := 0; i < 64; i++ {
			unit, ok, err := it.Next()
			if err != nil {
				return
			}
			if !ok {
				return
			}
			if len(unit.Raw) > len(src) {
				t.Fatalf("unit length=%d src=%d", len(unit.Raw), len(src))
			}
		}
	})
}

func appendLowOverheadTestOBU(dst []byte, typ Type, payload []byte) []byte {
	dst = append(dst, byte(typ)<<3|0x02)
	size := uint32(len(payload))
	for {
		b := byte(size & 0x7f)
		size >>= 7
		if size != 0 {
			b |= 0x80
		}
		dst = append(dst, b)
		if size == 0 {
			break
		}
	}
	return append(dst, payload...)
}
