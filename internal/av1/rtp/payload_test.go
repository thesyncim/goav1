package rtp

import (
	"errors"
	"testing"
)

func TestAggregationHeaderRoundTrip(t *testing.T) {
	header := AggregationHeader{
		ContinuesNext:               true,
		ElementCount:                2,
		StartsNewCodedVideoSequence: true,
	}

	var buf [1]byte
	n, err := PutAggregationHeader(buf[:], header)
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("wrote=%d want 1", n)
	}

	got, consumed, err := ParseAggregationHeader(buf[:])
	if err != nil {
		t.Fatal(err)
	}
	if consumed != 1 || got != header {
		t.Fatalf("got=%+v consumed=%d want=%+v", got, consumed, header)
	}
}

func TestAggregationHeaderRejectsInvalid(t *testing.T) {
	tests := []struct {
		name string
		in   []byte
		err  error
	}{
		{name: "reserved", in: []byte{0x01}, err: ErrReservedBit},
		{name: "n and z", in: []byte{0x88}, err: ErrInvalidAggregationHeader},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := ParseAggregationHeader(tt.in)
			if !errors.Is(err, tt.err) {
				t.Fatalf("ParseAggregationHeader err=%v want %v", err, tt.err)
			}
		})
	}
}

func TestPayloadIteratorLengthDelimited(t *testing.T) {
	payload := []byte{
		0x00,
		0x02, 0xaa, 0xbb,
		0x01, 0xcc,
	}

	it, err := NewIterator(payload)
	if err != nil {
		t.Fatal(err)
	}

	elem, ok, err := it.Next()
	if err != nil {
		t.Fatal(err)
	}
	if !ok || elem.Index != 0 || string(elem.Data) != string([]byte{0xaa, 0xbb}) {
		t.Fatalf("elem=%+v ok=%v", elem, ok)
	}

	elem, ok, err = it.Next()
	if err != nil {
		t.Fatal(err)
	}
	if !ok || elem.Index != 1 || string(elem.Data) != string([]byte{0xcc}) {
		t.Fatalf("elem=%+v ok=%v", elem, ok)
	}

	_, ok, err = it.Next()
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("unexpected third element")
	}
}

func TestPayloadIteratorInferredLastLength(t *testing.T) {
	payload := []byte{
		0x20,
		0x02, 0xaa, 0xbb,
		0xcc, 0xdd,
	}

	it, err := NewIterator(payload)
	if err != nil {
		t.Fatal(err)
	}

	elem, ok, err := it.Next()
	if err != nil || !ok || elem.InferredLastLength {
		t.Fatalf("first elem=%+v ok=%v err=%v", elem, ok, err)
	}

	elem, ok, err = it.Next()
	if err != nil || !ok || !elem.InferredLastLength || string(elem.Data) != string([]byte{0xcc, 0xdd}) {
		t.Fatalf("second elem=%+v ok=%v err=%v", elem, ok, err)
	}
}

func TestPayloadIteratorHeaderOnlyFragment(t *testing.T) {
	it, err := NewIterator([]byte{0x50}) // Y=1, W=1.
	if err != nil {
		t.Fatal(err)
	}
	elem, ok, err := it.Next()
	if err != nil {
		t.Fatal(err)
	}
	if !ok || len(elem.Data) != 0 || elem.Index != 0 || !elem.ContinuesNext || !elem.InferredLastLength {
		t.Fatalf("elem=%+v ok=%v", elem, ok)
	}
	_, ok, err = it.Next()
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("unexpected second element")
	}
}

func TestPayloadIteratorRejectsHeaderOnlyCompleteOBU(t *testing.T) {
	it, err := NewIterator([]byte{0x10}) // W=1, but neither Z nor Y.
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = it.Next()
	if !errors.Is(err, ErrZeroLengthElement) {
		t.Fatalf("Next err=%v want %v", err, ErrZeroLengthElement)
	}
}

func TestPayloadIteratorManyElementsNoIndexWrap(t *testing.T) {
	var payload [1 + 300*2]byte
	payload[0] = 0x00
	off := 1
	for i := 0; i < 300; i++ {
		payload[off] = 0x01
		payload[off+1] = byte(i)
		off += 2
	}

	it, err := NewIterator(payload[:])
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 300; i++ {
		elem, ok, err := it.Next()
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			t.Fatalf("iterator ended at %d", i)
		}
		if elem.Index != i {
			t.Fatalf("index=%d want %d", elem.Index, i)
		}
	}

	_, ok, err := it.Next()
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("unexpected extra element")
	}
}

func TestPutPayload(t *testing.T) {
	elements := []Element{
		{Data: []byte{0xaa}},
		{Data: []byte{0xbb, 0xcc}},
	}
	header := AggregationHeader{ElementCount: 2}
	var dst [16]byte
	n, err := PutPayload(dst[:], header, elements)
	if err != nil {
		t.Fatal(err)
	}

	it, err := NewIterator(dst[:n])
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < len(elements); i++ {
		elem, ok, err := it.Next()
		if err != nil {
			t.Fatal(err)
		}
		if !ok || string(elem.Data) != string(elements[i].Data) {
			t.Fatalf("element %d got=%+v ok=%v", i, elem, ok)
		}
	}
}

func TestPutPayloadAllowsEmptyFragment(t *testing.T) {
	var dst [1]byte
	n, err := PutPayload(dst[:], AggregationHeader{
		ContinuesNext: true,
		ElementCount:  1,
	}, []Element{{}})
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 || dst[0] != 0x50 {
		t.Fatalf("payload=%x n=%d want 50", dst[:n], n)
	}
}

func TestFragmentReassembler(t *testing.T) {
	obu := []byte{0x18, 0xaa, 0xbb, 0xcc, 0xdd}
	var packet [4]byte
	var out [16]byte
	var reassembler FragmentReassembler

	n, next, more, err := PutFragment(packet[:], obu, 0, len(packet), true)
	if err != nil {
		t.Fatal(err)
	}
	if !more {
		t.Fatal("expected more fragments")
	}
	it, err := NewIterator(packet[:n])
	if err != nil {
		t.Fatal(err)
	}
	elem, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("first fragment ok=%v err=%v", ok, err)
	}
	if _, complete, err := reassembler.Push(out[:], elem); err != nil || complete {
		t.Fatalf("first push complete=%v err=%v", complete, err)
	}

	n, _, more, err = PutFragment(packet[:], obu, next, len(packet), true)
	if err != nil {
		t.Fatal(err)
	}
	if more {
		t.Fatal("did not expect more fragments")
	}
	it, err = NewIterator(packet[:n])
	if err != nil {
		t.Fatal(err)
	}
	elem, ok, err = it.Next()
	if err != nil || !ok {
		t.Fatalf("second fragment ok=%v err=%v", ok, err)
	}
	got, complete, err := reassembler.Push(out[:], elem)
	if err != nil || !complete {
		t.Fatalf("second push complete=%v err=%v", complete, err)
	}
	if string(out[:got]) != string(obu) {
		t.Fatalf("reassembled=%x want=%x", out[:got], obu)
	}
}

func TestRTPAllocs(t *testing.T) {
	elements := []Element{{Data: []byte{0x18, 0xaa}}, {Data: []byte{0x20, 0xbb}}}
	var dst [32]byte

	allocs := testing.AllocsPerRun(1000, func() {
		n, err := PutPayload(dst[:], AggregationHeader{ElementCount: 2}, elements)
		if err != nil {
			t.Fatal(err)
		}
		it, err := NewIterator(dst[:n])
		if err != nil {
			t.Fatal(err)
		}
		for {
			_, ok, err := it.Next()
			if err != nil {
				t.Fatal(err)
			}
			if !ok {
				break
			}
		}
	})
	if allocs != 0 {
		t.Fatalf("RTP payload hot path allocated: %f", allocs)
	}
}

func BenchmarkPayloadIterator(b *testing.B) {
	payload := []byte{0x20, 0x02, 0xaa, 0xbb, 0xcc, 0xdd}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		it, _ := NewIterator(payload)
		for {
			_, ok, _ := it.Next()
			if !ok {
				break
			}
		}
	}
}
