package rtp

import (
	"errors"
	"testing"
)

func TestDepacketizerCompleteElements(t *testing.T) {
	elements := []Element{
		{Data: []byte{0x18, 0xaa}},
		{Data: []byte{0x20, 0xbb, 0xcc}},
	}
	var payload [32]byte
	n, err := PutPayload(payload[:], AggregationHeader{
		ElementCount:                2,
		StartsNewCodedVideoSequence: true,
	}, elements)
	if err != nil {
		t.Fatal(err)
	}

	var dep Depacketizer
	var out [32]byte
	var spans [4]OBUSpan
	used, count, header, err := dep.Push(out[:], 0, spans[:], payload[:n])
	if err != nil {
		t.Fatal(err)
	}
	if used != 5 || count != 2 || !header.StartsNewCodedVideoSequence {
		t.Fatalf("used=%d count=%d header=%+v", used, count, header)
	}
	if spans[0] != (OBUSpan{Offset: 0, Length: 2, NewSequence: true}) {
		t.Fatalf("span0=%+v", spans[0])
	}
	if spans[1] != (OBUSpan{Offset: 2, Length: 3}) {
		t.Fatalf("span1=%+v", spans[1])
	}
	if string(out[:used]) != string([]byte{0x18, 0xaa, 0x20, 0xbb, 0xcc}) {
		t.Fatalf("out=%x", out[:used])
	}
}

func TestDepacketizerPushSizeMatchesPush(t *testing.T) {
	elements := []Element{
		{Data: []byte{0x18, 0xaa}},
		{Data: []byte{0x20, 0xbb, 0xcc}},
	}
	var payload [32]byte
	n, err := PutPayload(payload[:], AggregationHeader{
		ElementCount:                2,
		StartsNewCodedVideoSequence: true,
	}, elements)
	if err != nil {
		t.Fatal(err)
	}

	var dep Depacketizer
	plannedUsed, plannedSpans, plannedHeader, err := dep.PushSize(3, payload[:n])
	if err != nil {
		t.Fatal(err)
	}
	if plannedUsed != 8 || plannedSpans != 2 || !plannedHeader.StartsNewCodedVideoSequence {
		t.Fatalf("planned used=%d spans=%d header=%+v", plannedUsed, plannedSpans, plannedHeader)
	}
	if dep.InFragment() {
		t.Fatal("PushSize mutated depacketizer state")
	}

	var out [16]byte
	copy(out[:3], []byte{0xf0, 0xf1, 0xf2})
	var spans [2]OBUSpan
	used, count, header, err := dep.Push(out[:plannedUsed], 3, spans[:plannedSpans], payload[:n])
	if err != nil {
		t.Fatal(err)
	}
	if used != plannedUsed || count != plannedSpans || header != plannedHeader {
		t.Fatalf("push used=%d count=%d header=%+v planned %d,%d,%+v", used, count, header, plannedUsed, plannedSpans, plannedHeader)
	}
	if spans[0] != (OBUSpan{Offset: 3, Length: 2, NewSequence: true}) {
		t.Fatalf("span0=%+v", spans[0])
	}
	if spans[1] != (OBUSpan{Offset: 5, Length: 3}) {
		t.Fatalf("span1=%+v", spans[1])
	}
	if string(out[:used]) != string([]byte{0xf0, 0xf1, 0xf2, 0x18, 0xaa, 0x20, 0xbb, 0xcc}) {
		t.Fatalf("out=%x", out[:used])
	}
}

func TestDepacketizerFragmentedElement(t *testing.T) {
	obu := []byte{0x18, 0xaa, 0xbb, 0xcc, 0xdd}
	var packet [4]byte
	var dep Depacketizer
	var out [16]byte
	var spans [2]OBUSpan

	n, next, more, err := PutFragment(packet[:], obu, 0, len(packet), true)
	if err != nil {
		t.Fatal(err)
	}
	if !more {
		t.Fatal("expected more fragments")
	}
	used, count, _, err := dep.Push(out[:], 0, spans[:], packet[:n])
	if err != nil {
		t.Fatal(err)
	}
	if count != 0 || !dep.InFragment() {
		t.Fatalf("count=%d inFragment=%v", count, dep.InFragment())
	}

	n, _, more, err = PutFragment(packet[:], obu, next, len(packet), true)
	if err != nil {
		t.Fatal(err)
	}
	if more {
		t.Fatal("unexpected extra fragment")
	}
	used, count, _, err = dep.Push(out[:], used, spans[:], packet[:n])
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 || dep.InFragment() {
		t.Fatalf("count=%d inFragment=%v", count, dep.InFragment())
	}
	if spans[0] != (OBUSpan{Offset: 0, Length: len(obu), Fragmented: true}) {
		t.Fatalf("span=%+v", spans[0])
	}
	if string(out[:used]) != string(obu) {
		t.Fatalf("out=%x want=%x", out[:used], obu)
	}
}

func TestDepacketizerPushSizePreservesFragmentState(t *testing.T) {
	obu := []byte{0x18, 0xaa, 0xbb, 0xcc, 0xdd}
	var packet [4]byte
	var dep Depacketizer
	var out [16]byte
	var spans [2]OBUSpan

	n, next, more, err := PutFragment(packet[:], obu, 0, len(packet), true)
	if err != nil {
		t.Fatal(err)
	}
	if !more {
		t.Fatal("expected more fragments")
	}
	plannedUsed, plannedSpans, _, err := dep.PushSize(0, packet[:n])
	if err != nil {
		t.Fatal(err)
	}
	if plannedUsed != 3 || plannedSpans != 0 || dep.InFragment() {
		t.Fatalf("start plan used=%d spans=%d inFragment=%v", plannedUsed, plannedSpans, dep.InFragment())
	}
	used, count, _, err := dep.Push(out[:plannedUsed], 0, spans[:], packet[:n])
	if err != nil {
		t.Fatal(err)
	}
	if used != plannedUsed || count != plannedSpans || !dep.InFragment() {
		t.Fatalf("start push used=%d count=%d inFragment=%v", used, count, dep.InFragment())
	}

	n, _, more, err = PutFragment(packet[:], obu, next, len(packet), true)
	if err != nil {
		t.Fatal(err)
	}
	if more {
		t.Fatal("unexpected extra fragment")
	}
	plannedUsed, plannedSpans, _, err = dep.PushSize(used, packet[:n])
	if err != nil {
		t.Fatal(err)
	}
	if plannedUsed != len(obu) || plannedSpans != 1 || !dep.InFragment() {
		t.Fatalf("end plan used=%d spans=%d inFragment=%v", plannedUsed, plannedSpans, dep.InFragment())
	}
	used, count, _, err = dep.Push(out[:plannedUsed], used, spans[:plannedSpans], packet[:n])
	if err != nil {
		t.Fatal(err)
	}
	if used != plannedUsed || count != plannedSpans || dep.InFragment() {
		t.Fatalf("end push used=%d count=%d inFragment=%v", used, count, dep.InFragment())
	}
}

func TestDepacketizerHeaderOnlyFragmentStart(t *testing.T) {
	var dep Depacketizer
	var out [8]byte
	var spans [2]OBUSpan

	used, count, _, err := dep.Push(out[:], 0, spans[:], []byte{0x50}) // Y=1, W=1.
	if err != nil {
		t.Fatal(err)
	}
	if used != 0 || count != 0 || !dep.InFragment() {
		t.Fatalf("start used=%d count=%d inFragment=%v", used, count, dep.InFragment())
	}

	used, count, _, err = dep.Push(out[:], used, spans[:], []byte{0x90, 0x18, 0xaa}) // Z=1, W=1.
	if err != nil {
		t.Fatal(err)
	}
	if used != 2 || count != 1 || dep.InFragment() {
		t.Fatalf("end used=%d count=%d inFragment=%v", used, count, dep.InFragment())
	}
	if spans[0] != (OBUSpan{Offset: 0, Length: 2, Fragmented: true}) {
		t.Fatalf("span=%+v", spans[0])
	}
	if string(out[:used]) != string([]byte{0x18, 0xaa}) {
		t.Fatalf("out=%x", out[:used])
	}
}

func TestDepacketizerHeaderOnlyFragmentEnd(t *testing.T) {
	var dep Depacketizer
	var out [8]byte
	var spans [2]OBUSpan

	used, count, _, err := dep.Push(out[:], 0, spans[:], []byte{0x50, 0x18, 0xaa}) // Y=1, W=1.
	if err != nil {
		t.Fatal(err)
	}
	if used != 2 || count != 0 || !dep.InFragment() {
		t.Fatalf("start used=%d count=%d inFragment=%v", used, count, dep.InFragment())
	}

	used, count, _, err = dep.Push(out[:], used, spans[:], []byte{0x90}) // Z=1, W=1.
	if err != nil {
		t.Fatal(err)
	}
	if used != 2 || count != 1 || dep.InFragment() {
		t.Fatalf("end used=%d count=%d inFragment=%v", used, count, dep.InFragment())
	}
	if spans[0] != (OBUSpan{Offset: 0, Length: 2, Fragmented: true}) {
		t.Fatalf("span=%+v", spans[0])
	}
	if string(out[:used]) != string([]byte{0x18, 0xaa}) {
		t.Fatalf("out=%x", out[:used])
	}
}

func TestDepacketizerPushSizeRejectsStateErrors(t *testing.T) {
	var dep Depacketizer
	_, _, _, err := dep.PushSize(0, []byte{0x90, 0x18, 0xaa}) // Z=1, W=1.
	if !errors.Is(err, ErrUnexpectedContinuation) {
		t.Fatalf("PushSize err=%v want %v", err, ErrUnexpectedContinuation)
	}
	if dep.InFragment() {
		t.Fatal("PushSize mutated depacketizer state")
	}

	_, _, _, err = dep.PushSize(-1, []byte{0x10, 0x18})
	if !errors.Is(err, ErrShortBuffer) {
		t.Fatalf("PushSize negative used err=%v want %v", err, ErrShortBuffer)
	}
}

func TestDepacketizerAllocs(t *testing.T) {
	elements := []Element{{Data: []byte{0x18, 0xaa}}, {Data: []byte{0x20, 0xbb}}}
	var payload [32]byte
	n, err := PutPayload(payload[:], AggregationHeader{ElementCount: 2}, elements)
	if err != nil {
		t.Fatal(err)
	}

	var out [32]byte
	var spans [4]OBUSpan
	allocs := testing.AllocsPerRun(1000, func() {
		var dep Depacketizer
		_, _, _, err := dep.Push(out[:], 0, spans[:], payload[:n])
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Depacketizer allocated: %f", allocs)
	}
}

func TestDepacketizerPushSizeAllocs(t *testing.T) {
	elements := []Element{{Data: []byte{0x18, 0xaa}}, {Data: []byte{0x20, 0xbb}}}
	var payload [32]byte
	n, err := PutPayload(payload[:], AggregationHeader{ElementCount: 2}, elements)
	if err != nil {
		t.Fatal(err)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		var dep Depacketizer
		used, count, _, err := dep.PushSize(0, payload[:n])
		if err != nil {
			t.Fatal(err)
		}
		if used != 4 || count != 2 {
			t.Fatalf("used=%d count=%d", used, count)
		}
	})
	if allocs != 0 {
		t.Fatalf("Depacketizer.PushSize allocated: %f", allocs)
	}
}

func BenchmarkDepacketizer(b *testing.B) {
	elements := []Element{{Data: []byte{0x18, 0xaa}}, {Data: []byte{0x20, 0xbb}}}
	var payload [32]byte
	n, err := PutPayload(payload[:], AggregationHeader{ElementCount: 2}, elements)
	if err != nil {
		b.Fatal(err)
	}

	var out [32]byte
	var spans [4]OBUSpan
	b.ReportAllocs()
	for b.Loop() {
		var dep Depacketizer
		_, _, _, _ = dep.Push(out[:], 0, spans[:], payload[:n])
	}
}

func BenchmarkDepacketizerPushSize(b *testing.B) {
	elements := []Element{{Data: []byte{0x18, 0xaa}}, {Data: []byte{0x20, 0xbb}}}
	var payload [32]byte
	n, err := PutPayload(payload[:], AggregationHeader{ElementCount: 2}, elements)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	for b.Loop() {
		var dep Depacketizer
		_, _, _, _ = dep.PushSize(0, payload[:n])
	}
}
