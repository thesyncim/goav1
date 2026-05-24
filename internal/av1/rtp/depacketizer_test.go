package rtp

import "testing"

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
	for i := 0; i < b.N; i++ {
		var dep Depacketizer
		_, _, _, _ = dep.Push(out[:], 0, spans[:], payload[:n])
	}
}
