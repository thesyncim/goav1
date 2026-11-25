package rtp

import (
	"bytes"
	"errors"
	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/testvector"
	"testing"
)

func TestAssembleFrameSinglePacket(t *testing.T) {
	var frame []byte
	frame = appendPacketizerOBU(frame, obu.TypeSequenceHeader, []byte{0xaa})
	frame = appendPacketizerOBU(frame, obu.TypeFrameHeader, []byte{0xbb, 0xcc})

	var packetizerOBUs [4]PacketizerOBU
	var plans [4]PacketPlan
	var work [4]PacketPlan
	packetizer, err := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 1200}, true, true, packetizerOBUs[:], plans[:], work[:])
	if err != nil {
		t.Fatal(err)
	}

	var packet [32]byte
	n, marker, ok, err := packetizer.NextPacket(packet[:])
	if err != nil {
		t.Fatal(err)
	}
	if !ok || !marker {
		t.Fatalf("ok=%v marker=%v", ok, marker)
	}

	payloads := [][]byte{packet[:n]}
	var out [32]byte
	var obus [4]FrameOBU
	wrote, count, err := AssembleFrame(out[:], payloads, obus[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}
	if string(out[:wrote]) != string(frame) {
		t.Fatalf("assembled=%x want=%x", out[:wrote], frame)
	}
	if obus[0].Header.Type != obu.TypeSequenceHeader ||
		!obus[0].Header.HasSizeField ||
		obus[0].Offset != 0 ||
		obus[0].Length != 3 ||
		obus[0].PrefixSize != 2 ||
		obus[0].PayloadSize != 1 ||
		obus[0].PayloadOffset() != 2 ||
		obus[0].PayloadEnd() != 3 ||
		obus[0].End() != 3 {
		t.Fatalf("obu0=%+v", obus[0])
	}
	if obus[1].Header.Type != obu.TypeFrameHeader ||
		!obus[1].Header.HasSizeField ||
		obus[1].Offset != 3 ||
		obus[1].Length != 4 ||
		obus[1].PrefixSize != 2 ||
		obus[1].PayloadSize != 2 ||
		obus[1].PayloadOffset() != 5 ||
		obus[1].PayloadEnd() != 7 ||
		obus[1].End() != 7 {
		t.Fatalf("obu1=%+v", obus[1])
	}
}

func TestAssembleFrameFragmentedOBU(t *testing.T) {
	payloadBytes := []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	frame := appendPacketizerOBU(nil, obu.TypeFrame, payloadBytes)

	var packetizerOBUs [2]PacketizerOBU
	var plans [8]PacketPlan
	var work [8]PacketPlan
	packetizer, err := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 6}, false, true, packetizerOBUs[:], plans[:], work[:])
	if err != nil {
		t.Fatal(err)
	}

	var packetData [4][16]byte
	var payloadSlots [4][]byte
	payloads := payloadSlots[:0]
	for i := 0; ; i++ {
		n, _, ok, err := packetizer.NextPacket(packetData[i][:])
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			break
		}
		payloads = append(payloads, packetData[i][:n])
	}
	if len(payloads) != 3 {
		t.Fatalf("payload count=%d", len(payloads))
	}

	var out [32]byte
	var obus [2]FrameOBU
	wrote, count, err := AssembleFrame(out[:], payloads, obus[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 {
		t.Fatalf("count=%d", count)
	}
	if string(out[:wrote]) != string(frame) {
		t.Fatalf("assembled=%x want=%x", out[:wrote], frame)
	}
	if obus[0].Header.Type != obu.TypeFrame ||
		!obus[0].Header.HasSizeField ||
		obus[0].PrefixSize != 2 ||
		obus[0].PayloadSize != len(payloadBytes) ||
		obus[0].PayloadOffset() != 2 ||
		obus[0].PayloadEnd() != wrote {
		t.Fatalf("obu=%+v wrote=%d", obus[0], wrote)
	}
}

func TestAssembleFrameSizeMatchesAssembly(t *testing.T) {
	var frame []byte
	frame = appendPacketizerOBU(frame, obu.TypeSequenceHeader, []byte{0xaa})
	frame = appendPacketizerOBU(frame, obu.TypeFrameHeader, []byte{0xbb, 0xcc})
	frame = appendPacketizerOBU(frame, obu.TypeFrame, []byte{0, 1, 2, 3, 4, 5, 6})

	var packetizerOBUs [4]PacketizerOBU
	var plans [8]PacketPlan
	var work [8]PacketPlan
	packetizer, err := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 8}, true, true, packetizerOBUs[:], plans[:], work[:])
	if err != nil {
		t.Fatal(err)
	}

	var packetData [4][16]byte
	var payloadSlots [4][]byte
	payloads := payloadSlots[:0]
	for i := 0; ; i++ {
		n, _, ok, err := packetizer.NextPacket(packetData[i][:])
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			break
		}
		payloads = append(payloads, packetData[i][:n])
	}

	size, count, err := AssembleFrameSize(payloads)
	if err != nil {
		t.Fatal(err)
	}
	if size != len(frame) || count != 3 {
		t.Fatalf("size=%d count=%d want %d,3", size, count, len(frame))
	}

	var out [64]byte
	var obus [4]FrameOBU
	wrote, assembledCount, err := AssembleFrame(out[:], payloads, obus[:])
	if err != nil {
		t.Fatal(err)
	}
	if wrote != size || assembledCount != count || string(out[:wrote]) != string(frame) {
		t.Fatalf("wrote=%d count=%d assembled=%x want size=%d count=%d frame=%x", wrote, assembledCount, out[:wrote], size, count, frame)
	}
}

func TestAssembleFrameSizeRejectsMalformedPayloads(t *testing.T) {
	tests := []struct {
		name    string
		payload [][]byte
		want    error
	}{
		{
			name:    "dangling-fragment",
			payload: [][]byte{{0x50, byte(obu.TypeFrame) << 3, 0xaa}},
			want:    ErrFragmentInterrupted,
		},
		{
			name:    "empty-frame",
			payload: nil,
			want:    ErrEmptyFrame,
		},
		{
			name:    "zero-length-obu",
			payload: [][]byte{{0x10}},
			want:    ErrZeroLengthElement,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := AssembleFrameSize(tt.payload)
			if !errors.Is(err, tt.want) {
				t.Fatalf("AssembleFrameSize err=%v want %v", err, tt.want)
			}
		})
	}
}

func TestAssembleFrameHeaderOnlyStart(t *testing.T) {
	payload0 := []byte{0x50} // Y=1, W=1.
	payload1 := []byte{0x90, byte(obu.TypeFrame) << 3, 0xaa, 0xbb}
	payloads := [][]byte{payload0, payload1}

	var out [16]byte
	var obus [1]FrameOBU
	wrote, count, err := AssembleFrame(out[:], payloads, obus[:])
	if err != nil {
		t.Fatal(err)
	}
	want := appendPacketizerOBU(nil, obu.TypeFrame, []byte{0xaa, 0xbb})
	if count != 1 || string(out[:wrote]) != string(want) {
		t.Fatalf("count=%d assembled=%x want=%x", count, out[:wrote], want)
	}
}

func TestAssembleFrameValidatesExistingSizeField(t *testing.T) {
	payload := []byte{0x10, byte(obu.TypeFrame)<<3 | 0x02, 0x02, 0xaa, 0xbb}
	payloads := [][]byte{payload}

	var out [16]byte
	var obus [1]FrameOBU
	wrote, count, err := AssembleFrame(out[:], payloads, obus[:])
	if err != nil {
		t.Fatal(err)
	}
	want := appendPacketizerOBU(nil, obu.TypeFrame, []byte{0xaa, 0xbb})
	if count != 1 || string(out[:wrote]) != string(want) {
		t.Fatalf("count=%d assembled=%x want=%x", count, out[:wrote], want)
	}

	payload[2] = 0x03
	_, _, err = AssembleFrame(out[:], payloads, obus[:])
	if !errors.Is(err, obu.ErrSizeMismatch) {
		t.Fatalf("err=%v want %v", err, obu.ErrSizeMismatch)
	}
}

func TestAssembleFramePrefixAcrossFragments(t *testing.T) {
	frame := appendPacketizerOBUExt(nil, obu.TypeTileGroup, 1, 2, []byte{0xaa})
	payload0 := []byte{0x50, frame[0]}
	payload1 := append([]byte{0x90}, frame[1:]...)
	payloads := [][]byte{payload0, payload1}

	var out [16]byte
	var obus [1]FrameOBU
	wrote, count, err := AssembleFrame(out[:], payloads, obus[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 || string(out[:wrote]) != string(frame) {
		t.Fatalf("count=%d assembled=%x want=%x", count, out[:wrote], frame)
	}
	if obus[0].Header.Type != obu.TypeTileGroup ||
		!obus[0].Header.Extension ||
		!obus[0].Header.HasSizeField ||
		obus[0].Header.TemporalID != 1 ||
		obus[0].Header.SpatialID != 2 ||
		obus[0].PrefixSize != 3 ||
		obus[0].PayloadOffset() != 3 ||
		obus[0].PayloadEnd() != wrote {
		t.Fatalf("obu=%+v wrote=%d", obus[0], wrote)
	}
}

func TestAssembleFrameRejectsUnexpectedContinuation(t *testing.T) {
	payloads := [][]byte{{0x90, byte(obu.TypeFrame) << 3, 0xaa}}
	var out [16]byte
	var obus [1]FrameOBU
	_, _, err := AssembleFrame(out[:], payloads, obus[:])
	if !errors.Is(err, ErrUnexpectedContinuation) {
		t.Fatalf("err=%v want %v", err, ErrUnexpectedContinuation)
	}
}

func TestAssembleFrameRejectsDanglingFragment(t *testing.T) {
	payloads := [][]byte{{0x50, byte(obu.TypeFrame) << 3, 0xaa}}
	var out [16]byte
	var obus [1]FrameOBU
	_, _, err := AssembleFrame(out[:], payloads, obus[:])
	if !errors.Is(err, ErrFragmentInterrupted) {
		t.Fatalf("err=%v want %v", err, ErrFragmentInterrupted)
	}
}

func TestAssembleFrameRejectsEmptyOBU(t *testing.T) {
	payloads := [][]byte{{0x10}}
	var out [16]byte
	var obus [1]FrameOBU
	_, _, err := AssembleFrame(out[:], payloads, obus[:])
	if !errors.Is(err, ErrZeroLengthElement) {
		t.Fatalf("err=%v want %v", err, ErrZeroLengthElement)
	}
}

func TestAssembleFrameAllocs(t *testing.T) {
	payload := []byte{0x10, byte(obu.TypeFrame) << 3, 0xaa, 0xbb}
	payloads := [][]byte{payload}
	var out [16]byte
	var obus [1]FrameOBU

	allocs := testing.AllocsPerRun(1000, func() {
		_, _, err := AssembleFrame(out[:], payloads, obus[:])
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("AssembleFrame allocated: %f", allocs)
	}
}

func TestAssembleFrameSizeAllocs(t *testing.T) {
	payload := []byte{0x10, byte(obu.TypeFrame) << 3, 0xaa, 0xbb}
	payloads := [][]byte{payload}

	allocs := testing.AllocsPerRun(1000, func() {
		_, _, err := AssembleFrameSize(payloads)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("AssembleFrameSize allocated: %f", allocs)
	}
}

func BenchmarkAssembleFrame(b *testing.B) {
	payload := []byte{0x10, byte(obu.TypeFrame) << 3, 0xaa, 0xbb}
	payloads := [][]byte{payload}
	var out [16]byte
	var obus [1]FrameOBU

	b.ReportAllocs()
	for b.Loop() {
		_, _, _ = AssembleFrame(out[:], payloads, obus[:])
	}
}

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

func TestDependencyDescriptorMandatoryRoundTrip(t *testing.T) {
	want := DependencyDescriptorMandatory{
		FirstPacketInFrame: true,
		LastPacketInFrame:  false,
		TemplateID:         17,
		FrameNumber:        0x1234,
	}
	var buf [DependencyDescriptorMandatorySize]byte
	n, err := PutDependencyDescriptorMandatory(buf[:], want)
	if err != nil {
		t.Fatal(err)
	}
	if n != DependencyDescriptorMandatorySize || buf != [3]byte{0x91, 0x12, 0x34} {
		t.Fatalf("encoded n=%d bytes=% x", n, buf)
	}

	got, consumed, err := ParseDependencyDescriptorMandatory(buf[:])
	if err != nil {
		t.Fatal(err)
	}
	if consumed != DependencyDescriptorMandatorySize || got != want {
		t.Fatalf("parsed=%+v consumed=%d want=%+v", got, consumed, want)
	}
}

func TestDependencyDescriptorMandatoryRejectsInvalid(t *testing.T) {
	if _, err := PutDependencyDescriptorMandatory(make([]byte, 2), DependencyDescriptorMandatory{}); !errors.Is(err, ErrShortBuffer) {
		t.Fatalf("short write err=%v want ErrShortBuffer", err)
	}
	var buf [3]byte
	if _, err := PutDependencyDescriptorMandatory(buf[:], DependencyDescriptorMandatory{TemplateID: 64}); !errors.Is(err, ErrInvalidDependencyDescriptor) {
		t.Fatalf("invalid template err=%v want ErrInvalidDependencyDescriptor", err)
	}
	if _, _, err := ParseDependencyDescriptorMandatory([]byte{0, 1}); !errors.Is(err, ErrShortPayload) {
		t.Fatalf("short parse err=%v want ErrShortPayload", err)
	}
}

func TestDependencyDescriptorMandatoryAllocs(t *testing.T) {
	descriptor := DependencyDescriptorMandatory{
		FirstPacketInFrame: true,
		LastPacketInFrame:  true,
		TemplateID:         3,
		FrameNumber:        9,
	}
	var buf [DependencyDescriptorMandatorySize]byte
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = PutDependencyDescriptorMandatory(buf[:], descriptor)
		_, _, _ = ParseDependencyDescriptorMandatory(buf[:])
	})
	if allocs != 0 {
		t.Fatalf("mandatory descriptor allocs=%f want 0", allocs)
	}
}

func TestDependencyDescriptorParseActiveTargetsAndCustomChains(t *testing.T) {
	structure := DependencyDescriptorStructure{
		StructureID:      5,
		NumDecodeTargets: 2,
		NumChains:        2,
		TemplateNum:      2,
		ResolutionNum:    2,
	}
	structure.DecodeTargetProtectedByChain[0] = 0
	structure.DecodeTargetProtectedByChain[1] = 1
	structure.DecodeTargetSpatialID[0] = 0
	structure.DecodeTargetSpatialID[1] = 1
	structure.Resolutions[0] = DependencyDescriptorResolution{Width: 320, Height: 180}
	structure.Resolutions[1] = DependencyDescriptorResolution{Width: 640, Height: 360}
	structure.Templates[0] = DependencyDescriptorFrameDependencies{
		SpatialID:    0,
		TemporalID:   0,
		DTINum:       2,
		ChainDiffNum: 2,
	}
	structure.Templates[0].DTIs[0] = DependencyDescriptorDecodeTargetSwitch
	structure.Templates[0].DTIs[1] = DependencyDescriptorDecodeTargetRequired
	structure.Templates[0].ChainDiffs[0] = 1
	structure.Templates[0].ChainDiffs[1] = 2
	structure.Templates[1] = DependencyDescriptorFrameDependencies{
		SpatialID:    1,
		TemporalID:   0,
		DTINum:       2,
		ChainDiffNum: 2,
	}
	structure.Templates[1].DTIs[1] = DependencyDescriptorDecodeTargetSwitch
	structure.Templates[1].FrameDiffNum = 1
	structure.Templates[1].FrameDiffs[0] = 1
	structure.Templates[1].ChainDiffs[0] = 3
	structure.Templates[1].ChainDiffs[1] = 4

	var w dependencyDescriptorTestBitWriter
	w.writeBool(true)
	w.writeBool(true)
	w.writeBits(6, 6)
	w.writeBits(200, 16)
	w.writeBool(false) // no attached structure
	w.writeBool(true)  // active decode targets
	w.writeBool(true)  // custom dtis
	w.writeBool(true)  // custom frame diffs
	w.writeBool(true)  // custom chains
	w.writeBits(0x01, 2)
	w.writeBits(uint64(DependencyDescriptorDecodeTargetRequired), 2)
	w.writeBits(uint64(DependencyDescriptorDecodeTargetNotPresent), 2)
	w.writeBits(2, 2)
	w.writeBits(16, 8)
	w.writeBits(0, 2)
	w.writeBits(9, 8)
	w.writeBits(10, 8)

	descriptorBytes := w.bytes()
	got, consumed, err := ParseDependencyDescriptor(descriptorBytes, &structure)
	if err != nil {
		t.Fatalf("ParseDependencyDescriptor: %v descriptor=% x", err, descriptorBytes)
	}
	if consumed != len(descriptorBytes) {
		t.Fatalf("consumed=%d len=%d", consumed, len(descriptorBytes))
	}
	if !got.HasActiveDecodeTargets || got.ActiveDecodeTargetsMask != 0x01 ||
		!got.HasCustomDTIs || !got.HasCustomFrameDiffs || !got.HasCustomChainDiffs {
		t.Fatalf("flags active=%v mask=%#x dtis=%v fdiffs=%v chains=%v", got.HasActiveDecodeTargets, got.ActiveDecodeTargetsMask, got.HasCustomDTIs, got.HasCustomFrameDiffs, got.HasCustomChainDiffs)
	}
	if got.FrameDependencies.SpatialID != 1 || got.FrameDependencies.TemporalID != 0 ||
		got.FrameDependencies.DTIs[0] != DependencyDescriptorDecodeTargetRequired ||
		got.FrameDependencies.DTIs[1] != DependencyDescriptorDecodeTargetNotPresent ||
		got.FrameDependencies.FrameDiffNum != 1 || got.FrameDependencies.FrameDiffs[0] != 17 ||
		got.FrameDependencies.ChainDiffNum != 2 || got.FrameDependencies.ChainDiffs[0] != 9 || got.FrameDependencies.ChainDiffs[1] != 10 {
		t.Fatalf("frame dependencies=%+v", got.FrameDependencies)
	}
	if !got.HasResolution || got.Resolution != (DependencyDescriptorResolution{Width: 640, Height: 360}) {
		t.Fatalf("resolution=%+v has=%v", got.Resolution, got.HasResolution)
	}
}

func TestDependencyDescriptorStateParseRemembersAttachedStructure(t *testing.T) {
	var w dependencyDescriptorTestBitWriter
	w.writeBool(true)
	w.writeBool(true)
	w.writeBits(3, 6)
	w.writeBits(1, 16)
	w.writeBool(true)  // attached structure
	w.writeBool(false) // active decode targets
	w.writeBool(false) // custom dtis
	w.writeBool(false) // custom frame diffs
	w.writeBool(false) // custom chains
	w.writeBits(3, 6)
	w.writeBits(0, 5)
	w.writeBits(3, 2)
	w.writeBits(uint64(DependencyDescriptorDecodeTargetSwitch), 2)
	w.writeBool(false)
	w.writeBits(0, 1)
	w.writeBool(false)

	var state DependencyDescriptorState
	key, consumed, err := state.Parse(w.bytes())
	if err != nil {
		t.Fatalf("state Parse key: %v", err)
	}
	if !state.Valid || consumed != len(w.bytes()) ||
		!state.ActiveDecodeTargetsValid || state.ActiveDecodeTargetsMask != 0x01 ||
		!key.HasAttachedStructure || !key.HasActiveDecodeTargets ||
		key.ActiveDecodeTargetsMask != 0x01 ||
		key.FrameDependencies.DTIs[0] != DependencyDescriptorDecodeTargetSwitch {
		t.Fatalf("key=%+v consumed=%d len=%d state=%+v", key, consumed, len(w.bytes()), state)
	}

	var compact [DependencyDescriptorMandatorySize]byte
	if _, err := PutDependencyDescriptorMandatory(compact[:], DependencyDescriptorMandatory{
		FirstPacketInFrame: true,
		LastPacketInFrame:  true,
		TemplateID:         3,
		FrameNumber:        2,
	}); err != nil {
		t.Fatalf("PutDependencyDescriptorMandatory: %v", err)
	}
	delta, consumed, err := state.Parse(compact[:])
	if err != nil {
		t.Fatalf("state Parse delta: %v", err)
	}
	if consumed != DependencyDescriptorMandatorySize ||
		delta.Mandatory.FrameNumber != 2 ||
		delta.HasActiveDecodeTargets ||
		delta.FrameDependencies.DTIs[0] != DependencyDescriptorDecodeTargetSwitch ||
		!state.ActiveDecodeTargetsValid || state.ActiveDecodeTargetsMask != 0x01 {
		t.Fatalf("delta=%+v consumed=%d", delta, consumed)
	}

	w = dependencyDescriptorTestBitWriter{}
	w.writeBool(true)
	w.writeBool(true)
	w.writeBits(3, 6)
	w.writeBits(3, 16)
	w.writeBool(false) // attached structure
	w.writeBool(true)  // active decode targets
	w.writeBool(false) // custom dtis
	w.writeBool(false) // custom frame diffs
	w.writeBool(false) // custom chains
	w.writeBits(0, 1)
	update, consumed, err := state.Parse(w.bytes())
	if err != nil {
		t.Fatalf("state Parse active update: %v", err)
	}
	if consumed != len(w.bytes()) ||
		!update.HasActiveDecodeTargets || update.ActiveDecodeTargetsMask != 0 ||
		!state.ActiveDecodeTargetsValid || state.ActiveDecodeTargetsMask != 0 {
		t.Fatalf("active update=%+v consumed=%d len=%d state=%+v", update, consumed, len(w.bytes()), state)
	}

	state.Reset()
	if state.Valid || state.ActiveDecodeTargetsValid || state.ActiveDecodeTargetsMask != 0 {
		t.Fatalf("Reset left state valid: %+v", state)
	}
	if _, _, err := state.Parse(compact[:]); !errors.Is(err, ErrInvalidDependencyDescriptor) {
		t.Fatalf("missing reset state err=%v want ErrInvalidDependencyDescriptor", err)
	}
}

func TestDependencyDescriptorRejectsInvalidFull(t *testing.T) {
	if _, _, err := ParseDependencyDescriptor([]byte{0xc0, 0, 1}, nil); !errors.Is(err, ErrInvalidDependencyDescriptor) {
		t.Fatalf("missing structure err=%v want ErrInvalidDependencyDescriptor", err)
	}
	if _, _, err := ParseDependencyDescriptor([]byte{0xc0, 0, 1, 0x80}, nil); !errors.Is(err, ErrShortPayload) {
		t.Fatalf("truncated attached err=%v want ErrShortPayload", err)
	}
	structure := DependencyDescriptorStructure{
		StructureID:      3,
		NumDecodeTargets: 1,
		NumChains:        0,
		TemplateNum:      1,
	}
	structure.DecodeTargetSpatialID[0] = 0
	structure.DecodeTargetTemporalID[0] = 0
	structure.Templates[0] = DependencyDescriptorFrameDependencies{DTINum: 1}
	structure.Templates[0].DTIs[0] = DependencyDescriptorDecodeTargetSwitch
	if _, _, err := ParseDependencyDescriptor([]byte{0xc5, 0, 1}, &structure); !errors.Is(err, ErrInvalidDependencyDescriptor) {
		t.Fatalf("bad template err=%v want ErrInvalidDependencyDescriptor", err)
	}
}

type dependencyDescriptorTestBitWriter struct {
	buf []byte
	bit int
}

func (w *dependencyDescriptorTestBitWriter) writeBool(value bool) {
	if value {
		w.writeBits(1, 1)
		return
	}
	w.writeBits(0, 1)
}

func (w *dependencyDescriptorTestBitWriter) writeBits(value uint64, n uint8) {
	for i := int(n) - 1; i >= 0; i-- {
		byteIndex := w.bit >> 3
		if byteIndex == len(w.buf) {
			w.buf = append(w.buf, 0)
		}
		if value&(1<<uint(i)) != 0 {
			w.buf[byteIndex] |= 1 << uint(7-(w.bit&7))
		}
		w.bit++
	}
}

func (w *dependencyDescriptorTestBitWriter) bytes() []byte {
	return w.buf
}

func FuzzPayloadIterator(f *testing.F) {
	for _, seed := range [][]byte{
		{0x00, 0x01, 0xaa},
		{0x10, 0xaa},
		{0x20, 0x01, 0xaa, 0xbb},
		{0x88, 0xaa},
		{0x01},
	} {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, payload []byte) {
		it, err := NewIterator(payload)
		if err != nil {
			return
		}
		var dep Depacketizer
		plannedUsed, plannedSpans, plannedHeader, err := dep.PushSize(0, payload)
		if err == nil && plannedUsed <= 512 && plannedSpans <= 16 {
			var out [512]byte
			var spans [16]OBUSpan
			used, count, header, err := dep.Push(out[:plannedUsed], 0, spans[:plannedSpans], payload)
			if err != nil {
				t.Fatalf("Push after PushSize: %v", err)
			}
			if used != plannedUsed || count != plannedSpans || header != plannedHeader {
				t.Fatalf("Push used=%d count=%d header=%+v planned %d,%d,%+v", used, count, header, plannedUsed, plannedSpans, plannedHeader)
			}
		}
		lastIndex := -1
		for {
			elem, ok, err := it.Next()
			if err != nil {
				return
			}
			if !ok {
				return
			}
			if len(elem.Data) == 0 && !elem.fragmented() {
				t.Fatal("empty element returned without error")
			}
			if elem.Index < lastIndex {
				t.Fatalf("index regressed from %d to %d", lastIndex, elem.Index)
			}
			lastIndex = elem.Index
		}
	})
}

func FuzzPacketizer(f *testing.F) {
	f.Add(appendPacketizerOBU(nil, obu.TypeFrameHeader, []byte{0x80}), uint8(6))
	f.Add(appendPacketizerOBU(nil, obu.TypeFrame, []byte{0, 1, 2, 3, 4, 5, 6}), uint8(8))

	f.Fuzz(func(t *testing.T, payload []byte, maxPayloadLen uint8) {
		limit := int(maxPayloadLen%64) + 3
		var obus [16]PacketizerOBU
		var packets [128]PacketPlan
		var work [128]PacketPlan

		obuCount, err := CountPacketizerOBUs(payload)
		if err != nil {
			return
		}
		scratch, err := PacketizerScratchLen(payload, PayloadSizeLimits{MaxPayloadLen: limit}, nil)
		if err != nil {
			t.Fatalf("PacketizerScratchLen first pass after count: %v", err)
		}
		if scratch.OBUs != obuCount || scratch.Packets != 0 || scratch.Work != 0 {
			t.Fatalf("first pass scratch=%+v count=%d", scratch, obuCount)
		}
		if obuCount > len(obus) {
			return
		}
		parsed, err := ParsePacketizerOBUs(payload, obus[:])
		if err != nil {
			t.Fatalf("ParsePacketizerOBUs after count: %v", err)
		}
		if parsed != obuCount {
			t.Fatalf("parsed=%d count=%d", parsed, obuCount)
		}
		for i, info := range obus[:parsed] {
			if info.Header.Type != info.Type ||
				info.Header.HasSizeField ||
				info.HeaderSize != info.Header.HeaderLen() ||
				info.Size != info.HeaderSize+info.PayloadSize {
				t.Fatalf("obu[%d]=%+v", i, info)
			}
		}
		packetCount, err := PacketizeOBUCount(obus[:parsed], PayloadSizeLimits{MaxPayloadLen: limit})
		if err != nil {
			t.Fatalf("PacketizeOBUCount: %v", err)
		}
		scratch, err = PacketizerScratchLen(payload, PayloadSizeLimits{MaxPayloadLen: limit}, obus[:])
		if err != nil {
			t.Fatalf("PacketizerScratchLen second pass: %v", err)
		}
		if scratch.OBUs != obuCount || scratch.Packets != packetCount {
			t.Fatalf("second pass scratch=%+v count=%d packets=%d", scratch, obuCount, packetCount)
		}
		if packetCount <= 1 {
			if scratch.Work != 0 {
				t.Fatalf("scratch work=%d want 0 for packetCount=%d", scratch.Work, packetCount)
			}
		} else if scratch.Work != packetCount {
			t.Fatalf("scratch work=%d packetCount=%d", scratch.Work, packetCount)
		}
		if packetCount > len(packets) {
			return
		}

		packetizer, err := NewPacketizer(payload, PayloadSizeLimits{MaxPayloadLen: limit}, true, true, obus[:], packets[:], work[:])
		if err != nil {
			t.Fatalf("NewPacketizer after sizing: %v", err)
		}
		if packetizer.NumPackets() != packetCount {
			t.Fatalf("NumPackets=%d count=%d", packetizer.NumPackets(), packetCount)
		}

		var out [128]byte
		for {
			nextSize, sizeOK := packetizer.NextPacketSize()
			nextPlan, planOK := packetizer.NextPacketPlan()
			if sizeOK != planOK {
				t.Fatalf("NextPacketSize ok=%v NextPacketPlan ok=%v", sizeOK, planOK)
			}
			if sizeOK && nextSize != nextPlan.PacketSize+1 {
				t.Fatalf("next size=%d plan=%+v", nextSize, nextPlan)
			}
			n, _, ok, err := packetizer.NextPacket(out[:])
			if err != nil {
				t.Fatalf("NextPacket: %v", err)
			}
			if !ok {
				if sizeOK {
					t.Fatalf("NextPacket done after size=%d plan=%+v", nextSize, nextPlan)
				}
				return
			}
			if !sizeOK || n != nextSize {
				t.Fatalf("packet n=%d size=%d ok=%v", n, nextSize, sizeOK)
			}
			if n > limit {
				t.Fatalf("packet size=%d limit=%d", n, limit)
			}
			it, err := NewIterator(out[:n])
			if err != nil {
				t.Fatalf("NewIterator: %v", err)
			}
			for {
				_, ok, err := it.Next()
				if err != nil {
					t.Fatalf("Iterator.Next: %v", err)
				}
				if !ok {
					break
				}
			}
		}
	})
}

func FuzzAssembleFrame(f *testing.F) {
	f.Add([]byte{0x10, byte(obu.TypeFrame) << 3, 0xaa})
	f.Add([]byte{0x20, 0x02, byte(obu.TypeSequenceHeader) << 3, 0xaa, byte(obu.TypeFrameHeader) << 3, 0xbb})
	f.Add([]byte{0x50, byte(obu.TypeFrame) << 3, 0xaa})

	f.Fuzz(func(t *testing.T, payload []byte) {
		var out [512]byte
		var obus [16]FrameOBU
		size, count, err := AssembleFrameSize([][]byte{payload})
		if err != nil {
			return
		}
		if size > len(out) || count > len(obus) {
			return
		}
		wrote, assembledCount, err := AssembleFrame(out[:], [][]byte{payload}, obus[:])
		if err != nil {
			t.Fatalf("AssembleFrame after sizing: %v", err)
		}
		if wrote != size || assembledCount != count {
			t.Fatalf("assembled size=%d count=%d want %d,%d", wrote, assembledCount, size, count)
		}

		it := obu.NewLowOverheadIterator(out[:wrote])
		offset := 0
		index := 0
		for {
			unit, ok, err := it.Next()
			if err != nil {
				t.Fatalf("assembled low-overhead parse failed: %v", err)
			}
			if !ok {
				if index != assembledCount || offset != wrote {
					t.Fatalf("parsed index=%d offset=%d want count=%d wrote=%d", index, offset, assembledCount, wrote)
				}
				return
			}
			if index >= assembledCount {
				t.Fatalf("parsed more OBUs than metadata count=%d", assembledCount)
			}
			info := obus[index]
			if info.Header != unit.Header ||
				info.Offset != offset ||
				info.Length != len(unit.Raw) ||
				info.PrefixSize != int(unit.HeaderLen)+int(unit.SizeLen) ||
				info.PayloadSize != len(unit.Payload) ||
				info.PayloadOffset() != offset+int(unit.HeaderLen)+int(unit.SizeLen) ||
				info.PayloadEnd() != offset+len(unit.Raw) {
				t.Fatalf("metadata[%d]=%+v unit=%+v offset=%d", index, info, unit, offset)
			}
			offset += len(unit.Raw)
			index++
		}
	})
}

func appendPacketizerOBU(dst []byte, typ obu.Type, payload []byte) []byte {
	var header [2]byte
	n, err := obu.PutHeader(header[:], obu.Header{Type: typ, HasSizeField: true})
	if err != nil {
		panic(err)
	}
	dst = append(dst, header[:n]...)
	var size [bitstream.MaxLEB128Bytes]byte
	n, err = bitstream.PutLEB128(size[:], uint32(len(payload)))
	if err != nil {
		panic(err)
	}
	dst = append(dst, size[:n]...)
	dst = append(dst, payload...)
	return dst
}

func appendPacketizerOBUExt(dst []byte, typ obu.Type, temporalID uint8, spatialID uint8, payload []byte) []byte {
	var header [2]byte
	n, err := obu.PutHeader(header[:], obu.Header{
		Type:         typ,
		Extension:    true,
		HasSizeField: true,
		TemporalID:   temporalID,
		SpatialID:    spatialID,
	})
	if err != nil {
		panic(err)
	}
	dst = append(dst, header[:n]...)
	var size [bitstream.MaxLEB128Bytes]byte
	n, err = bitstream.PutLEB128(size[:], uint32(len(payload)))
	if err != nil {
		panic(err)
	}
	dst = append(dst, size[:n]...)
	dst = append(dst, payload...)
	return dst
}

func TestPacketizerFiltersAndStripsSizeFields(t *testing.T) {
	var frame []byte
	frame = appendPacketizerOBU(frame, obu.TypeTemporalDelimiter, nil)
	frame = appendPacketizerOBU(frame, obu.TypeSequenceHeader, []byte{0xaa})
	frame = appendPacketizerOBU(frame, obu.TypeFrameHeader, []byte{0xbb, 0xcc})
	frame = appendPacketizerOBU(frame, obu.TypePadding, []byte{0x00})

	var obus [4]PacketizerOBU
	var packets [4]PacketPlan
	var work [4]PacketPlan
	packetizer, err := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 1200}, true, true, obus[:], packets[:], work[:])
	if err != nil {
		t.Fatal(err)
	}
	if packetizer.NumPackets() != 1 {
		t.Fatalf("NumPackets=%d", packetizer.NumPackets())
	}

	var payload [32]byte
	n, marker, ok, err := packetizer.NextPacket(payload[:])
	if err != nil {
		t.Fatal(err)
	}
	if !ok || !marker {
		t.Fatalf("ok=%v marker=%v", ok, marker)
	}

	want := []byte{
		0x28, // W=2, N=1.
		0x02, byte(obu.TypeSequenceHeader) << 3, 0xaa,
		byte(obu.TypeFrameHeader) << 3, 0xbb, 0xcc,
	}
	if string(payload[:n]) != string(want) {
		t.Fatalf("payload=%x want=%x", payload[:n], want)
	}
}

func TestCountPacketizerOBUsMatchesParse(t *testing.T) {
	var frame []byte
	frame = appendPacketizerOBU(frame, obu.TypeTemporalDelimiter, nil)
	frame = appendPacketizerOBU(frame, obu.TypeSequenceHeader, []byte{0xaa})
	frame = appendPacketizerOBUExt(frame, obu.TypeFrameHeader, 1, 2, []byte{0xbb})
	frame = appendPacketizerOBU(frame, obu.TypePadding, []byte{0x00})

	count, err := CountPacketizerOBUs(frame)
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d want 2", count)
	}

	var obus [2]PacketizerOBU
	parsed, err := ParsePacketizerOBUs(frame, obus[:])
	if err != nil {
		t.Fatal(err)
	}
	if parsed != count {
		t.Fatalf("parsed=%d count=%d", parsed, count)
	}
	if obus[0].Type != obu.TypeSequenceHeader ||
		obus[0].Header.Type != obu.TypeSequenceHeader ||
		obus[0].Header.HasSizeField ||
		obus[0].HeaderSize != 1 ||
		obus[0].PayloadSize != 1 ||
		obus[0].Size != 2 {
		t.Fatalf("obu0=%+v", obus[0])
	}
	if obus[1].Type != obu.TypeFrameHeader ||
		obus[1].Header.Type != obu.TypeFrameHeader ||
		!obus[1].Header.Extension ||
		obus[1].Header.HasSizeField ||
		obus[1].Header.TemporalID != 1 ||
		obus[1].Header.SpatialID != 2 ||
		obus[1].HeaderSize != 2 ||
		obus[1].PayloadSize != 1 ||
		obus[1].Size != 3 ||
		!obus[1].hasExtension {
		t.Fatalf("obus=%+v", obus)
	}
}

func TestCountPacketizerOBUsRejectsMalformedOBU(t *testing.T) {
	payload := appendPacketizerOBU(nil, obu.TypeFrame, []byte{0xaa, 0xbb})
	payload[1] = 3
	if _, err := CountPacketizerOBUs(payload); !errors.Is(err, obu.ErrShortPayload) {
		t.Fatalf("CountPacketizerOBUs err=%v want %v", err, obu.ErrShortPayload)
	}
}

func TestPacketizerWritesLengthForMoreThanThreeOBUs(t *testing.T) {
	var frame []byte
	for i := range 4 {
		frame = appendPacketizerOBU(frame, obu.TypeMetadata, []byte{byte(i)})
	}

	var obus [4]PacketizerOBU
	var packets [2]PacketPlan
	var work [2]PacketPlan
	packetizer, err := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 1200}, false, true, obus[:], packets[:], work[:])
	if err != nil {
		t.Fatal(err)
	}

	var payload [32]byte
	n, _, ok, err := packetizer.NextPacket(payload[:])
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("no packet")
	}
	want := []byte{
		0x00,
		0x02, byte(obu.TypeMetadata) << 3, 0x00,
		0x02, byte(obu.TypeMetadata) << 3, 0x01,
		0x02, byte(obu.TypeMetadata) << 3, 0x02,
		0x02, byte(obu.TypeMetadata) << 3, 0x03,
	}
	if string(payload[:n]) != string(want) {
		t.Fatalf("payload=%x want=%x", payload[:n], want)
	}
}

func TestPacketizerNextPacketSize(t *testing.T) {
	payloadBytes := []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	frame := appendPacketizerOBU(nil, obu.TypeFrame, payloadBytes)

	var obus [2]PacketizerOBU
	var packets [8]PacketPlan
	var work [8]PacketPlan
	packetizer, err := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 6}, false, true, obus[:], packets[:], work[:])
	if err != nil {
		t.Fatal(err)
	}

	size, ok := packetizer.NextPacketSize()
	if !ok || size != 6 {
		t.Fatalf("first size=%d ok=%v want 6,true", size, ok)
	}
	if remaining := packetizer.RemainingPacketPayloadSize(); remaining != 14 {
		t.Fatalf("remaining payload size=%d want 14", remaining)
	}
	plan, ok := packetizer.NextPacketPlan()
	if !ok || plan.PacketSize+1 != size || plan.FirstOBU != 0 || plan.NumOBUElements != 1 || plan.FirstOBUOffset != 0 || plan.LastOBUSize != 5 {
		t.Fatalf("first plan=%+v ok=%v size=%d", plan, ok, size)
	}
	var short [5]byte
	if _, _, _, err := packetizer.NextPacket(short[:]); !errors.Is(err, ErrShortBuffer) {
		t.Fatalf("short NextPacket err=%v want %v", err, ErrShortBuffer)
	}
	if retryPlan, ok := packetizer.NextPacketPlan(); !ok || retryPlan != plan {
		t.Fatalf("retry plan=%+v ok=%v want %+v,true", retryPlan, ok, plan)
	}
	if retrySize, ok := packetizer.NextPacketSize(); !ok || retrySize != size || packetizer.NumPackets() != 3 {
		t.Fatalf("retry size=%d ok=%v remaining=%d want %d,true,3", retrySize, ok, packetizer.NumPackets(), size)
	}

	var payload [16]byte
	wantRemaining := [...]int{8, 2, 0}
	for i := range 3 {
		size, ok := packetizer.NextPacketSize()
		if !ok {
			t.Fatalf("missing size for packet %d", i)
		}
		n, _, ok, err := packetizer.NextPacket(payload[:size])
		if err != nil {
			t.Fatal(err)
		}
		if !ok || n != size {
			t.Fatalf("packet %d n=%d ok=%v want %d,true", i, n, ok, size)
		}
		if remaining := packetizer.RemainingPacketPayloadSize(); remaining != wantRemaining[i] {
			t.Fatalf("packet %d remaining payload size=%d want %d", i, remaining, wantRemaining[i])
		}
	}
	if size, ok := packetizer.NextPacketSize(); ok || size != 0 {
		t.Fatalf("done size=%d ok=%v want 0,false", size, ok)
	}
	if plan, ok := packetizer.NextPacketPlan(); ok || plan != (PacketPlan{}) {
		t.Fatalf("done plan=%+v ok=%v want zero,false", plan, ok)
	}
}

func TestPacketizerNextPacketDependencyDescriptorFlags(t *testing.T) {
	frame := appendPacketizerOBU(nil, obu.TypeFrame, []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

	var obus [2]PacketizerOBU
	var packets [8]PacketPlan
	var work [8]PacketPlan
	packetizer, err := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 6}, false, true, obus[:], packets[:], work[:])
	if err != nil {
		t.Fatal(err)
	}

	want := [...]PacketDependencyDescriptorFlags{
		{FirstPacketInFrame: true},
		{},
		{LastPacketInFrame: true},
	}
	var payload [16]byte
	for i := range want {
		got, ok := packetizer.NextPacketDependencyDescriptorFlags()
		if !ok || got != want[i] {
			t.Fatalf("packet %d flags=%+v ok=%v want %+v,true", i, got, ok, want[i])
		}
		if _, _, _, err := packetizer.NextPacket(payload[:]); err != nil {
			t.Fatal(err)
		}
	}
	if got, ok := packetizer.NextPacketDependencyDescriptorFlags(); ok || got != (PacketDependencyDescriptorFlags{}) {
		t.Fatalf("done flags=%+v ok=%v want zero,false", got, ok)
	}
}

func TestPacketizerSinglePacketDependencyDescriptorFlags(t *testing.T) {
	frame := appendPacketizerOBU(nil, obu.TypeFrame, []byte{0xaa})

	var obus [1]PacketizerOBU
	var packets [1]PacketPlan
	var work [1]PacketPlan
	packetizer, err := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 1200}, false, true, obus[:], packets[:], work[:])
	if err != nil {
		t.Fatal(err)
	}
	got, ok := packetizer.NextPacketDependencyDescriptorFlags()
	if !ok || got != (PacketDependencyDescriptorFlags{FirstPacketInFrame: true, LastPacketInFrame: true}) {
		t.Fatalf("single flags=%+v ok=%v", got, ok)
	}
}

func TestPacketizerFragmentsOBU(t *testing.T) {
	payloadBytes := []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	frame := appendPacketizerOBU(nil, obu.TypeFrame, payloadBytes)

	var obus [2]PacketizerOBU
	var packets [8]PacketPlan
	var work [8]PacketPlan
	packetizer, err := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 6}, false, true, obus[:], packets[:], work[:])
	if err != nil {
		t.Fatal(err)
	}
	if packetizer.NumPackets() != 3 {
		t.Fatalf("NumPackets=%d", packetizer.NumPackets())
	}

	wantHeaders := []byte{0x50, 0xd0, 0x90}
	var out [16]byte
	var assembled [32]byte
	var spans [4]OBUSpan
	var dep Depacketizer
	used := 0
	for i := range 3 {
		n, marker, ok, err := packetizer.NextPacket(out[:])
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			t.Fatalf("missing packet %d", i)
		}
		if out[0] != wantHeaders[i] {
			t.Fatalf("packet %d header=%02x want %02x", i, out[0], wantHeaders[i])
		}
		if marker != (i == 2) {
			t.Fatalf("packet %d marker=%v", i, marker)
		}
		var count int
		used, count, _, err = dep.Push(assembled[:], used, spans[:], out[:n])
		if err != nil {
			t.Fatal(err)
		}
		if i < 2 && count != 0 {
			t.Fatalf("packet %d completed early", i)
		}
		if i == 2 && count != 1 {
			t.Fatalf("final count=%d", count)
		}
	}

	wantOBU := append([]byte{byte(obu.TypeFrame) << 3}, payloadBytes...)
	if string(assembled[:used]) != string(wantOBU) {
		t.Fatalf("assembled=%x want=%x", assembled[:used], wantOBU)
	}
}

func TestPacketizeOBUCountMatchesPacketizeOBUs(t *testing.T) {
	payloadBytes := []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	frame := appendPacketizerOBU(nil, obu.TypeFrame, payloadBytes)

	obuCount, err := CountPacketizerOBUs(frame)
	if err != nil {
		t.Fatal(err)
	}
	var obus [2]PacketizerOBU
	parsed, err := ParsePacketizerOBUs(frame, obus[:])
	if err != nil {
		t.Fatal(err)
	}
	if parsed != obuCount {
		t.Fatalf("parsed=%d count=%d", parsed, obuCount)
	}

	limits := PayloadSizeLimits{MaxPayloadLen: 6}
	planCount, err := PacketizeOBUCount(obus[:parsed], limits)
	if err != nil {
		t.Fatal(err)
	}
	var packets [8]PacketPlan
	var work [8]PacketPlan
	packetized, err := PacketizeOBUs(obus[:parsed], limits, packets[:], work[:])
	if err != nil {
		t.Fatal(err)
	}
	if planCount != packetized || planCount != 3 {
		t.Fatalf("planCount=%d packetized=%d want 3", planCount, packetized)
	}
	for i := range planCount {
		if packets[i].PacketSize == 0 || packets[i].NumOBUElements == 0 {
			t.Fatalf("packet[%d]=%+v", i, packets[i])
		}
	}
}

func TestPacketizeOBUCountRejectsInvalidLimits(t *testing.T) {
	obus := []PacketizerOBU{{Type: obu.TypeFrame, Size: 3}}
	if _, err := PacketizeOBUCount(obus, PayloadSizeLimits{MaxPayloadLen: 2}); !errors.Is(err, ErrInvalidPayloadLimits) {
		t.Fatalf("PacketizeOBUCount err=%v want %v", err, ErrInvalidPayloadLimits)
	}
}

func TestPacketizerScratchLenTwoPass(t *testing.T) {
	payloadBytes := []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	var frame []byte
	frame = appendPacketizerOBU(frame, obu.TypeTemporalDelimiter, nil)
	frame = appendPacketizerOBU(frame, obu.TypeSequenceHeader, []byte{0xaa})
	frame = appendPacketizerOBU(frame, obu.TypeFrame, payloadBytes)
	limits := PayloadSizeLimits{MaxPayloadLen: 6}

	size, err := PacketizerScratchLen(frame, limits, nil)
	if err != nil {
		t.Fatal(err)
	}
	if size.OBUs != 2 || size.Packets != 0 || size.Work != 0 {
		t.Fatalf("first pass size=%+v want OBUs=2 only", size)
	}

	var obus [4]PacketizerOBU
	size, err = PacketizerScratchLen(frame, limits, obus[:])
	if err != nil {
		t.Fatal(err)
	}
	if size.OBUs != 2 || size.Packets <= 1 || size.Work != size.Packets {
		t.Fatalf("second pass size=%+v", size)
	}

	var packets [8]PacketPlan
	var work [8]PacketPlan
	packetizer, err := NewPacketizer(frame, limits, true, true, obus[:size.OBUs], packets[:size.Packets], work[:size.Work])
	if err != nil {
		t.Fatal(err)
	}
	if packetizer.NumPackets() != size.Packets {
		t.Fatalf("NumPackets=%d scratch packets=%d", packetizer.NumPackets(), size.Packets)
	}
}

func TestPacketizerScratchLenWaitsForOBUScratchBeforePacketSizing(t *testing.T) {
	var frame []byte
	frame = appendPacketizerOBU(frame, obu.TypeSequenceHeader, []byte{0xaa})
	frame = appendPacketizerOBU(frame, obu.TypeFrameHeader, []byte{0xbb})
	limits := PayloadSizeLimits{MaxPayloadLen: 2}

	var short [1]PacketizerOBU
	size, err := PacketizerScratchLen(frame, limits, short[:])
	if err != nil {
		t.Fatal(err)
	}
	if size.OBUs != 2 || size.Packets != 0 || size.Work != 0 {
		t.Fatalf("short scratch size=%+v want OBUs=2 only", size)
	}

	var obus [2]PacketizerOBU
	size, err = PacketizerScratchLen(frame, limits, obus[:])
	if !errors.Is(err, ErrInvalidPayloadLimits) {
		t.Fatalf("PacketizerScratchLen err=%v want %v", err, ErrInvalidPayloadLimits)
	}
	if size.OBUs != 2 || size.Packets != 0 || size.Work != 0 {
		t.Fatalf("invalid-limit size=%+v want OBUs=2 only", size)
	}
}

func TestPacketizerScratchLenEmptyPayloadIgnoresLimits(t *testing.T) {
	size, err := PacketizerScratchLen(nil, PayloadSizeLimits{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if size != (PacketizerScratchSize{}) {
		t.Fatalf("size=%+v want zero", size)
	}
}

func TestPacketizerPreservesExtensionAndClearsSize(t *testing.T) {
	frame := appendPacketizerOBUExt(nil, obu.TypeTileGroup, 1, 2, []byte{0x99})
	var obus [1]PacketizerOBU
	var packets [1]PacketPlan
	var work [1]PacketPlan
	packetizer, err := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 1200}, false, true, obus[:], packets[:], work[:])
	if err != nil {
		t.Fatal(err)
	}

	var payload [16]byte
	n, _, ok, err := packetizer.NextPacket(payload[:])
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("no packet")
	}
	want := []byte{
		0x10,
		byte(obu.TypeTileGroup)<<3 | 0x04,
		(1 << 5) | (2 << 3),
		0x99,
	}
	if string(payload[:n]) != string(want) {
		t.Fatalf("payload=%x want=%x", payload[:n], want)
	}
}

func TestPacketizerSplitsLayerTaggedOBUsByLayer(t *testing.T) {
	var frame []byte
	frame = appendPacketizerOBUExt(frame, obu.TypeFrameHeader, 0, 0, []byte{0x10})
	frame = appendPacketizerOBUExt(frame, obu.TypeTileGroup, 0, 0, []byte{0x11})
	frame = appendPacketizerOBUExt(frame, obu.TypeTileGroup, 1, 0, []byte{0x12})

	var obus [3]PacketizerOBU
	var packets [3]PacketPlan
	var work [3]PacketPlan
	packetizer, err := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 1200}, false, true, obus[:], packets[:], work[:])
	if err != nil {
		t.Fatal(err)
	}
	if packetizer.NumPackets() != 2 {
		t.Fatalf("NumPackets=%d want 2", packetizer.NumPackets())
	}

	var payload [64]byte
	n, marker, ok, err := packetizer.NextPacket(payload[:])
	if err != nil {
		t.Fatal(err)
	}
	if !ok || marker {
		t.Fatalf("first packet ok=%v marker=%v", ok, marker)
	}
	firstHeaders := packetOBUHeaders(t, payload[:n])
	if len(firstHeaders) != 2 ||
		!firstHeaders[0].Extension || firstHeaders[0].TemporalID != 0 || firstHeaders[0].SpatialID != 0 ||
		!firstHeaders[1].Extension || firstHeaders[1].TemporalID != 0 || firstHeaders[1].SpatialID != 0 {
		t.Fatalf("first packet headers=%+v", firstHeaders)
	}

	n, marker, ok, err = packetizer.NextPacket(payload[:])
	if err != nil {
		t.Fatal(err)
	}
	if !ok || !marker {
		t.Fatalf("second packet ok=%v marker=%v", ok, marker)
	}
	secondHeaders := packetOBUHeaders(t, payload[:n])
	if len(secondHeaders) != 1 ||
		!secondHeaders[0].Extension || secondHeaders[0].TemporalID != 1 || secondHeaders[0].SpatialID != 0 {
		t.Fatalf("second packet headers=%+v", secondHeaders)
	}
}

func TestPacketizerKeepsSequenceHeaderWithFirstLayer(t *testing.T) {
	var frame []byte
	frame = appendPacketizerOBU(frame, obu.TypeSequenceHeader, []byte{0xaa})
	frame = appendPacketizerOBUExt(frame, obu.TypeFrameHeader, 0, 0, []byte{0x10})
	frame = appendPacketizerOBU(frame, obu.TypeSequenceHeader, []byte{0xbb})
	frame = appendPacketizerOBUExt(frame, obu.TypeTileGroup, 0, 1, []byte{0x11})

	var obus [4]PacketizerOBU
	var packets [4]PacketPlan
	var work [4]PacketPlan
	packetizer, err := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 1200}, true, true, obus[:], packets[:], work[:])
	if err != nil {
		t.Fatal(err)
	}
	if packetizer.NumPackets() != 2 {
		t.Fatalf("NumPackets=%d want 2", packetizer.NumPackets())
	}

	var payload [64]byte
	n, marker, ok, err := packetizer.NextPacket(payload[:])
	if err != nil {
		t.Fatal(err)
	}
	if !ok || marker {
		t.Fatalf("first packet ok=%v marker=%v", ok, marker)
	}
	it, err := NewIterator(payload[:n])
	if err != nil {
		t.Fatal(err)
	}
	if header := it.Header(); !header.StartsNewCodedVideoSequence || header.ElementCount != 2 {
		t.Fatalf("first packet aggregation header=%+v", header)
	}
	firstHeaders := packetOBUHeaders(t, payload[:n])
	if len(firstHeaders) != 2 ||
		firstHeaders[0].Type != obu.TypeSequenceHeader || firstHeaders[0].Extension ||
		firstHeaders[1].Type != obu.TypeFrameHeader || !firstHeaders[1].Extension ||
		firstHeaders[1].TemporalID != 0 || firstHeaders[1].SpatialID != 0 {
		t.Fatalf("first packet headers=%+v", firstHeaders)
	}

	n, marker, ok, err = packetizer.NextPacket(payload[:])
	if err != nil {
		t.Fatal(err)
	}
	if !ok || !marker {
		t.Fatalf("second packet ok=%v marker=%v", ok, marker)
	}
	secondHeaders := packetOBUHeaders(t, payload[:n])
	if len(secondHeaders) != 2 ||
		secondHeaders[0].Type != obu.TypeSequenceHeader || secondHeaders[0].Extension ||
		secondHeaders[1].Type != obu.TypeTileGroup || !secondHeaders[1].Extension ||
		secondHeaders[1].TemporalID != 0 || secondHeaders[1].SpatialID != 1 {
		t.Fatalf("second packet headers=%+v", secondHeaders)
	}
}

func packetOBUHeaders(t *testing.T, payload []byte) []obu.Header {
	t.Helper()
	it, err := NewIterator(payload)
	if err != nil {
		t.Fatal(err)
	}
	var headers []obu.Header
	for {
		element, ok, err := it.Next()
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			return headers
		}
		header, _, err := obu.ParseHeader(element.Data)
		if err != nil {
			t.Fatal(err)
		}
		headers = append(headers, header)
	}
}

func TestPacketizerScratchTooSmall(t *testing.T) {
	frame := appendPacketizerOBU(nil, obu.TypeFrameHeader, []byte{0xaa})
	_, err := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 1200}, false, true, nil, nil, nil)
	if !errors.Is(err, ErrOBUBufferTooSmall) {
		t.Fatalf("NewPacketizer err=%v want %v", err, ErrOBUBufferTooSmall)
	}

	var obus [1]PacketizerOBU
	count, err := ParsePacketizerOBUs(frame, obus[:])
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PacketizeOBUs(obus[:count], PayloadSizeLimits{MaxPayloadLen: 1200}, nil, nil); !errors.Is(err, ErrPacketPlanTooSmall) {
		t.Fatalf("PacketizeOBUs err=%v want %v", err, ErrPacketPlanTooSmall)
	}
}

func TestPacketizerAllocs(t *testing.T) {
	var frame []byte
	frame = appendPacketizerOBU(frame, obu.TypeSequenceHeader, []byte{0xaa})
	frame = appendPacketizerOBU(frame, obu.TypeFrameHeader, []byte{0xbb})
	var obus [4]PacketizerOBU
	var packets [4]PacketPlan
	var work [4]PacketPlan
	var payload [32]byte

	allocs := testing.AllocsPerRun(1000, func() {
		packetizer, err := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 1200}, true, true, obus[:], packets[:], work[:])
		if err != nil {
			t.Fatal(err)
		}
		_, _, _, err = packetizer.NextPacket(payload[:])
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Packetizer allocated: %f", allocs)
	}
}

func TestPacketizerSizingAllocs(t *testing.T) {
	var frame []byte
	frame = appendPacketizerOBU(frame, obu.TypeSequenceHeader, []byte{0xaa})
	frame = appendPacketizerOBU(frame, obu.TypeFrameHeader, []byte{0xbb})
	var obus [4]PacketizerOBU
	count, err := ParsePacketizerOBUs(frame, obus[:])
	if err != nil {
		t.Fatal(err)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		obuCount, err := CountPacketizerOBUs(frame)
		if err != nil {
			t.Fatal(err)
		}
		if obuCount != count {
			t.Fatalf("obuCount=%d want %d", obuCount, count)
		}
		packetCount, err := PacketizeOBUCount(obus[:count], PayloadSizeLimits{MaxPayloadLen: 1200})
		if err != nil {
			t.Fatal(err)
		}
		if packetCount != 1 {
			t.Fatalf("packetCount=%d want 1", packetCount)
		}
		size, err := PacketizerScratchLen(frame, PayloadSizeLimits{MaxPayloadLen: 1200}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if size.OBUs != count || size.Packets != 0 || size.Work != 0 {
			t.Fatalf("first pass size=%+v", size)
		}
		size, err = PacketizerScratchLen(frame, PayloadSizeLimits{MaxPayloadLen: 1200}, obus[:])
		if err != nil {
			t.Fatal(err)
		}
		if size.OBUs != count || size.Packets != packetCount || size.Work != 0 {
			t.Fatalf("second pass size=%+v packetCount=%d", size, packetCount)
		}
	})
	if allocs != 0 {
		t.Fatalf("packetizer sizing allocated: %f", allocs)
	}
}

func BenchmarkPacketizer(b *testing.B) {
	var frame []byte
	frame = appendPacketizerOBU(frame, obu.TypeSequenceHeader, []byte{0xaa})
	frame = appendPacketizerOBU(frame, obu.TypeFrameHeader, []byte{0xbb})
	var obus [4]PacketizerOBU
	var packets [4]PacketPlan
	var work [4]PacketPlan
	var payload [32]byte

	b.ReportAllocs()
	for b.Loop() {
		packetizer, _ := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 1200}, true, true, obus[:], packets[:], work[:])
		_, _, _, _ = packetizer.NextPacket(payload[:])
	}
}

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
	for i := range 300 {
		payload[off] = 0x01
		payload[off+1] = byte(i)
		off += 2
	}

	it, err := NewIterator(payload[:])
	if err != nil {
		t.Fatal(err)
	}

	for i := range 300 {
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
	for i := range elements {
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
	for b.Loop() {
		it, _ := NewIterator(payload)
		for {
			_, ok, _ := it.Next()
			if !ok {
				break
			}
		}
	}
}

func TestRTPCoreVectors(t *testing.T) {
	var oracle testvector.Oracle
	if testvector.OracleEnabled {
		oracle = testvector.NewOracle(testvector.CoreSuite().Manifest)
	}
	for _, tag := range []testvector.Tag{
		testvector.TagRTPPayloadSingleOBU,
		testvector.TagRTPPayloadFragmentedOBU,
	} {
		vector, ok := testvector.CoreVector(tag)
		if !ok {
			t.Fatalf("missing vector tag=%d", tag)
		}
		it, err := NewIterator(vector.Input)
		if err != nil {
			t.Fatalf("%s NewIterator: %v", vector.Name, err)
		}
		elem, ok, err := it.Next()
		if err != nil {
			t.Fatalf("%s Next: %v", vector.Name, err)
		}
		if !ok {
			t.Fatalf("%s missing element", vector.Name)
		}
		if !bytes.Equal(elem.Data, vector.Want) {
			t.Fatalf("%s data=%x want %x", vector.Name, elem.Data, vector.Want)
		}
		if tag == testvector.TagRTPPayloadFragmentedOBU && !elem.ContinuesNext {
			t.Fatalf("%s ContinuesNext=false", vector.Name)
		}
		if testvector.OracleEnabled {
			if err := oracle.CheckBytes(vector.Tag, elem.Data); err != nil {
				t.Fatalf("%s oracle err=%v", vector.Name, err)
			}
		}
	}
}
