package rtp

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/obu"
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
	for i := 0; i < b.N; i++ {
		_, _, _ = AssembleFrame(out[:], payloads, obus[:])
	}
}
