package goav1_test

import (
	"errors"
	"runtime"
	"testing"

	goav1 "github.com/thesyncim/goav1"
)

// encodeLimitTestStream encodes n small frames and returns the temporal units.
func encodeLimitTestStream(t *testing.T, w, h, n int) [][]byte {
	t.Helper()
	enc, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{Width: w, Height: h, QIndex: 80})
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	var stream [][]byte
	for i := range n {
		frame := goav1.I420Frame{
			Y: make([]byte, w*h), U: make([]byte, w/2*h/2), V: make([]byte, w/2*h/2),
			YStride: w, ChromaStride: w / 2, Width: w, Height: h,
		}
		for j := range frame.Y {
			frame.Y[j] = uint8(30 + (j*7+i*11)%180)
		}
		for j := range frame.U {
			frame.U[j], frame.V[j] = 118, 134
		}
		out, err := enc.Encode(frame, false)
		if err != nil {
			t.Fatal(err)
		}
		stream = append(stream, append([]byte(nil), out.Data...))
	}
	return stream
}

func TestDecoderFrameSizeLimit(t *testing.T) {
	const w, h = 192, 128
	stream := encodeLimitTestStream(t, w, h, 3)

	for _, tc := range []struct {
		name  string
		limit int
		want  error
	}{
		{name: "unlimited", limit: 0},
		{name: "exact", limit: w * h},
		{name: "one below", limit: w*h - 1, want: goav1.ErrDecoderFrameSizeLimit},
		{name: "tiny", limit: 1, want: goav1.ErrDecoderFrameSizeLimit},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dec, err := goav1.NewDecoder(stream, goav1.WithFrameSizeLimit(tc.limit))
			if !errors.Is(err, tc.want) {
				t.Fatalf("NewDecoder: err=%v want %v", err, tc.want)
			}
			if err == nil {
				frames, err := dec.DecodeAll()
				dec.Close()
				if err != nil || len(frames) != len(stream) {
					t.Fatalf("DecodeAll: %d frames, err=%v", len(frames), err)
				}
			}
			layered, err := goav1.NewLayeredDecoder(stream, goav1.WithFrameSizeLimit(tc.limit))
			if !errors.Is(err, tc.want) {
				t.Fatalf("NewLayeredDecoder: err=%v want %v", err, tc.want)
			}
			if err == nil {
				layered.Close()
			}
		})
	}

	if _, err := goav1.NewDecoder(stream, goav1.WithFrameSizeLimit(-1)); err == nil {
		t.Fatal("negative frame size limit accepted")
	}
}

// TestDecoderBoundsEventsPerPayload feeds payloads made of thousands of
// two-byte OBUs. Each OBU is parsed into a multi-kilobyte event, so the
// constructors must reject the payload instead of sizing buffers by its OBU
// count.
func TestDecoderBoundsEventsPerPayload(t *testing.T) {
	const obus = 20000
	const budget = 8 << 20

	// RTP: aggregation header, then length-prefixed one-byte OBU elements.
	for _, header := range []byte{0x78, 0x28, 0x10} { // padding, metadata, temporal delimiter
		rtp := make([]byte, 0, 1+2*obus)
		rtp = append(rtp, 0x00)
		for range obus {
			rtp = append(rtp, 0x01, header)
		}
		allocated := allocatedBy(func() {
			if dec, err := goav1.NewDecoderFromRTPPayloads([][]byte{rtp}); err == nil {
				dec.Close()
				t.Errorf("header %#x: NewDecoderFromRTPPayloads accepted %d OBUs in one payload", header, obus)
			}
			if dec, err := goav1.NewLayeredDecoderFromRTPPayloads([][]byte{rtp}); err == nil {
				dec.Close()
				t.Errorf("header %#x: NewLayeredDecoderFromRTPPayloads accepted %d OBUs in one payload", header, obus)
			}
		})
		if allocated > budget {
			t.Errorf("header %#x: RTP constructors allocated %d bytes for a %d-byte payload", header, allocated, len(rtp))
		}
	}

	// Low-overhead: temporal delimiters with a zero size field.
	low := make([]byte, 0, 2*obus)
	for range obus {
		low = append(low, 0x12, 0x00)
	}
	allocated := allocatedBy(func() {
		if dec, err := goav1.NewDecoder([][]byte{low}); err == nil {
			dec.Close()
			t.Errorf("NewDecoder accepted %d OBUs in one payload", obus)
		}
		if dec, err := goav1.NewLayeredDecoder([][]byte{low}); err == nil {
			dec.Close()
			t.Errorf("NewLayeredDecoder accepted %d OBUs in one payload", obus)
		}
	})
	if allocated > budget {
		t.Errorf("low-overhead constructors allocated %d bytes for a %d-byte payload", allocated, len(low))
	}
}

func allocatedBy(f func()) uint64 {
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	f()
	runtime.ReadMemStats(&after)
	return after.TotalAlloc - before.TotalAlloc
}
