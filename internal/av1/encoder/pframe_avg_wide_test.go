package encoder

import (
	"math/rand"
	"testing"
)

// The SWAR 8x8 average is portable (no build tag): it must match the byte-loop
// reference on every architecture and build mode.
func TestRealtimeAvg8x8WideMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xA7A8))
	for trial := range 5000 {
		stride := 8 + rng.Intn(56)
		src := make([]byte, stride*8+8)
		fillPFrameTestBytes(rng, src)
		if g, w := realtimeAvg8x8Wide(src, stride), realtimeAvg8x8PureGo(src, stride); g != w {
			t.Fatalf("trial %d stride %d: wide %d want %d", trial, stride, g, w)
		}
	}
}

func TestRealtimeAvg8x8WideExtremes(t *testing.T) {
	for _, v := range []byte{0, 1, 127, 128, 254, 255} {
		for _, stride := range []int{8, 9, 16, 33} {
			src := make([]byte, stride*8+8)
			for i := range src {
				src[i] = v
			}
			if g, w := realtimeAvg8x8Wide(src, stride), realtimeAvg8x8PureGo(src, stride); g != w {
				t.Fatalf("value %d stride %d: wide %d want %d", v, stride, g, w)
			}
		}
	}
	// Alternating 0/255 bytes exercise both halves of every 16-bit lane.
	src := make([]byte, 8*8+8)
	for i := range src {
		if i&1 == 0 {
			src[i] = 255
		}
	}
	if g, w := realtimeAvg8x8Wide(src, 8), realtimeAvg8x8PureGo(src, 8); g != w {
		t.Fatalf("alternating: wide %d want %d", g, w)
	}
}

func TestRealtimeAvg8x8QuadWideMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xA7A9))
	for trial := range 5000 {
		stride := 16 + rng.Intn(56)
		src := make([]byte, stride*16+16)
		fillPFrameTestBytes(rng, src)
		g0, g1, g2, g3 := realtimeAvg8x8QuadWide(src, stride)
		w0, w1, w2, w3 := realtimeAvg8x8QuadPureGo(src, stride)
		if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
			t.Fatalf("trial %d stride %d: wide (%d,%d,%d,%d) want (%d,%d,%d,%d)",
				trial, stride, g0, g1, g2, g3, w0, w1, w2, w3)
		}
	}
}

// fillPFrameTestBytes fills b with random bytes, mixing full-range noise with
// saturated runs so lane-overflow edges are reached.
func fillPFrameTestBytes(rng *rand.Rand, b []byte) {
	for i := range b {
		switch rng.Intn(4) {
		case 0:
			b[i] = 255
		case 1:
			b[i] = 0
		default:
			b[i] = byte(rng.Intn(256))
		}
	}
}
