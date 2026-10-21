package goav1_test

import (
	"fmt"
	"runtime"
	"testing"

	goav1 "github.com/thesyncim/goav1"
)

// TestVideoEncoderThreadsSteadyStateZeroAlloc pins the zero-allocation design
// on the threaded encoder: wavefront and tile workers, plus forced keyframes
// in the steady state. Parking a goroutine draws a sudog from the runtime's
// per-P cache, and that cache fills over the first few hundred frames. The
// test therefore pins GOMAXPROCS and requires an allocation-free window once
// warm. A per-frame allocation in goav1 would never produce one.
func TestVideoEncoderThreadsSteadyStateZeroAlloc(t *testing.T) {
	if testing.Short() {
		t.Skip("encodes up to a few thousand frames")
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(2))
	const w, h, window, maxWindows = 640, 360, 40, 40
	for _, tc := range []struct {
		threads, tiles, keyEvery int
	}{
		{threads: 4},
		{threads: 4, keyEvery: 13},
		{threads: 4, tiles: 2, keyEvery: 13},
	} {
		t.Run(fmt.Sprintf("threads=%d/tiles=%d/key=%d", tc.threads, tc.tiles, tc.keyEvery), func(t *testing.T) {
			enc, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{
				Width: w, Height: h, TargetBitrate: 1_000_000, Framerate: 30, MaxThreads: tc.threads,
			})
			if err != nil {
				t.Fatalf("NewVideoEncoder: %v", err)
			}
			defer enc.Close()
			if tc.tiles > 0 {
				enc.SetTileColumns(tc.tiles)
			}
			frame := goav1.I420Frame{
				Y: make([]byte, w*h), U: make([]byte, w*h/4), V: make([]byte, w*h/4),
				YStride: w, ChromaStride: w / 2, Width: w, Height: h,
			}
			n := 0
			encodeWindow := func() {
				for range window {
					fillMovingI420(frame, n)
					forceKey := n == 0 || (tc.keyEvery > 0 && n%tc.keyEvery == 0)
					if _, err := enc.Encode(frame, forceKey); err != nil {
						t.Fatalf("Encode frame %d: %v", n, err)
					}
					n++
				}
			}
			encodeWindow()
			var before, after runtime.MemStats
			var allocs uint64
			for range maxWindows {
				runtime.ReadMemStats(&before)
				encodeWindow()
				runtime.ReadMemStats(&after)
				if allocs = after.Mallocs - before.Mallocs; allocs == 0 {
					return
				}
			}
			t.Fatalf("no allocation-free window of %d frames after %d frames; last window allocated %d", window, n, allocs)
		})
	}
}

// fillMovingI420 draws a moving XOR texture with a bright disc so that inter
// frames carry real motion and residual.
func fillMovingI420(f goav1.I420Frame, t int) {
	for y := 0; y < f.Height; y++ {
		for x := 0; x < f.Width; x++ {
			v := (x + 2*t) ^ (y + t)
			dx, dy := x-f.Width/2-3*t%200, y-f.Height/2
			if dx*dx+dy*dy < 60*60 {
				v = 230
			}
			f.Y[y*f.YStride+x] = byte(v)
		}
	}
	for y := 0; y < f.Height/2; y++ {
		for x := 0; x < f.Width/2; x++ {
			f.U[y*f.ChromaStride+x] = byte(128 + (x+t)%32)
			f.V[y*f.ChromaStride+x] = byte(128 - (y+t)%32)
		}
	}
}
