package encoder

import (
	"bytes"
	"fmt"
	"os"
	"testing"
)

// The split (decision pass + serial entropy pass) P-frame pipeline must be
// byte-identical to the fused single-pass coder at every tile count: the
// write pass replays the recorded decisions through the same adaptive-CDF
// writers in the same order, so any divergence is a bug in the split. These
// tests are the correctness oracle for PERF_PLAN §6 E1 stage (a).

func encodeSplitOracleSequence(t *testing.T, frames []SourceFrame420, w, h, fps, bitrate, tileCols int, fused bool) [][]byte {
	t.Helper()
	enc, err := NewVideoEncoderCBR(w, h, RateControlConfig{
		TargetBitsPerSecond: bitrate, FramesPerSecond: fps, MinQIndex: 20, MaxQIndex: 200,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	enc.fusedPipeline = fused
	if err := enc.SetTemporalLayers(2); err != nil {
		t.Fatal(err)
	}
	if tileCols > 0 {
		enc.SetTileColumns(tileCols)
	}
	out := make([][]byte, 0, len(frames))
	for i, f := range frames {
		data, _, err := enc.Encode(f, i == 0)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		out = append(out, append([]byte(nil), data...))
	}
	return out
}

func assertSplitMatchesFused(t *testing.T, frames []SourceFrame420, w, h, fps, bitrate int, tileCounts []int) {
	t.Helper()
	for _, tiles := range tileCounts {
		t.Run(fmt.Sprintf("tiles=%d", tiles), func(t *testing.T) {
			fused := encodeSplitOracleSequence(t, frames, w, h, fps, bitrate, tiles, true)
			split := encodeSplitOracleSequence(t, frames, w, h, fps, bitrate, tiles, false)
			if len(fused) != len(split) {
				t.Fatalf("frame count mismatch: fused=%d split=%d", len(fused), len(split))
			}
			for i := range fused {
				if !bytes.Equal(fused[i], split[i]) {
					t.Fatalf("frame %d: split bitstream differs from fused (fused=%d bytes, split=%d bytes)", i, len(fused[i]), len(split[i]))
				}
			}
		})
	}
}

// syntheticMotionFrames builds a deterministic moving-texture clip that
// exercises skip blocks, coded residuals, golden probes, and the 8x8 intra
// fallback (via the injected scene cut at the midpoint).
func syntheticMotionFrames(n, w, h int) []SourceFrame420 {
	frames := make([]SourceFrame420, n)
	for i := range frames {
		y := make([]byte, w*h)
		u := make([]byte, w*h/4)
		v := make([]byte, w*h/4)
		phase := i * 3
		flip := 0
		if i >= n/2 {
			flip = 97 // scene change: shifts every value, forcing intra rescue
		}
		for r := 0; r < h; r++ {
			for c := 0; c < w; c++ {
				val := (c+phase)&255 ^ (r & 63) * 2
				if (c/48+r/32)%2 == 0 {
					val += ((c * r) >> 6) & 31
				}
				y[r*w+c] = byte(val + flip)
			}
		}
		for r := 0; r < h/2; r++ {
			for c := 0; c < w/2; c++ {
				u[r*w/2+c] = byte(128 + ((c - phase) & 31) + flip/2)
				v[r*w/2+c] = byte(128 - (r & 31) - flip/3)
			}
		}
		frames[i] = SourceFrame420{
			Y: y, U: u, V: v,
			YStride: w, ChromaStride: w / 2, Width: w, Height: h,
		}
	}
	return frames
}

func TestPFrameSplitMatchesFusedSynthetic(t *testing.T) {
	const w, h = 640, 384
	frames := syntheticMotionFrames(12, w, h)
	assertSplitMatchesFused(t, frames, w, h, 30, 1_200_000, []int{1, 2, 4})
}

// TestPFrameSplitMatchesFusedRealC runs the oracle on a real 1080p camera
// segment when the local corpus is present (complex motion: compound probes,
// realtime TX splits, interpolation filter search all engage).
func TestPFrameSplitMatchesFusedRealC(t *testing.T) {
	const path = "/tmp/corpus/realC.yuv"
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("local corpus unavailable: %v", err)
	}
	const w, h = 1920, 1080
	fl := w * h * 3 / 2
	nf := len(raw) / fl
	if nf > 16 {
		nf = 16
	}
	if nf < 3 {
		t.Skipf("%s too short", path)
	}
	frames := make([]SourceFrame420, nf)
	for i := 0; i < nf; i++ {
		base := i * fl
		frames[i] = SourceFrame420{
			Y: raw[base : base+w*h], U: raw[base+w*h : base+w*h+w*h/4], V: raw[base+w*h+w*h/4 : base+fl],
			YStride: w, ChromaStride: w / 2, Width: w, Height: h,
		}
	}
	assertSplitMatchesFused(t, frames, w, h, 30, 5_000_000, []int{1, 4, 32})
}
