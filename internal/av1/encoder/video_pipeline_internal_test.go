package encoder

import (
	"math/rand"
	"testing"
)

// TestPipelineOverlapFires confirms the overlap-eligible configuration actually
// exercises the concurrent leaf/base path (rather than silently falling back to
// the serial FIFO), so the external byte-identity oracle is testing real
// concurrency. It also checks that golden-enabled and scene-cut-enabled streams
// do NOT overlap.
func TestPipelineOverlapFires(t *testing.T) {
	const w, h = 192, 128
	newEnc := func(golden int, sceneCut bool) *VideoEncoder {
		enc, err := NewVideoEncoderCBR(w, h, RateControlConfig{
			TargetBitsPerSecond: 800_000, FramesPerSecond: 30, MinQIndex: 20, MaxQIndex: 200,
		})
		if err != nil {
			t.Fatal(err)
		}
		if err := enc.SetTemporalLayers(2); err != nil {
			t.Fatal(err)
		}
		enc.SetGoldenInterval(golden)
		enc.SetSceneCutKeyframes(sceneCut)
		return enc
	}
	run := func(enc *VideoEncoder, frames int) {
		defer enc.Close()
		if err := enc.SetThroughputPipelining(true); err != nil {
			t.Fatal(err)
		}
		rng := rand.New(rand.NewSource(3))
		bg := make([]byte, w*h)
		for i := range bg {
			bg[i] = uint8(70 + rng.Intn(50))
		}
		for i := 0; i < frames; i++ {
			f := SourceFrame420{
				Y: append([]byte(nil), bg...), U: make([]byte, w/2*h/2), V: make([]byte, w/2*h/2),
				YStride: w, ChromaStride: w / 2, Width: w, Height: h,
			}
			for j := range f.U {
				f.U[j] = 121
				f.V[j] = 129
			}
			for y := 10 + i*3; y < 10+i*3+20 && y < h; y++ {
				for x := 8 + i*2; x < 8+i*2+20 && x < w; x++ {
					f.Y[y*w+x] = 210
				}
			}
			if _, _, _, err := enc.EncodeThroughput(f, false); err != nil {
				t.Fatal(err)
			}
		}
		if _, _, _, err := enc.Drain(); err != nil {
			t.Fatal(err)
		}
	}

	const frames = 12
	eligible := newEnc(0, false)
	run(eligible, frames)
	if eligible.pipeOverlaps == 0 {
		t.Fatalf("overlap-eligible stream never overlapped (pipeOverlaps=0)")
	}

	golden := newEnc(16, false)
	run(golden, frames)
	if golden.pipeOverlaps != 0 {
		t.Fatalf("golden-enabled stream overlapped (pipeOverlaps=%d), want 0", golden.pipeOverlaps)
	}

	sceneCut := newEnc(0, true)
	run(sceneCut, frames)
	if sceneCut.pipeOverlaps != 0 {
		t.Fatalf("scene-cut stream overlapped (pipeOverlaps=%d), want 0", sceneCut.pipeOverlaps)
	}
}
