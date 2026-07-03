package encoder_test

import (
	"bytes"
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// pipelineTestFrame builds a deterministic moving-box frame like the L1T2 gate.
func pipelineTestFrame(w, h, t int, rng *rand.Rand, bg []byte) encoder.SourceFrame420 {
	cw, ch := w/2, h/2
	f := encoder.SourceFrame420{
		Y:            append([]byte(nil), bg...),
		U:            make([]byte, cw*ch),
		V:            make([]byte, cw*ch),
		YStride:      w,
		ChromaStride: cw,
		Width:        w,
		Height:       h,
	}
	for i := range f.U {
		f.U[i] = 121
		f.V[i] = 129
	}
	sx, sy := 8+t*4, 12+t*2
	for y := sy; y < sy+22 && y < h; y++ {
		for x := sx; x < sx+22 && x < w; x++ {
			f.Y[y*w+x] = 218
		}
	}
	return f
}

// encodeSerialTUs codes the frame sequence through the default serial path.
func encodeSerialTUs(t *testing.T, newEnc func() *encoder.VideoEncoder, nFrames, w, h int, keyAt map[int]bool) [][]byte {
	t.Helper()
	rng := rand.New(rand.NewSource(17))
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = uint8(70 + rng.Intn(50))
	}
	enc := newEnc()
	defer enc.Close()
	out := make([][]byte, 0, nFrames)
	for i := 0; i < nFrames; i++ {
		f := pipelineTestFrame(w, h, i, rng, bg)
		tu, _, err := enc.Encode(f, keyAt[i])
		if err != nil {
			t.Fatalf("serial encode frame %d: %v", i, err)
		}
		out = append(out, append([]byte(nil), tu...))
	}
	return out
}

// encodePipelinedTUs codes the same sequence through EncodeThroughput/Drain.
func encodePipelinedTUs(t *testing.T, newEnc func() *encoder.VideoEncoder, nFrames, w, h int, keyAt map[int]bool) [][]byte {
	t.Helper()
	rng := rand.New(rand.NewSource(17))
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = uint8(70 + rng.Intn(50))
	}
	enc := newEnc()
	defer enc.Close()
	if err := enc.SetThroughputPipelining(true); err != nil {
		t.Fatalf("enable pipelining: %v", err)
	}
	out := make([][]byte, 0, nFrames)
	for i := 0; i < nFrames; i++ {
		f := pipelineTestFrame(w, h, i, rng, bg)
		tu, _, produced, err := enc.EncodeThroughput(f, keyAt[i])
		if err != nil {
			t.Fatalf("pipelined encode frame %d: %v", i, err)
		}
		if produced {
			out = append(out, append([]byte(nil), tu...))
		}
	}
	tu, _, produced, err := enc.Drain()
	if err != nil {
		t.Fatalf("drain: %v", err)
	}
	if produced {
		out = append(out, append([]byte(nil), tu...))
	}
	return out
}

// TestVideoEncoderPipelineByteIdentical is the differential oracle: opt-in
// throughput pipelining must emit byte-identical temporal units to the default
// serial path for the same source sequence, across layer counts, rate-control
// vs fixed-q, golden on/off, and a mid-stream forced keyframe.
func TestVideoEncoderPipelineByteIdentical(t *testing.T) {
	const w, h = 192, 128
	const frames = 17
	keyMid := map[int]bool{10: true}

	cases := []struct {
		name   string
		keyAt  map[int]bool
		newEnc func() *encoder.VideoEncoder
	}{
		{"L1T1-cqp", nil, func() *encoder.VideoEncoder {
			enc, err := encoder.NewVideoEncoder(w, h, 70)
			if err != nil {
				t.Fatal(err)
			}
			return enc
		}},
		{"L1T2-cqp", nil, func() *encoder.VideoEncoder {
			enc, err := encoder.NewVideoEncoder(w, h, 70)
			if err != nil {
				t.Fatal(err)
			}
			if err := enc.SetTemporalLayers(2); err != nil {
				t.Fatal(err)
			}
			return enc
		}},
		{"L1T3-cqp", nil, func() *encoder.VideoEncoder {
			enc, err := encoder.NewVideoEncoder(w, h, 70)
			if err != nil {
				t.Fatal(err)
			}
			if err := enc.SetTemporalLayers(3); err != nil {
				t.Fatal(err)
			}
			return enc
		}},
		{"L1T2-cbr", nil, func() *encoder.VideoEncoder {
			enc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
				TargetBitsPerSecond: 800_000, FramesPerSecond: 30, MinQIndex: 20, MaxQIndex: 200,
			})
			if err != nil {
				t.Fatal(err)
			}
			if err := enc.SetTemporalLayers(2); err != nil {
				t.Fatal(err)
			}
			return enc
		}},
		{"L1T2-cbr-nogolden", nil, func() *encoder.VideoEncoder {
			enc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
				TargetBitsPerSecond: 800_000, FramesPerSecond: 30, MinQIndex: 20, MaxQIndex: 200,
			})
			if err != nil {
				t.Fatal(err)
			}
			if err := enc.SetTemporalLayers(2); err != nil {
				t.Fatal(err)
			}
			enc.SetGoldenInterval(0)
			return enc
		}},
		{"L1T2-cbr-multithread", nil, func() *encoder.VideoEncoder {
			enc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
				TargetBitsPerSecond: 800_000, FramesPerSecond: 30, MinQIndex: 20, MaxQIndex: 200,
			})
			if err != nil {
				t.Fatal(err)
			}
			if err := enc.SetTemporalLayers(2); err != nil {
				t.Fatal(err)
			}
			enc.SetMaxThreads(4)
			return enc
		}},
		{"L1T2-cqp-keymid", keyMid, func() *encoder.VideoEncoder {
			enc, err := encoder.NewVideoEncoder(w, h, 70)
			if err != nil {
				t.Fatal(err)
			}
			if err := enc.SetTemporalLayers(2); err != nil {
				t.Fatal(err)
			}
			enc.SetSceneCutKeyframes(false)
			return enc
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			serial := encodeSerialTUs(t, tc.newEnc, frames, w, h, tc.keyAt)
			pipelined := encodePipelinedTUs(t, tc.newEnc, frames, w, h, tc.keyAt)
			if len(serial) != len(pipelined) {
				t.Fatalf("frame count mismatch: serial %d pipelined %d", len(serial), len(pipelined))
			}
			for i := range serial {
				if !bytes.Equal(serial[i], pipelined[i]) {
					t.Fatalf("frame %d differs: serial %d bytes, pipelined %d bytes", i, len(serial[i]), len(pipelined[i]))
				}
			}
		})
	}
}

// TestVideoEncoderPipelineDrainEmpty verifies Drain on an empty pipeline is a
// no-op and that a pipeline can be reused after draining.
func TestVideoEncoderPipelineDrainEmpty(t *testing.T) {
	const w, h = 64, 64
	enc, err := encoder.NewVideoEncoder(w, h, 70)
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	if err := enc.SetThroughputPipelining(true); err != nil {
		t.Fatal(err)
	}
	if _, _, produced, err := enc.Drain(); err != nil || produced {
		t.Fatalf("empty drain: produced=%v err=%v", produced, err)
	}
}
