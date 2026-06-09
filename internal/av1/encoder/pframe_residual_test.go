package encoder_test

import (
	"fmt"
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestEncodePFrameDecodeMatchesRecon is the temporal-compression gate: a KEY +
// residual-P sequence (the second source differing from the first by a small
// brightness drift plus noise) must decode with frame 1 equal to the encoder's
// P-frame reconstruction bit for bit, with sane fidelity against the second
// source and a P-frame substantially smaller than the keyframe — evidence the
// temporal prediction is doing the work.
func TestEncodePFrameDecodeMatchesRecon(t *testing.T) {
	sizes := []struct{ w, h int }{{64, 64}, {128, 128}}
	for _, sz := range sizes {
		t.Run(fmt.Sprintf("%dx%d", sz.w, sz.h), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(sz.w)*131 + int64(sz.h)))
			cw, ch := sz.w/2, sz.h/2
			newFrame := func() encoder.SourceFrame420 {
				return encoder.SourceFrame420{
					Y:            make([]byte, sz.w*sz.h),
					U:            make([]byte, cw*ch),
					V:            make([]byte, cw*ch),
					YStride:      sz.w,
					ChromaStride: cw,
					Width:        sz.w,
					Height:       sz.h,
				}
			}
			src1 := newFrame()
			for y := range sz.h {
				for x := range sz.w {
					src1.Y[y*sz.w+x] = uint8((100 + x + y/2 + rng.Intn(10)) & 0xff)
				}
			}
			for i := range src1.U {
				src1.U[i] = uint8(118 + rng.Intn(8))
				src1.V[i] = uint8(106 + rng.Intn(8))
			}
			// Frame 2: frame 1 content with a small uniform drift plus noise —
			// temporally predictable, so residuals stay small.
			src2 := newFrame()
			for i := range src2.Y {
				src2.Y[i] = uint8(min(255, int(src1.Y[i])+3+rng.Intn(3)))
			}
			for i := range src2.U {
				src2.U[i] = uint8(min(255, int(src1.U[i])+2))
				src2.V[i] = uint8(min(255, int(src1.V[i])+2))
			}

			const qIndex = 50
			keyTU, keyRecon, err := encoder.EncodeKeyframe(src1, qIndex)
			if err != nil {
				t.Fatalf("encode keyframe: %v", err)
			}
			pTU, pRecon, err := encoder.EncodePFrame(src2, keyRecon, qIndex)
			if err != nil {
				t.Fatalf("encode p-frame: %v", err)
			}
			t.Logf("key TU %d bytes, P TU %d bytes", len(keyTU), len(pTU))
			if len(pTU) >= len(keyTU) {
				t.Fatalf("P TU %d bytes not smaller than key TU %d bytes", len(pTU), len(keyTU))
			}

			dec, err := goav1.NewDecoder([][]byte{keyTU, pTU})
			if err != nil {
				t.Fatalf("new decoder: %v", err)
			}
			defer dec.Close()
			frames, err := dec.DecodeAll()
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			if len(frames) != 2 {
				t.Fatalf("decoded %d frames, want 2", len(frames))
			}
			f := frames[1]
			comparePlane(t, "Y", f.Y, pRecon.Y, sz.w, sz.h, sz.w)
			comparePlane(t, "U", f.U, pRecon.U, cw, ch, cw)
			comparePlane(t, "V", f.V, pRecon.V, cw, ch, cw)

			psnr := planePSNR(src2.Y, pRecon.Y)
			t.Logf("P-frame luma PSNR(src2, recon) = %.2f dB", psnr)
			if psnr < 30 {
				t.Fatalf("P-frame luma PSNR %.2f dB below sanity floor", psnr)
			}
		})
	}
}
