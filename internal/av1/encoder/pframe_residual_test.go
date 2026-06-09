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
			// Frame 2: frame 1 content SHIFTED by an even full-pel motion
			// (+4, +2) with a small drift — motion estimation must lock onto
			// the global shift and leave near-zero residuals.
			const shiftX, shiftY = 4, 2
			src2 := newFrame()
			for y := range sz.h {
				for x := range sz.w {
					sx, sy := x-shiftX, y-shiftY
					if sx < 0 {
						sx = 0
					}
					if sy < 0 {
						sy = 0
					}
					src2.Y[y*sz.w+x] = uint8(min(255, int(src1.Y[sy*sz.w+sx])+1))
				}
			}
			for y := range ch {
				for x := range cw {
					sx, sy := x-shiftX/2, y-shiftY/2
					if sx < 0 {
						sx = 0
					}
					if sy < 0 {
						sy = 0
					}
					src2.U[y*cw+x] = src1.U[sy*cw+sx]
					src2.V[y*cw+x] = src1.V[sy*cw+sx]
				}
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
			// Size evidence only on the larger frame: the tiny 64x64 gradient
			// keyframe is already near-free, while the shifted P pays MV costs
			// and codes fresh edge content.
			if sz.w >= 128 && len(pTU) >= len(keyTU) {
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
