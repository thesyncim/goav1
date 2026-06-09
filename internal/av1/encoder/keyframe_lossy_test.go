package encoder_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestEncodeKeyframeDecodeMatchesRecon is the non-lossless end-to-end gate:
// the decoder's output must equal the ENCODER'S OWN RECONSTRUCTION bit-for-bit
// (proving headers, mode/tx_type/coefficient coding, contexts, and the
// dequant+inverse recon loop all agree with the decoder), while PSNR against
// the source stays sane for the quantizer strength.
func TestEncodeKeyframeDecodeMatchesRecon(t *testing.T) {
	cases := []struct {
		w, h    int
		qIndex  uint8
		minPSNR float64
	}{
		{64, 64, 50, 30},
		{96, 96, 50, 30},
		{176, 144, 100, 25},
		{128, 64, 200, 18},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("%dx%d-q%d", tc.w, tc.h, tc.qIndex), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(tc.w)*7919 + int64(tc.qIndex)))
			cw, ch := tc.w/2, tc.h/2
			src := encoder.SourceFrame420{
				Y:            make([]byte, tc.w*tc.h),
				U:            make([]byte, cw*ch),
				V:            make([]byte, cw*ch),
				YStride:      tc.w,
				ChromaStride: cw,
				Width:        tc.w,
				Height:       tc.h,
			}
			// Smooth gradient plus mild noise: realistic DC-predictable content.
			for y := range tc.h {
				for x := range tc.w {
					src.Y[y*tc.w+x] = uint8((128 + x + y/2 + rng.Intn(12)) & 0xff)
				}
			}
			for y := range ch {
				for x := range cw {
					src.U[y*cw+x] = uint8(120 + x/4 + rng.Intn(6))
					src.V[y*cw+x] = uint8(110 + y/4 + rng.Intn(6))
				}
			}

			tu, recon, err := encoder.EncodeKeyframe(src, tc.qIndex)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			t.Logf("encoded TU: %d bytes (%.2f bpp)", len(tu), float64(len(tu)*8)/float64(tc.w*tc.h))

			dec, err := goav1.NewDecoder([][]byte{tu})
			if err != nil {
				t.Fatalf("new decoder: %v", err)
			}
			defer dec.Close()
			frames, err := dec.DecodeAll()
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			if len(frames) != 1 {
				t.Fatalf("decoded %d frames, want 1", len(frames))
			}
			f := frames[0]
			comparePlane(t, "Y", f.Y, recon.Y, tc.w, tc.h, tc.w)
			comparePlane(t, "U", f.U, recon.U, cw, ch, cw)
			comparePlane(t, "V", f.V, recon.V, cw, ch, cw)

			psnr := planePSNR(src.Y, recon.Y)
			t.Logf("luma PSNR(src, recon) = %.2f dB", psnr)
			if psnr < tc.minPSNR {
				t.Fatalf("luma PSNR %.2f dB below sanity floor %.2f dB", psnr, tc.minPSNR)
			}
		})
	}
}

func planePSNR(a, b []byte) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var sse float64
	for i := range a {
		d := float64(int(a[i]) - int(b[i]))
		sse += d * d
	}
	if sse == 0 {
		return math.Inf(1)
	}
	mse := sse / float64(len(a))
	return 10 * math.Log10(255*255/mse)
}
