package encoder_test

import (
	"fmt"
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestEncodeRepeatPFrameDecodesAsReference is the encoder's first inter-frame
// gate: a KEY + repeat-P two-frame sequence must decode with frame 1 equal to
// the keyframe reconstruction bit for bit. Every block in the P-frame is a
// skipped GLOBALMV LAST block, so any divergence in the inter symbol surface,
// the shared reference-MV stack / mode contexts, the inter header, or the
// reference state fails loudly here.
func TestEncodeRepeatPFrameDecodesAsReference(t *testing.T) {
	sizes := []struct{ w, h int }{{64, 64}, {128, 64}, {128, 128}}
	for _, sz := range sizes {
		t.Run(fmt.Sprintf("%dx%d", sz.w, sz.h), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(sz.w)*31 + int64(sz.h)))
			cw, ch := sz.w/2, sz.h/2
			src := encoder.SourceFrame420{
				Y:            make([]byte, sz.w*sz.h),
				U:            make([]byte, cw*ch),
				V:            make([]byte, cw*ch),
				YStride:      sz.w,
				ChromaStride: cw,
				Width:        sz.w,
				Height:       sz.h,
			}
			for i := range src.Y {
				src.Y[i] = uint8(rng.Intn(256))
			}
			for i := range src.U {
				src.U[i] = uint8(rng.Intn(256))
				src.V[i] = uint8(rng.Intn(256))
			}

			const qIndex = 50
			keyTU, recon, err := encoder.EncodeKeyframe(src, qIndex)
			if err != nil {
				t.Fatalf("encode keyframe: %v", err)
			}
			pTU, err := encoder.EncodeRepeatPFrame(sz.w, sz.h, qIndex)
			if err != nil {
				t.Fatalf("encode p-frame: %v", err)
			}
			t.Logf("key TU %d bytes, P TU %d bytes", len(keyTU), len(pTU))

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
			comparePlane(t, "Y", f.Y, recon.Y, sz.w, sz.h, sz.w)
			comparePlane(t, "U", f.U, recon.U, cw, ch, cw)
			comparePlane(t, "V", f.V, recon.V, cw, ch, cw)
		})
	}
}
