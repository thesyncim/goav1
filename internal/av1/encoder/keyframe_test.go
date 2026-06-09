package encoder_test

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestEncodeLosslessKeyframeDecodesBitExact is the encoder's end-to-end gate:
// the emitted temporal unit must decode in the goav1 decoder (itself byte-exact
// against libaom) to a frame bit-identical to the source. Lossless means any
// divergence anywhere in the pipeline — headers, partition, modes, contexts,
// prediction, transform, or coefficients — fails loudly here. Multi-superblock
// sizes exercise the cross-SB context carry on both axes.
func TestEncodeLosslessKeyframeDecodesBitExact(t *testing.T) {
	sizes := []struct{ w, h int }{
		{64, 64},   // single superblock
		{128, 64},  // horizontal carry
		{64, 128},  // vertical carry (left-context reset per SB row)
		{192, 128}, // both axes, 3x2 superblocks
	}
	for _, sz := range sizes {
		t.Run(fmt.Sprintf("%dx%d", sz.w, sz.h), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(sz.w)*1000 + int64(sz.h)))
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
			// A gradient with noise: exercises non-trivial DC predictions,
			// residual signs, golomb tails, and txb-skip choices.
			for y := range sz.h {
				for x := range sz.w {
					src.Y[y*sz.w+x] = uint8((x*2 + y + rng.Intn(32)) & 0xff)
				}
			}
			for y := range ch {
				for x := range cw {
					src.U[y*cw+x] = uint8((128 + x - y + rng.Intn(16)) & 0xff)
					src.V[y*cw+x] = uint8((64 + x + y/2 + rng.Intn(16)) & 0xff)
				}
			}

			tu, err := encoder.EncodeLosslessKeyframe(src)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			t.Logf("encoded TU: %d bytes", len(tu))

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
			comparePlane(t, "Y", f.Y, src.Y, sz.w, sz.h, sz.w)
			comparePlane(t, "U", f.U, src.U, cw, ch, cw)
			comparePlane(t, "V", f.V, src.V, cw, ch, cw)
		})
	}
}

func comparePlane(t *testing.T, name string, got goav1.FramePlane, want []byte, w, h, wantStride int) {
	t.Helper()
	if got.Width != w || got.Height != h {
		t.Fatalf("%s plane %dx%d, want %dx%d", name, got.Width, got.Height, w, h)
	}
	for row := range h {
		gotRow := got.Pix[row*got.Stride : row*got.Stride+w]
		wantRow := want[row*wantStride : row*wantStride+w]
		if !bytes.Equal(gotRow, wantRow) {
			col := firstDiff(gotRow, wantRow)
			t.Fatalf("%s plane mismatch at (%d,%d): got %d want %d", name, col, row, gotRow[col], wantRow[col])
		}
	}
}

func firstDiff(a, b []byte) int {
	n := min(len(a), len(b))
	for i := range n {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}
