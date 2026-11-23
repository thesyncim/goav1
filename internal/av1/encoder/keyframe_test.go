package encoder_test

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

func TestEncodeLosslessMonochromeKeyframeDecodesBitExact(t *testing.T) {
	sizes := []struct{ w, h int }{
		{64, 64},
		{128, 64},
		{64, 128},
		{96, 96},
		{176, 144},
		{200, 120},
		{72, 72},
	}
	for _, sz := range sizes {
		t.Run(fmt.Sprintf("%dx%d", sz.w, sz.h), func(t *testing.T) {
			rng := rand.New(rand.NewSource(0x400000 + int64(sz.w)*1000 + int64(sz.h)))
			src := encoder.SourceFrameMono{
				Y:       make([]byte, sz.w*sz.h),
				YStride: sz.w,
				Width:   sz.w,
				Height:  sz.h,
			}
			for y := range sz.h {
				for x := range sz.w {
					src.Y[y*sz.w+x] = uint8((x*3 + y*2 + rng.Intn(64)) & 0xff)
				}
			}

			tu, err := encoder.EncodeLosslessMonochromeKeyframe(src)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			t.Logf("encoded monochrome TU: %d bytes", len(tu))

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
			if !f.Format.MonoChrome || f.Format.BitDepth != 8 {
				t.Fatalf("decoded format=%+v, want 8-bit monochrome", f.Format)
			}
			comparePlane(t, "Y", f.Y, src.Y, sz.w, sz.h, sz.w)
			compareAbsentPlane(t, "U", f.U)
			compareAbsentPlane(t, "V", f.V)
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

func compareAbsentPlane(t *testing.T, name string, got goav1.FramePlane) {
	t.Helper()
	if got.Width != 0 || got.Height != 0 || got.Stride != 0 || len(got.Pix) != 0 {
		t.Fatalf("%s plane=%+v, want absent", name, got)
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
