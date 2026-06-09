package encoder_test

import (
	"bytes"
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestEncodeLosslessKeyframe64x64DecodesBitExact is the encoder's first
// end-to-end gate: the emitted temporal unit must decode in the goav1 decoder
// (itself byte-exact against libaom) to a frame that is bit-identical to the
// source. Lossless means any divergence anywhere in the pipeline — headers,
// partition, modes, contexts, prediction, transform, or coefficients — fails
// loudly here.
func TestEncodeLosslessKeyframe64x64DecodesBitExact(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	src := encoder.SourceFrame420{
		Y:            make([]byte, 64*64),
		U:            make([]byte, 32*32),
		V:            make([]byte, 32*32),
		YStride:      64,
		ChromaStride: 32,
		Width:        64,
		Height:       64,
	}
	// A gradient with noise: exercises non-trivial DC predictions, residual
	// signs, golomb tails, and txb-skip choices without being pathological.
	for y := range 64 {
		for x := range 64 {
			src.Y[y*64+x] = uint8((x*2 + y + rng.Intn(32)) & 0xff)
		}
	}
	for y := range 32 {
		for x := range 32 {
			src.U[y*32+x] = uint8((128 + x - y + rng.Intn(16)) & 0xff)
			src.V[y*32+x] = uint8((64 + x + y/2 + rng.Intn(16)) & 0xff)
		}
	}

	tu, err := encoder.EncodeLosslessKeyframe64x64(src)
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
	comparePlane(t, "Y", f.Y, src.Y, 64, 64, 64)
	comparePlane(t, "U", f.U, src.U, 32, 32, 32)
	comparePlane(t, "V", f.V, src.V, 32, 32, 32)
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
