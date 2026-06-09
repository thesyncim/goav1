package tile

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// TestWriteCoefficientsTXBRoundTrip is the oracle gate for the coefficient writer:
// it encodes a sequence of random quantized transform blocks with shared, adapting
// CDFs and decodes them back with ReadCoefficientsTXB (the byte-exact libaom
// inverse) using CDFs that start identical and adapt in lockstep. Every block must
// reconstruct its coefficients and eob exactly. Covered: all tx classes, square and
// rectangular sizes, all-zero (txb_skip) blocks, zero runs, base levels, base-range
// extension, golomb tails, and both signs.
func TestWriteCoefficientsTXBRoundTrip(t *testing.T) {
	sizes := []TransformSize{
		TransformSize4x4, TransformSize8x8, TransformSize16x16, TransformSize32x32,
		TransformSize4x8, TransformSize8x4, TransformSize8x16, TransformSize16x8,
		TransformSize4x16, TransformSize16x4,
	}
	classes := []transform.Class{transform.Class2D, transform.ClassHoriz, transform.ClassVert}
	const q = 0
	rng := rand.New(rand.NewSource(7))

	for _, size := range sizes {
		txSize, err := size.TransformSize()
		if err != nil {
			t.Fatalf("size %v: %v", size, err)
		}
		for _, class := range classes {
			scan, encScratch := coeffScanAndScratch(t, size, txSize, class)
			maxEOB := len(scan)
			eobMultiCtx := uint8(0)
			if class != transform.Class2D {
				eobMultiCtx = 1
			}

			const nTXB = 50
			recs := make([][]int16, nTXB)
			var encCDFs CoeffCDFs
			if err := encCDFs.InitDefault(q); err != nil {
				t.Fatal(err)
			}
			w := entropy.NewWriter(make([]byte, 0, 1<<16))
			for n := range nTXB {
				coeffs := randomCoeffs(rng, scan, maxEOB)
				recs[n] = coeffs
				if err := WriteCoefficientsTXB(&w, &encCDFs, TXBEncodeRequest{
					Size: size, Plane: CoeffPlaneY, Class: class,
				}, coeffs, scan, encScratch); err != nil {
					t.Fatalf("size=%v class=%v txb=%d write: %v", size, class, n, err)
				}
			}
			buf, err := w.Finish()
			if err != nil {
				t.Fatalf("size=%v class=%v finish: %v", size, class, err)
			}

			var decCDFs CoeffCDFs
			if err := decCDFs.InitDefault(q); err != nil {
				t.Fatal(err)
			}
			var state DecodeState
			if err := state.Reset(buf, Job{Offset: 0, Size: uint32(len(buf))}, DecodeOptions{}); err != nil {
				t.Fatal(err)
			}
			decScan, decScratch := coeffScanAndScratch(t, size, txSize, class)
			for n := range nTXB {
				out := make([]int16, maxEOB)
				result, err := state.ReadCoefficientsTXB(&decCDFs, TXBDecodeRequest{
					Size: size, Plane: CoeffPlaneY, Class: class, EOBMultiContext: eobMultiCtx,
				}, out, decScan, decScratch)
				if err != nil {
					t.Fatalf("size=%v class=%v txb=%d read: %v", size, class, n, err)
				}
				want := recs[n]
				wantEOB := 0
				for c := range maxEOB {
					if want[int(scan[c])] != 0 {
						wantEOB = c + 1
					}
				}
				if int(result.EOB) != wantEOB {
					t.Fatalf("size=%v class=%v txb=%d eob=%d want %d", size, class, n, result.EOB, wantEOB)
				}
				for i := range maxEOB {
					if out[i] != want[i] {
						t.Fatalf("size=%v class=%v txb=%d coeff[%d]=%d want %d", size, class, n, i, out[i], want[i])
					}
				}
			}
		}
	}
}

// randomCoeffs builds a random quantized coefficient block in raster order with a
// random eob, exercising zero runs, base levels (1..3), base-range extension
// (3..14), golomb tails (>=15), and both signs. Magnitudes stay below 256 so the
// level-context buffer (clamped to INT8_MAX) never wraps.
func randomCoeffs(rng *rand.Rand, scan []int16, maxEOB int) []int16 {
	coeffs := make([]int16, maxEOB)
	if rng.Intn(6) == 0 { // some all-zero (txb_skip) blocks
		return coeffs
	}
	eob := 1 + rng.Intn(maxEOB)
	for c := range eob {
		level := 0
		switch r := rng.Intn(10); {
		case r < 4 && c != eob-1:
			level = 0 // zero (the eob coefficient itself must be non-zero)
		case r < 8:
			level = 1 + rng.Intn(3)
		case r < 9:
			level = 3 + rng.Intn(12)
		default:
			level = 15 + rng.Intn(200)
		}
		if c == eob-1 && level == 0 {
			level = 1
		}
		if level != 0 && rng.Intn(2) == 0 {
			level = -level
		}
		coeffs[int(scan[c])] = int16(level)
	}
	return coeffs
}
