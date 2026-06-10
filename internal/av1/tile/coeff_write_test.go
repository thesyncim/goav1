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
				if _, err := WriteCoefficientsTXB(&w, &encCDFs, TXBEncodeRequest{
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

// TestWriteCoefficientsTXBWithContextRoundTrip drives the carrier path: a
// sequence of TXBs at random plane positions coded through the encoder-side
// CoeffEntropyContext carrier must decode exactly through the decoder's
// ReadCoefficientsTXBWithContext, with both carriers (txb_skip/dc_sign context
// derivation and MarkTXB updates) and both CDF sets evolving in lockstep.
func TestWriteCoefficientsTXBWithContextRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	sizes := []TransformSize{
		TransformSize4x4, TransformSize8x8, TransformSize16x16,
		TransformSize4x8, TransformSize8x4, TransformSize16x8,
	}
	classes := []transform.Class{transform.Class2D, transform.ClassHoriz, transform.ClassVert}
	const q = 0

	type rec struct {
		ctxReq  CoeffContextRequest
		class   transform.Class
		coeffs  []int16
		result  TXBDecodeResult
		carrier CoeffEntropyContext // encoder carrier snapshot after this TXB
	}

	var encCDFs CoeffCDFs
	if err := encCDFs.InitDefault(q); err != nil {
		t.Fatal(err)
	}
	var encCarrier CoeffEntropyContext
	w := entropy.NewWriter(make([]byte, 0, 1<<17))

	const n = 400
	recs := make([]rec, 0, n)
	for range n {
		size := sizes[rng.Intn(len(sizes))]
		class := classes[rng.Intn(len(classes))]
		txSize, err := size.TransformSize()
		if err != nil {
			t.Fatal(err)
		}
		txDims, ok := size.Dimensions()
		if !ok {
			t.Fatalf("no dims for %v", size)
		}
		scan, scratch := coeffScanAndScratch(t, size, txSize, class)
		maxEOB := len(scan)
		plane := uint8(rng.Intn(3))
		planeBlock := BlockSize128x128 // larger than any tx: exercises the block!=tx contexts
		if rng.Intn(3) == 0 {
			planeBlock = BlockSize4x4 // equal/smaller: exercises the equal-size context path
		}
		ctxReq := CoeffContextRequest{
			Plane:      plane,
			PlaneBlock: planeBlock,
			Size:       size,
			X4:         uint8(rng.Intn(MaxBlockModeSlots - int(txDims.W4) + 1)),
			Y4:         uint8(rng.Intn(MaxBlockModeSlots - int(txDims.H4) + 1)),
		}
		coeffs := randomCoeffs(rng, scan, maxEOB)
		result, err := WriteCoefficientsTXBWithContext(&w, &encCDFs, &encCarrier, ctxReq, class, coeffs, scan, scratch)
		if err != nil {
			t.Fatalf("write with context: %v", err)
		}
		recs = append(recs, rec{ctxReq: ctxReq, class: class, coeffs: coeffs, result: result, carrier: encCarrier})
	}
	buf, err := w.Finish()
	if err != nil {
		t.Fatal(err)
	}

	var decCDFs CoeffCDFs
	if err := decCDFs.InitDefault(q); err != nil {
		t.Fatal(err)
	}
	var decCarrier CoeffEntropyContext
	var state DecodeState
	if err := state.Reset(buf, Job{Offset: 0, Size: uint32(len(buf))}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	for i, r := range recs {
		txSize, err := r.ctxReq.Size.TransformSize()
		if err != nil {
			t.Fatal(err)
		}
		scan, scratch := coeffScanAndScratch(t, r.ctxReq.Size, txSize, r.class)
		out := make([]int16, len(scan))
		eobMultiCtx := uint8(0)
		if r.class != transform.Class2D {
			eobMultiCtx = 1
		}
		result, err := state.ReadCoefficientsTXBWithContext(&decCDFs, &decCarrier, r.ctxReq, TXBDecodeRequest{
			Class:           r.class,
			EOBMultiContext: eobMultiCtx,
		}, out, scan, scratch)
		if err != nil {
			t.Fatalf("rec %d read with context: %v", i, err)
		}
		if result.AllZero != r.result.AllZero || result.EOB != r.result.EOB ||
			result.CulLevel != r.result.CulLevel || result.MaxScanLine != r.result.MaxScanLine {
			t.Fatalf("rec %d result=%+v want %+v", i, result, r.result)
		}
		for p := range out {
			if out[p] != r.coeffs[p] {
				t.Fatalf("rec %d coeff[%d]=%d want %d", i, p, out[p], r.coeffs[p])
			}
		}
		if decCarrier != r.carrier {
			t.Fatalf("rec %d carrier state diverged from encoder snapshot", i)
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
