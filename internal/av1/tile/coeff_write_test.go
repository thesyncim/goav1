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

func TestWriteCoefficientsTXB8x8Y2DTrustedMatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewSource(133))
	txSize, err := TransformSize8x8.TransformSize()
	if err != nil {
		t.Fatal(err)
	}
	scan, genericScratch := coeffScanAndScratch(t, TransformSize8x8, txSize, transform.Class2D)
	fastScratch := make([]uint8, len(genericScratch))

	for _, withAfterSkip := range []bool{false, true} {
		var genericCDFs, fastCDFs CoeffCDFs
		if err := genericCDFs.InitDefault(96); err != nil {
			t.Fatal(err)
		}
		if err := fastCDFs.InitDefault(96); err != nil {
			t.Fatal(err)
		}
		genericWriter := entropy.NewWriter(make([]byte, 0, 1<<16))
		fastWriter := entropy.NewWriter(make([]byte, 0, 1<<16))
		var genericTXCDF, fastTXCDF entropy.CDF
		if err := genericTXCDF.InitUniform(16); err != nil {
			t.Fatal(err)
		}
		fastTXCDF = genericTXCDF

		for attempt := range 256 {
			coeffs := randomCoeffs(rng, scan, len(scan))
			var genericAfterSkip func() error
			var fastTXCDFPtr *entropy.CDF
			fastTXSymbol := 0
			if withAfterSkip {
				symbol := attempt & 15
				genericAfterSkip = func() error {
					saved := genericTXCDF
					genericWriter.WriteCDF(&genericTXCDF, symbol)
					genericTXCDF = saved
					return nil
				}
				fastTXCDFPtr = &fastTXCDF
				fastTXSymbol = symbol
			}

			genericResult, err := WriteCoefficientsTXB(&genericWriter, &genericCDFs, TXBEncodeRequest{
				Size: TransformSize8x8, Plane: CoeffPlaneY, Class: transform.Class2D, AfterSkip: genericAfterSkip,
			}, coeffs, scan, genericScratch)
			if err != nil {
				t.Fatalf("generic attempt %d withAfterSkip=%t: %v", attempt, withAfterSkip, err)
			}
			fastResult := WriteCoefficientsTXB8x8Y2DTrusted(&fastWriter, &fastCDFs, coeffs, fastScratch, fastTXCDFPtr, fastTXSymbol)
			if fastResult != genericResult {
				t.Fatalf("attempt %d withAfterSkip=%t result=%+v want %+v", attempt, withAfterSkip, fastResult, genericResult)
			}
			if fastCDFs != genericCDFs {
				t.Fatalf("attempt %d withAfterSkip=%t coefficient CDFs diverged at %s", attempt, withAfterSkip, firstCoeffCDFDiff(fastCDFs, genericCDFs))
			}
			if fastTXCDF != genericTXCDF {
				t.Fatalf("attempt %d withAfterSkip=%t tx CDFs diverged", attempt, withAfterSkip)
			}
			if fastWriter.Tell() != genericWriter.Tell() {
				t.Fatalf("attempt %d withAfterSkip=%t tell=%d want %d", attempt, withAfterSkip, fastWriter.Tell(), genericWriter.Tell())
			}
		}

		genericBytes, err := genericWriter.Finish()
		if err != nil {
			t.Fatalf("generic finish withAfterSkip=%t: %v", withAfterSkip, err)
		}
		fastBytes, err := fastWriter.Finish()
		if err != nil {
			t.Fatalf("fast finish withAfterSkip=%t: %v", withAfterSkip, err)
		}
		if string(fastBytes) != string(genericBytes) {
			t.Fatalf("withAfterSkip=%t final bytes diverged", withAfterSkip)
		}
	}
}

func TestWriteCoefficientsTXB8x8Y2DContextTrustedMatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewSource(177))
	txSize, err := TransformSize8x8.TransformSize()
	if err != nil {
		t.Fatal(err)
	}
	scan, genericScratch := coeffScanAndScratch(t, TransformSize8x8, txSize, transform.Class2D)
	fastScratch := make([]uint8, len(genericScratch))

	var genericCDFs, fastCDFs CoeffCDFs
	if err := genericCDFs.InitDefault(96); err != nil {
		t.Fatal(err)
	}
	if err := fastCDFs.InitDefault(96); err != nil {
		t.Fatal(err)
	}
	genericWriter := entropy.NewWriter(make([]byte, 0, 1<<16))
	fastWriter := entropy.NewWriter(make([]byte, 0, 1<<16))
	var genericTXCDF, fastTXCDF entropy.CDF
	if err := genericTXCDF.InitUniform(16); err != nil {
		t.Fatal(err)
	}
	fastTXCDF = genericTXCDF

	for attempt := range 256 {
		coeffs := randomCoeffs(rng, scan, len(scan))
		txbCtx := uint8(rng.Intn(TXBSkipContexts))
		dcCtx := uint8(rng.Intn(3))
		symbol := rng.Intn(16)
		genericAfterSkip := func() error {
			genericWriter.WriteCDF(&genericTXCDF, symbol)
			return nil
		}

		genericResult, err := WriteCoefficientsTXB(&genericWriter, &genericCDFs, TXBEncodeRequest{
			Size:           TransformSize8x8,
			Plane:          CoeffPlaneY,
			Class:          transform.Class2D,
			TXBSkipContext: txbCtx,
			DCSignContext:  dcCtx,
			AfterSkip:      genericAfterSkip,
		}, coeffs, scan, genericScratch)
		if err != nil {
			t.Fatalf("generic attempt %d: %v", attempt, err)
		}
		fastResult := WriteCoefficientsTXB8x8Y2DContextTrusted(&fastWriter, &fastCDFs, coeffs, fastScratch, txbCtx, dcCtx, &fastTXCDF, symbol)
		if fastResult != genericResult {
			t.Fatalf("attempt %d result=%+v want %+v", attempt, fastResult, genericResult)
		}
		if fastCDFs != genericCDFs {
			t.Fatalf("attempt %d coefficient CDFs diverged at %s", attempt, firstCoeffCDFDiff(fastCDFs, genericCDFs))
		}
		if fastTXCDF != genericTXCDF {
			t.Fatalf("attempt %d tx CDF diverged", attempt)
		}
		if fastWriter.Tell() != genericWriter.Tell() {
			t.Fatalf("attempt %d tell=%d want %d", attempt, fastWriter.Tell(), genericWriter.Tell())
		}
	}

	genericBytes, err := genericWriter.Finish()
	if err != nil {
		t.Fatalf("generic finish: %v", err)
	}
	fastBytes, err := fastWriter.Finish()
	if err != nil {
		t.Fatalf("fast finish: %v", err)
	}
	if string(fastBytes) != string(genericBytes) {
		t.Fatal("final bytes diverged")
	}
}

func firstCoeffCDFDiff(a, b CoeffCDFs) string {
	for tx := range a.TXBSkip {
		for ctx := range a.TXBSkip[tx] {
			if a.TXBSkip[tx][ctx] != b.TXBSkip[tx][ctx] {
				return "TXBSkip"
			}
		}
	}
	for tx := range a.EOBExtra {
		for plane := range a.EOBExtra[tx] {
			for ctx := range a.EOBExtra[tx][plane] {
				if a.EOBExtra[tx][plane][ctx] != b.EOBExtra[tx][plane][ctx] {
					return "EOBExtra"
				}
			}
		}
	}
	for plane := range a.DCSign {
		for ctx := range a.DCSign[plane] {
			if a.DCSign[plane][ctx] != b.DCSign[plane][ctx] {
				return "DCSign"
			}
		}
	}
	for tx := range a.CoeffBR {
		for plane := range a.CoeffBR[tx] {
			for ctx := range a.CoeffBR[tx][plane] {
				if a.CoeffBR[tx][plane][ctx] != b.CoeffBR[tx][plane][ctx] {
					return "CoeffBR"
				}
			}
		}
	}
	for tx := range a.CoeffBase {
		for plane := range a.CoeffBase[tx] {
			for ctx := range a.CoeffBase[tx][plane] {
				if a.CoeffBase[tx][plane][ctx] != b.CoeffBase[tx][plane][ctx] {
					return "CoeffBase"
				}
			}
		}
	}
	for tx := range a.CoeffBaseEOB {
		for plane := range a.CoeffBaseEOB[tx] {
			for ctx := range a.CoeffBaseEOB[tx][plane] {
				if a.CoeffBaseEOB[tx][plane][ctx] != b.CoeffBaseEOB[tx][plane][ctx] {
					return "CoeffBaseEOB"
				}
			}
		}
	}
	for plane := range a.EOBFlag16 {
		for ctx := range a.EOBFlag16[plane] {
			if a.EOBFlag16[plane][ctx] != b.EOBFlag16[plane][ctx] {
				return "EOBFlag16"
			}
		}
	}
	for plane := range a.EOBFlag32 {
		for ctx := range a.EOBFlag32[plane] {
			if a.EOBFlag32[plane][ctx] != b.EOBFlag32[plane][ctx] {
				return "EOBFlag32"
			}
		}
	}
	for plane := range a.EOBFlag64 {
		for ctx := range a.EOBFlag64[plane] {
			if a.EOBFlag64[plane][ctx] != b.EOBFlag64[plane][ctx] {
				return "EOBFlag64"
			}
		}
	}
	for plane := range a.EOBFlag128 {
		for ctx := range a.EOBFlag128[plane] {
			if a.EOBFlag128[plane][ctx] != b.EOBFlag128[plane][ctx] {
				return "EOBFlag128"
			}
		}
	}
	for plane := range a.EOBFlag256 {
		for ctx := range a.EOBFlag256[plane] {
			if a.EOBFlag256[plane][ctx] != b.EOBFlag256[plane][ctx] {
				return "EOBFlag256"
			}
		}
	}
	for plane := range a.EOBFlag512 {
		for ctx := range a.EOBFlag512[plane] {
			if a.EOBFlag512[plane][ctx] != b.EOBFlag512[plane][ctx] {
				return "EOBFlag512"
			}
		}
	}
	for plane := range a.EOBFlag1024 {
		for ctx := range a.EOBFlag1024[plane] {
			if a.EOBFlag1024[plane][ctx] != b.EOBFlag1024[plane][ctx] {
				return "EOBFlag1024"
			}
		}
	}
	return "unknown"
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

func BenchmarkWriteCoefficientsTXB8x8Y2D(b *testing.B) {
	rng := rand.New(rand.NewSource(233))
	txSize, err := TransformSize8x8.TransformSize()
	if err != nil {
		b.Fatal(err)
	}
	scan, scratch := coeffScanAndScratch(b, TransformSize8x8, txSize, transform.Class2D)
	blocks := make([][]int16, 256)
	for i := range blocks {
		blocks[i] = randomCoeffs(rng, scan, len(scan))
	}

	b.Run("generic", func(b *testing.B) {
		var cdfs CoeffCDFs
		if err := cdfs.InitDefault(96); err != nil {
			b.Fatal(err)
		}
		buf := make([]byte, 0, 4096)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			w := entropy.NewWriter(buf[:0])
			if _, err := WriteCoefficientsTXB(&w, &cdfs, TXBEncodeRequest{
				Size: TransformSize8x8, Plane: CoeffPlaneY, Class: transform.Class2D,
			}, blocks[i&255], scan, scratch); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("trusted", func(b *testing.B) {
		var cdfs CoeffCDFs
		if err := cdfs.InitDefault(96); err != nil {
			b.Fatal(err)
		}
		buf := make([]byte, 0, 4096)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			w := entropy.NewWriter(buf[:0])
			WriteCoefficientsTXB8x8Y2DTrusted(&w, &cdfs, blocks[i&255], scratch, nil, 0)
		}
	})
}
