package tile

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// decodeTXBForTest is DecodeCoefs' TXB read without the binding: the
// txb_skip symbol, then decode_coefs at unit dequantization, so out (raster
// order, scan area) receives the signed quantized levels.
func decodeTXBForTest(s *DecodeState, cdfs *CoeffCDFs, size TransformSize, plane CoeffPlaneType, class transform.Class, txbSkipCtx, dcSignCtx uint8, out []int16) TXBDecodeResult {
	geo := &coeffGeometryTable[size]
	clear(out)
	if s.Reader.ReadBinaryCDFUnchecked(&cdfs.TXBSkip[geo.txCtx][txbSkipCtx]) != 0 {
		return TXBDecodeResult{AllZero: true}
	}
	var cf [1024]int32
	var levels [coefLevelsLen]uint8
	c := s.Reader.Cursor()
	r := decodeCoefTokens(&c, cdfs, size, geo, plane, transform.TypeDCTDCT, class, dcSignCtx, &cf, true, &levels, coefDequant{dc: 1, ac: 1, cfMax: 1 << 24}, nil)
	c.CommitStateTo(&s.Reader)
	for i := range out {
		out[i] = int16(cf[i])
	}
	return TXBDecodeResult{EOB: uint16(r.eob + 1), CulLevel: r.culLevel}
}

// TestDecodeCoefTokensDequantizes codes random quantized TXBs with the
// coefficient writer and decodes them through the dav1d decode_coefs port with
// random dequantization factors, with and without quantizer matrices, TXB after
// TXB on one evolving reader/CDF state. Every dequantized coefficient must
// match AV1's dequantization of the written level: (factor * level) & 0xffffff
// shifted by dq_shift and clamped to cf_max, with the factor first weighted
// by the matrix entry, (factor * qm + 16) >> 5. The active row/column extent
// and the coefficient context byte must match the writer's.
func TestDecodeCoefTokensDequantizes(t *testing.T) {
	types := []transform.Type{
		transform.TypeDCTDCT, transform.TypeADSTADST, transform.TypeIDTX,
		transform.TypeVDCT, transform.TypeHDCT, transform.TypeVADST, transform.TypeHFlipADST,
	}
	rng := rand.New(rand.NewSource(1))
	checked := 0
	for size := TransformSize(0); size < transformSizeCount; size++ {
		geo := &coeffGeometryTable[size]
		for _, typ := range types {
			class, _ := typ.Class()
			if class != transform.Class2D && (geo.scanWidth > 16 || geo.scanHeight > 16) {
				continue
			}
			for _, plane := range []CoeffPlaneType{CoeffPlaneY, CoeffPlaneUV} {
				bitDepth := []uint8{8, 10, 12}[rng.Intn(3)]
				checked += runDecodeCoefTokensDequant(t, rng, size, geo, typ, class, plane, bitDepth)
			}
		}
	}
	if checked < 2000 {
		t.Fatalf("only %d TXBs compared", checked)
	}
}

func runDecodeCoefTokensDequant(t *testing.T, rng *rand.Rand, size TransformSize, geo *coeffGeometry, typ transform.Type, class transform.Class, plane CoeffPlaneType, bitDepth uint8) int {
	t.Helper()
	txSize, err := size.TransformSize()
	if err != nil {
		t.Fatal(err)
	}
	scan, levels := coeffScanAndScratch(t, size, txSize, class)
	maxEOB := len(scan)
	const nTXB = 24
	type rec struct {
		coeffs    []int16
		result    TXBDecodeResult
		dcSignCtx uint8
		dc, ac    int32
		qm        []uint16
	}
	recs := make([]rec, nTXB)
	q := uint8(rng.Intn(256))
	var encCDFs CoeffCDFs
	if err := encCDFs.InitDefault(q); err != nil {
		t.Fatal(err)
	}
	w := entropy.NewWriter(make([]byte, 0, 1<<16))
	for n := range recs {
		r := &recs[n]
		r.coeffs = randomCoeffs(rng, scan, maxEOB)
		r.dcSignCtx = uint8(rng.Intn(3))
		r.dc, r.ac = int32(4+rng.Intn(1300)), int32(4+rng.Intn(1800))
		r.dc <<= 2 * ((bitDepth - 8) / 2)
		r.ac <<= 2 * ((bitDepth - 8) / 2)
		if typ < transform.TypeIDTX && rng.Intn(2) == 0 {
			r.qm = QMatrices(uint8(rng.Intn(quantize.QMLevels-1)), plane == CoeffPlaneUV)[size]
		}
		if r.result, err = WriteCoefficientsTXB(&w, &encCDFs, TXBEncodeRequest{
			Size: size, Plane: plane, Class: class, DCSignContext: r.dcSignCtx,
		}, r.coeffs, scan, levels); err != nil {
			t.Fatalf("%v %v txb=%d write: %v", size, typ, n, err)
		}
	}
	buf, err := w.Finish()
	if err != nil {
		t.Fatal(err)
	}

	var decCDFs CoeffCDFs
	if err := decCDFs.InitDefault(q); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset(buf, Job{Offset: 0, Size: uint32(len(buf))}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	_, dqMax := quantize.DequantCoefficientBoundsTrusted(bitDepth)
	txScale := uint8(0)
	if geo.txCtx > 2 {
		txScale = geo.txCtx - 2
	}
	cf := new([1024]int32)
	var coefLevels [coefLevelsLen]uint8
	for n := range recs {
		r := &recs[n]
		if state.Reader.ReadBinaryCDFUnchecked(&decCDFs.TXBSkip[geo.txCtx][0]) != 0 {
			if !r.result.AllZero {
				t.Fatalf("%v %v txb=%d: read as all-zero", size, typ, n)
			}
			continue
		}
		// Stale data: the zero-on-entry contract covers only the scan area.
		for i := range cf {
			cf[i] = int32(rng.Uint32())
		}
		c := state.Reader.Cursor()
		got := decodeCoefTokens(&c, &decCDFs, size, geo, plane, typ, class, r.dcSignCtx, cf, false, &coefLevels, coefDequant{
			dc: uint32(r.dc), ac: uint32(r.ac), shift: txScale, cfMax: uint32(dqMax),
		}, r.qm)
		c.CommitStateTo(&state.Reader)
		if got.eob+1 != int(r.result.EOB) || got.culLevel != r.result.CulLevel {
			t.Fatalf("%v %v txb=%d: eob %d cul %#x, want %d %#x", size, typ, n, got.eob+1, got.culLevel, r.result.EOB, r.result.CulLevel)
		}
		dcOnly := got.eob == 0 && typ == transform.TypeDCTDCT
		height := int(geo.scanHeight)
		activeRows, activeCols := 0, 0
		for pos := range maxEOB {
			want := int64(0)
			if level := int64(r.coeffs[pos]); level != 0 {
				factor := int64(r.ac)
				if pos == 0 {
					factor = int64(r.dc)
				}
				if r.qm != nil {
					factor = (factor*int64(r.qm[pos]) + 16) >> 5
				}
				mag := (factor * max(level, -level) & 0xffffff) >> txScale
				if level < 0 {
					want = -min(mag, int64(dqMax)+1)
				} else {
					want = min(mag, int64(dqMax))
				}
				activeRows = max(activeRows, pos%height+1)
				activeCols = max(activeCols, pos/height+1)
			}
			if dcOnly && pos > 0 {
				continue
			}
			if int64(cf[pos]) != want {
				t.Fatalf("%v %v plane=%d txb=%d eob=%d qm=%v: cf[%d] = %d, want %d", size, typ, plane, n, got.eob, r.qm != nil, pos, cf[pos], want)
			}
		}
		if got.activeRows != activeRows || got.activeCols != activeCols {
			t.Fatalf("%v %v txb=%d: active %dx%d, want %dx%d", size, typ, n, got.activeRows, got.activeCols, activeRows, activeCols)
		}
	}
	return nTXB
}
