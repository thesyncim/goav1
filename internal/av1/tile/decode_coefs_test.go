package tile

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// coefTestDecoder drives entropy.(*Reader).DecodeCoefs as an inter block at
// qindex > 0: luma transform types are read (the writer codes them through
// coefTestTxtpWriter), chroma types derive from the co-located luma type.
type coefTestDecoder struct {
	tx     TransformTypeCDFs
	ctx    CoeffEntropyContext
	levels [coefLevelsLen]uint8
	e      entropy.CoefState
}

func newCoefTestDecoder(t testing.TB, cdfs *CoeffCDFs, ctx *CoeffEntropyContext) *coefTestDecoder {
	t.Helper()
	d := new(coefTestDecoder)
	if err := d.tx.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if ctx == nil {
		ctx = &d.ctx
	}
	d.e = entropy.CoefState{
		CDF:     (*entropy.CoefCDFs)(cdfs),
		TxIntra: &d.tx.Intra,
		TxInter: &d.tx.Inter,
		A:       &ctx.Above,
		L:       &ctx.Left,
		Levels:  &d.levels,
	}
	return d
}

// decode reads one TXB of the plane block bw4 x bh4. typ is the co-located
// luma type a chroma TXB derives its own from.
func (d *coefTestDecoder) decode(s *DecodeState, size TransformSize, plane uint8, x4, y4 int, bw4, bh4 uint8, typ transform.Type, cf *[1024]int32, dq *CoefDequant) (eob int, txtp transform.Type, rows, cols int) {
	chroma := min(plane, 1)
	d.e.BW4[chroma], d.e.BH4[chroma] = bw4, bh4
	sz := &coefSizeTable[size]
	e, tt, ext := s.Reader.DecodeCoefs(&d.e, uint(plane), sz, cf, dq, uint(x4), uint(y4), uint(sz.W4), uint(sz.H4), uint(typ))
	return e, transform.Type(tt), int(uint8(ext)), int(uint8(ext >> 8))
}

// coefTestTxtpWriter returns the TXBEncodeRequest.AfterSkip hook that codes a
// luma inter transform type as coefTestDecoder reads it (nil for chroma).
func coefTestTxtpWriter(w *entropy.Writer, tx *TransformTypeCDFs, size TransformSize, plane uint8, typ transform.Type) func() error {
	if plane != 0 {
		return nil
	}
	return func() error {
		return WriteInterTransformType(w, tx, InterTransformTypeRequest{Size: size, QIndexKnown: true, QIndex: 1}, typ)
	}
}

// coefTestType is a transform type of class for size whose inter luma set
// allows it.
func coefTestType(t testing.TB, size TransformSize, class transform.Class) (transform.Type, bool) {
	t.Helper()
	typ := transform.TypeDCTDCT
	switch class {
	case transform.ClassHoriz:
		typ = transform.TypeHDCT
	case transform.ClassVert:
		typ = transform.TypeVDCT
	}
	set, err := ExtTXSetTypeFor(size, true, false)
	if err != nil {
		t.Fatal(err)
	}
	ok, err := ExtTXTypeAllowed(set, typ)
	return typ, err == nil && ok
}

// decodeTXBForTest is DecodeCoefs for a TXB at (0, 0) of an equally sized
// plane block with zero a/l contexts (skip and dc_sign contexts 0 for luma),
// at unit dequantization: out (raster order, scan area) receives the signed
// quantized levels.
func decodeTXBForTest(t testing.TB, s *DecodeState, d *coefTestDecoder, size TransformSize, typ transform.Type, out []int16) TXBDecodeResult {
	t.Helper()
	d.ctx = CoeffEntropyContext{}
	sz := &coefSizeTable[size]
	unit := uint32(1) << (max(sz.Ctx, 2) - 2)
	var cf [1024]int32
	eob, got, _, _ := d.decode(s, size, 0, 0, 0, sz.W4, sz.H4, 0, &cf, &CoefDequant{DC: unit, AC: unit, Max: 1 << 24})
	clear(out)
	if eob < 0 {
		return TXBDecodeResult{AllZero: true}
	}
	if got != typ {
		t.Fatalf("size %v: decoded txtp %v, want %v", size, got, typ)
	}
	for i := range out {
		out[i] = int16(cf[i])
	}
	return TXBDecodeResult{EOB: uint16(eob + 1)}
}

// TestDecodeCoefTokensDequantizes codes random quantized TXBs with the
// coefficient writer and decodes them through the dav1d decode_coefs port with
// random dequantization factors, with and without quantizer matrices, TXB after
// TXB on one evolving reader/CDF state, in both plane types and with every
// dc_sign context. Every dequantized coefficient must match AV1's
// dequantization of the written level: (factor * level) & 0xffffff shifted by
// dq_shift and clamped to cf_max, with the factor first weighted by the matrix
// entry, (factor * qm + 16) >> 5. The active row/column extent and the a/l
// coefficient context byte must match the writer's.
func TestDecodeCoefTokensDequantizes(t *testing.T) {
	types := []transform.Type{
		transform.TypeDCTDCT, transform.TypeADSTADST, transform.TypeIDTX,
		transform.TypeVDCT, transform.TypeHDCT, transform.TypeVADST, transform.TypeHFlipADST,
	}
	rng := rand.New(rand.NewSource(1))
	checked := 0
	for size := TransformSize(0); size < transformSizeCount; size++ {
		dims := &transformDimensions[size]
		for _, ytyp := range types {
			for _, plane := range []uint8{0, 1} {
				typ := ytyp
				if plane != 0 {
					if dims.Max == 4 {
						continue // chroma transforms are at most 32x32
					}
					typ = transform.Type(entropy.CoefUVInterTxtp(dims.Max, dims.Min, uint8(ytyp)))
				} else if dims.Max+1 > 4 {
					if ytyp != transform.TypeDCTDCT {
						continue // 64-sample sides code DCT_DCT only
					}
				} else if set, err := ExtTXSetTypeFor(size, true, false); err != nil {
					t.Fatal(err)
				} else if ok, err := ExtTXTypeAllowed(set, typ); err != nil || !ok {
					continue
				}
				bitDepth := []uint8{8, 10, 12}[rng.Intn(3)]
				checked += runDecodeCoefTokensDequant(t, rng, size, ytyp, typ, plane, bitDepth)
			}
		}
	}
	if checked < 2000 {
		t.Fatalf("only %d TXBs compared", checked)
	}
}

func runDecodeCoefTokensDequant(t *testing.T, rng *rand.Rand, size TransformSize, ytyp, typ transform.Type, plane uint8, bitDepth uint8) int {
	t.Helper()
	txSize, err := size.TransformSize()
	if err != nil {
		t.Fatal(err)
	}
	class, err := typ.Class()
	if err != nil {
		t.Fatal(err)
	}
	geo := &coeffGeometryTable[size]
	scan, levels := coeffScanAndScratch(t, size, txSize, class)
	maxEOB := len(scan)
	planeType := CoeffPlaneY
	if plane != 0 {
		planeType = CoeffPlaneUV
	}
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
	var encTx TransformTypeCDFs
	if err := encTx.InitDefault(); err != nil {
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
			r.qm = QMatrices(uint8(rng.Intn(quantize.QMLevels-1)), plane != 0)[size]
		}
		skipCtx := uint8(0)
		if plane != 0 {
			skipCtx = 7 // chroma, block == TXB, left context zero
			if r.dcSignCtx != 0 {
				skipCtx++ // the above context carries the dc_sign flag
			}
		}
		if r.result, err = WriteCoefficientsTXB(&w, &encCDFs, TXBEncodeRequest{
			Size: size, Plane: planeType, Class: class, TXBSkipContext: skipCtx, DCSignContext: r.dcSignCtx,
			AfterSkip: coefTestTxtpWriter(&w, &encTx, size, plane, typ),
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
	d := newCoefTestDecoder(t, &decCDFs, nil)
	var state DecodeState
	if err := state.Reset(buf, Job{Offset: 0, Size: uint32(len(buf))}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	_, dqMax := quantize.DequantCoefficientBoundsTrusted(bitDepth)
	txScale := uint8(0)
	if geo.txCtx > 2 {
		txScale = geo.txCtx - 2
	}
	dims := &transformDimensions[size]
	cf := new([1024]int32)
	for n := range recs {
		r := &recs[n]
		// dc_sign context r.dcSignCtx: 1 for a negative, 2 for a positive
		// above neighbour.
		d.ctx = CoeffEntropyContext{}
		d.ctx.Above[plane][0] = [3]uint8{0, 1 << CoeffContextBits, 2 << CoeffContextBits}[r.dcSignCtx]
		clear(cf[:])
		var qm *CoefQMatrices
		if r.qm != nil {
			qm = new(CoefQMatrices)
			qm[size] = r.qm
		}
		eob, gotTyp, rows, cols := d.decode(&state, size, plane, 0, 0, dims.W4, dims.H4, ytyp, cf, &CoefDequant{
			DC: uint32(r.dc), AC: uint32(r.ac), Max: uint32(dqMax), QM: qm,
		})
		if eob < 0 {
			if !r.result.AllZero {
				t.Fatalf("%v %v txb=%d: read as all-zero", size, typ, n)
			}
			if d.ctx.Above[plane][0] != 0 {
				t.Fatalf("%v %v txb=%d: all-zero context %#x", size, typ, n, d.ctx.Above[plane][0])
			}
			continue
		}
		if gotTyp != typ {
			t.Fatalf("%v plane=%d txb=%d: txtp %v, want %v", size, plane, n, gotTyp, typ)
		}
		if eob+1 != int(r.result.EOB) || d.ctx.Above[plane][0] != r.result.CulLevel || d.ctx.Left[plane][0] != r.result.CulLevel {
			t.Fatalf("%v %v txb=%d: eob %d ctx %#x, want %d %#x", size, typ, n, eob+1, d.ctx.Above[plane][0], r.result.EOB, r.result.CulLevel)
		}
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
			if int64(cf[pos]) != want {
				t.Fatalf("%v %v plane=%d txb=%d eob=%d qm=%v: cf[%d] = %d, want %d", size, typ, plane, n, eob, r.qm != nil, pos, cf[pos], want)
			}
		}
		if rows != activeRows || cols != activeCols {
			t.Fatalf("%v %v txb=%d: active %dx%d, want %dx%d", size, typ, n, rows, cols, activeRows, activeCols)
		}
	}
	return nTXB
}
