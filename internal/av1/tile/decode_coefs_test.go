package tile

import (
	"math/rand/v2"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// TestDecodeCoefTokensMatchesQuantizedReader decodes the same random payloads
// through the quantized TXB reader (followed by scalar dequantization) and
// through the dav1d decode_coefs port, TXB after TXB on one evolving
// reader/CDF state, and requires identical range-decoder state, CDFs,
// dequantized coefficients, EOB and coefficient context.
func TestDecodeCoefTokensMatchesQuantizedReader(t *testing.T) {
	types := []transform.Type{
		transform.TypeDCTDCT, transform.TypeADSTADST, transform.TypeIDTX,
		transform.TypeVDCT, transform.TypeHDCT, transform.TypeVADST, transform.TypeHFlipADST,
	}
	bitDepths := []uint8{8, 10, 12}
	rng := rand.New(rand.NewPCG(1, 2))
	checked := 0
	for size := TransformSize(0); size < transformSizeCount; size++ {
		geo, ok := coeffGeoPtr(size)
		if !ok {
			continue
		}
		for _, typ := range types {
			class, _ := typ.Class()
			if class != transform.Class2D && (geo.scanWidth > 16 || geo.scanHeight > 16) {
				continue
			}
			for _, disableUpdate := range []bool{false, true} {
				for _, plane := range []CoeffPlaneType{CoeffPlaneY, CoeffPlaneUV} {
					bitDepth := bitDepths[rng.IntN(len(bitDepths))]
					checked += runDecodeCoefTokensDiff(t, rng, size, geo, typ, class, plane, disableUpdate, bitDepth)
				}
			}
		}
	}
	if checked < 2000 {
		t.Fatalf("only %d TXBs compared", checked)
	}
}

func runDecodeCoefTokensDiff(t *testing.T, rng *rand.Rand, size TransformSize, geo *coeffGeometry, typ transform.Type, class transform.Class, plane CoeffPlaneType, disableUpdate bool, bitDepth uint8) int {
	t.Helper()
	payload := make([]byte, 4096)
	for i := range payload {
		payload[i] = byte(rng.Uint32())
	}
	// Skew some payloads toward zero bytes so long Golomb prefixes and large
	// tokens show up.
	if rng.IntN(4) == 0 {
		for i := range payload {
			if rng.IntN(3) == 0 {
				payload[i] = 0
			}
		}
	}
	q := uint8(rng.IntN(256))
	var oldCDFs, newCDFs CoeffCDFs
	if err := oldCDFs.InitDefault(q); err != nil {
		t.Fatal(err)
	}
	newCDFs = oldCDFs
	opts := DecodeOptions{DisableCDFUpdate: disableUpdate}
	var oldState, newState DecodeState
	job := Job{Offset: 0, Size: uint32(len(payload))}
	if err := oldState.Reset(payload, job, opts); err != nil {
		t.Fatal(err)
	}
	if err := newState.Reset(payload, job, opts); err != nil {
		t.Fatal(err)
	}
	_, dqMax := quantize.DequantCoefficientBoundsTrusted(bitDepth)
	txScale := uint8(0)
	if geo.txCtx > 2 {
		txScale = geo.txCtx - 2
	}
	maxEOB := int(geo.maxEOB)
	scan := coeffScanTable[size][class]
	coeffs := make([]int16, maxEOB)
	levels := make([]uint8, geo.scratchLen)
	cf := new([1024]int32)
	var coefLevels [coefLevelsLen]uint8
	compared := 0
	for txb := 0; txb < 40; txb++ {
		dcFactor := int32(4 + rng.IntN(1300))
		acFactor := int32(4 + rng.IntN(1800))
		if bitDepth > 8 {
			dcFactor *= 4
			acFactor *= 4
		}
		if bitDepth > 10 {
			dcFactor *= 4
			acFactor *= 4
		}
		dcSignCtx := uint8(rng.IntN(3))
		req := TXBDecodeRequest{
			Size:            size,
			Plane:           plane,
			Class:           class,
			DCSignContext:   dcSignCtx,
			EOBMultiContext: eobMultiContextForClass(class),
			TXBSkipKnown:    true,
		}
		oldResult, err := oldState.ReadCoefficientsTXB(&oldCDFs, req, coeffs, scan, levels)
		if err != nil {
			// The quantized reader rejects levels beyond int16 and long
			// Golomb tails; dav1d decodes those, so stop this stream here.
			return compared
		}
		// Stale data: the zero-on-entry contract covers only the scan area.
		for i := range cf {
			cf[i] = int32(rng.Uint32())
		}
		for i := range coefLevels {
			coefLevels[i] = byte(rng.Uint32())
		}
		c := newState.Reader.Cursor()
		r := decodeCoefTokens(&c, &newCDFs, size, geo, plane, typ, class, dcSignCtx, cf, false, &coefLevels, coefDequant{
			dc: uint32(dcFactor), ac: uint32(acFactor), shift: txScale, cfMax: uint32(dqMax),
		})
		c.CommitStateTo(&newState.Reader)

		if got, want := newState.Reader.State(), oldState.Reader.State(); got != want {
			t.Fatalf("%v %v plane=%d upd=%v txb=%d: reader state %+v, want %+v", size, typ, plane, !disableUpdate, txb, got, want)
		}
		if newCDFs != oldCDFs {
			t.Fatalf("%v %v plane=%d upd=%v txb=%d: CDF state diverged", size, typ, plane, !disableUpdate, txb)
		}
		if got, want := r.eob+1, int(oldResult.EOB); got != want {
			t.Fatalf("%v %v txb=%d: eob %d, want %d", size, typ, txb, got, want)
		}
		if r.culLevel != oldResult.CulLevel {
			t.Fatalf("%v %v txb=%d: cul level %#x, want %#x", size, typ, txb, r.culLevel, oldResult.CulLevel)
		}
		dcOnly := r.eob == 0 && typ == transform.TypeDCTDCT
		activeRows := 0
		for pos := 0; pos < maxEOB; pos++ {
			want := int32(0)
			if level := int32(coeffs[pos]); level != 0 {
				factor := acFactor
				if pos == 0 {
					factor = dcFactor
				}
				neg := level < 0
				if neg {
					level = -level
				}
				want = quantize.DequantizeMagnitudeWithBoundsTrusted(level, neg, factor, txScale, dqMax)
				activeRows = max(activeRows, pos&(int(geo.scanHeight)-1)+1)
			}
			if dcOnly && pos > 0 {
				continue
			}
			if cf[pos] != want {
				t.Fatalf("%v %v plane=%d txb=%d eob=%d: cf[%d] = %d, want %d", size, typ, plane, txb, r.eob, pos, cf[pos], want)
			}
		}
		if r.activeRows != activeRows {
			t.Fatalf("%v %v txb=%d: active rows %d, want %d", size, typ, txb, r.activeRows, activeRows)
		}
		compared++
	}
	return compared
}
