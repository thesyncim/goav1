package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestCoeffQContextMatchesLibaom(t *testing.T) {
	tests := []struct {
		q    uint8
		want int
	}{
		{q: 0, want: 0},
		{q: 20, want: 0},
		{q: 21, want: 1},
		{q: 60, want: 1},
		{q: 61, want: 2},
		{q: 120, want: 2},
		{q: 121, want: 3},
		{q: 255, want: 3},
	}
	for _, tt := range tests {
		if got := CoeffQContext(tt.q); got != tt.want {
			t.Fatalf("CoeffQContext(%d)=%d want %d", tt.q, got, tt.want)
		}
	}
}

func TestCoeffCDFsInitDefaultMatchesLibaom(t *testing.T) {
	var cdfs CoeffCDFs
	if err := cdfs.InitDefault(0); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		cdf  *entropy.CDF
		want []uint16
	}{
		{name: "dc sign y ctx0", cdf: &cdfs.DCSign[CoeffPlaneY][0], want: []uint16{16768, 0, 0}},
		{name: "txb skip 4x4 ctx0", cdf: &cdfs.TXBSkip[0][0], want: []uint16{919, 0, 0}},
		{name: "eob extra 4x4 y ctx0", cdf: &cdfs.EOBExtra[0][CoeffPlaneY][0], want: []uint16{15807, 0, 0}},
		{name: "eob flag 16 y ctx0", cdf: &cdfs.EOBFlag16[CoeffPlaneY][0], want: []uint16{31928, 31729, 30788, 27873, 0, 0}},
		{name: "coeff br 4x4 y ctx0", cdf: &cdfs.CoeffBR[0][CoeffPlaneY][0], want: []uint16{18470, 12050, 8594, 0, 0}},
		{name: "coeff base 4x4 y ctx0", cdf: &cdfs.CoeffBase[0][CoeffPlaneY][0], want: []uint16{28734, 23838, 20041, 0, 0}},
		{name: "coeff base eob 4x4 y ctx0", cdf: &cdfs.CoeffBaseEOB[0][CoeffPlaneY][0], want: []uint16{14931, 3713, 0, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertEntropyCDFValues(t, tt.cdf.Values(), tt.want)
		})
	}

	var q1 CoeffCDFs
	if err := q1.InitDefault(21); err != nil {
		t.Fatal(err)
	}
	assertEntropyCDFValues(t, q1.TXBSkip[0][0].Values(), []uint16{2397, 0, 0})
}

func TestCoeffContextsMatchLibaom(t *testing.T) {
	ctxTests := []struct {
		size TransformSize
		want int
	}{
		{size: TransformSize4x4, want: 0},
		{size: TransformSize4x8, want: 1},
		{size: TransformSize8x16, want: 2},
		{size: TransformSize16x64, want: 3},
		{size: TransformSize64x64, want: 4},
	}
	for _, tt := range ctxTests {
		got, err := CoeffTransformSizeContext(tt.size)
		if err != nil {
			t.Fatal(err)
		}
		if got != tt.want {
			t.Fatalf("CoeffTransformSizeContext(%d)=%d want %d", tt.size, got, tt.want)
		}
	}

	eobTests := []struct {
		size TransformSize
		want int
	}{
		{size: TransformSize4x4, want: 0},
		{size: TransformSize4x8, want: 1},
		{size: TransformSize8x8, want: 2},
		{size: TransformSize16x16, want: 4},
		{size: TransformSize16x64, want: 5},
		{size: TransformSize64x64, want: 6},
	}
	for _, tt := range eobTests {
		got, err := EOBMultiSize(tt.size)
		if err != nil {
			t.Fatal(err)
		}
		if got != tt.want {
			t.Fatalf("EOBMultiSize(%d)=%d want %d", tt.size, got, tt.want)
		}
	}

	if got, err := RecEOBPosition(3, 1); err != nil || got != 4 {
		t.Fatalf("RecEOBPosition(3,1)=%d err=%v want 4", got, err)
	}
	if got, err := RecEOBPosition(4, 2); err != nil || got != 7 {
		t.Fatalf("RecEOBPosition(4,2)=%d err=%v want 7", got, err)
	}
	if got, err := CoeffPlaneTypeForPlane(2); err != nil || got != CoeffPlaneUV {
		t.Fatalf("CoeffPlaneTypeForPlane(2)=%d err=%v want UV", got, err)
	}
}

func TestReadCoeffPrimitives(t *testing.T) {
	var cdfs CoeffCDFs
	if err := cdfs.InitDefault(0); err != nil {
		t.Fatal(err)
	}

	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	skip, err := state.ReadTXBSkip(&cdfs, TXBSkipRequest{Size: TransformSize4x4, Context: 0})
	if err != nil {
		t.Fatal(err)
	}
	if skip {
		t.Fatal("txb skip=true want false")
	}
	if got := cdfs.TXBSkip[0][0].Values()[2]; got != 1 {
		t.Fatalf("txb skip count=%d want 1", got)
	}

	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	eob, err := state.ReadEOB(&cdfs, EOBRequest{Size: TransformSize4x4, Plane: CoeffPlaneY})
	if err != nil {
		t.Fatal(err)
	}
	if eob != (EOBResult{Token: 1, Position: 1}) {
		t.Fatalf("eob=%+v want token 1 position 1", eob)
	}

	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	level, err := state.ReadCoeffBaseEOB(&cdfs, CoeffTokenRequest{Size: TransformSize4x4, Plane: CoeffPlaneY, Context: 0})
	if err != nil {
		t.Fatal(err)
	}
	if level != 1 {
		t.Fatalf("base-eob level=%d want 1", level)
	}
	base, err := state.ReadCoeffBase(&cdfs, CoeffTokenRequest{Size: TransformSize4x4, Plane: CoeffPlaneY, Context: 0})
	if err != nil {
		t.Fatal(err)
	}
	if base != 0 {
		t.Fatalf("base=%d want 0", base)
	}
	br, err := state.ReadCoeffBR(&cdfs, CoeffTokenRequest{Size: TransformSize4x4, Plane: CoeffPlaneY, Context: 0})
	if err != nil {
		t.Fatal(err)
	}
	if br != 0 {
		t.Fatalf("br=%d want 0", br)
	}
	sign, err := state.ReadDCSign(&cdfs, CoeffPlaneY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if sign {
		t.Fatal("sign=true want false")
	}
}

func TestCoeffLevelContextsMatchLibaom(t *testing.T) {
	scratchLen, err := CoeffLevelsScratchLen(TransformSize4x4)
	if err != nil {
		t.Fatal(err)
	}
	if scratchLen != 64 {
		t.Fatalf("CoeffLevelsScratchLen(4x4)=%d want 64", scratchLen)
	}

	brEOBTests := []struct {
		name  string
		size  TransformSize
		class transform.Class
		index int
		want  int
	}{
		{name: "dc", size: TransformSize4x4, class: transform.Class2D, index: 0, want: 0},
		{name: "2d top-left band", size: TransformSize4x4, class: transform.Class2D, index: 1, want: 7},
		{name: "2d outer band", size: TransformSize4x4, class: transform.Class2D, index: 10, want: 14},
		{name: "horiz first column", size: TransformSize4x4, class: transform.ClassHoriz, index: 3, want: 7},
		{name: "horiz outer column", size: TransformSize4x4, class: transform.ClassHoriz, index: 4, want: 14},
		{name: "vert first row", size: TransformSize4x4, class: transform.ClassVert, index: 4, want: 7},
		{name: "vert outer row", size: TransformSize4x4, class: transform.ClassVert, index: 1, want: 14},
	}
	for _, tt := range brEOBTests {
		t.Run("br_eob "+tt.name, func(t *testing.T) {
			got, err := CoeffBRContextEOB(tt.size, tt.class, tt.index)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("CoeffBRContextEOB=%d want %d", got, tt.want)
			}
		})
	}

	levels := make([]uint8, scratchLen)
	mustSetCoeffLevel(t, levels, TransformSize4x4, 1, 3)
	mustSetCoeffLevel(t, levels, TransformSize4x4, 4, 2)
	mustSetCoeffLevel(t, levels, TransformSize4x4, 5, 1)
	if got, err := CoeffBRContext(levels, TransformSize4x4, transform.Class2D, 0); err != nil || got != 3 {
		t.Fatalf("CoeffBRContext populated=%d err=%v want 3", got, err)
	}
	if got, err := CoeffBRContext(make([]uint8, scratchLen), TransformSize4x4, transform.Class2D, 10); err != nil || got != 14 {
		t.Fatalf("CoeffBRContext outer=%d err=%v want 14", got, err)
	}

	lowerTests := []struct {
		name  string
		size  TransformSize
		class transform.Class
		index int
		want  int
	}{
		{name: "2d dc", size: TransformSize4x4, class: transform.Class2D, index: 0, want: 0},
		{name: "2d near dc", size: TransformSize4x4, class: transform.Class2D, index: 1, want: 1},
		{name: "2d middle band", size: TransformSize4x4, class: transform.Class2D, index: 2, want: 6},
		{name: "2d outer band", size: TransformSize4x4, class: transform.Class2D, index: 15, want: 21},
		{name: "tall early row", size: TransformSize4x8, class: transform.Class2D, index: 1, want: 11},
		{name: "tall regular band", size: TransformSize4x8, class: transform.Class2D, index: 2, want: 6},
		{name: "wide early column", size: TransformSize8x4, class: transform.Class2D, index: 4, want: 16},
		{name: "wide regular band", size: TransformSize8x4, class: transform.Class2D, index: 8, want: 6},
		{name: "horiz col 0", size: TransformSize8x8, class: transform.ClassHoriz, index: 0, want: 26},
		{name: "horiz col 1", size: TransformSize8x8, class: transform.ClassHoriz, index: 8, want: 31},
		{name: "horiz col 2", size: TransformSize8x8, class: transform.ClassHoriz, index: 16, want: 36},
		{name: "vert row 0", size: TransformSize8x8, class: transform.ClassVert, index: 0, want: 26},
		{name: "vert row 1", size: TransformSize8x8, class: transform.ClassVert, index: 1, want: 31},
		{name: "vert row 2", size: TransformSize8x8, class: transform.ClassVert, index: 2, want: 36},
	}
	for _, tt := range lowerTests {
		t.Run("lower "+tt.name, func(t *testing.T) {
			n, err := CoeffLevelsScratchLen(tt.size)
			if err != nil {
				t.Fatal(err)
			}
			got, err := CoeffLowerLevelsContext(make([]uint8, n), tt.size, tt.class, tt.index)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("CoeffLowerLevelsContext=%d want %d", got, tt.want)
			}
		})
	}
}

func TestReadCoefficientsTXBDecodesSingleDC(t *testing.T) {
	result, coeffs := readCoefficientsTXBForTest(t, []byte{0x00}, TransformSize4x4, transform.Class2D, CoeffPlaneY, 0)
	if result.AllZero {
		t.Fatal("AllZero=true want false")
	}
	if result.EOB != 1 || result.MaxScanLine != 0 {
		t.Fatalf("result=%+v want eob=1 maxScanLine=0", result)
	}
	if coeffs[0] != 1 {
		t.Fatalf("coeff[0]=%d want 1", coeffs[0])
	}
	for i, coeff := range coeffs[1:] {
		if coeff != 0 {
			t.Fatalf("coeff[%d]=%d want 0", i+1, coeff)
		}
	}
	if result.CulLevel != 17 {
		t.Fatalf("cul_level=%d want 17", result.CulLevel)
	}
}

func TestCoeffRejectsInvalidInputs(t *testing.T) {
	var cdfs CoeffCDFs
	if _, err := cdfs.TXBSkipCDF(TransformSize4x4, 0); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("uninitialized cdf err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if err := cdfs.InitDefault(0); err != nil {
		t.Fatal(err)
	}
	if _, err := cdfs.TXBSkipCDF(transformSizeCount, 0); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("bad tx cdf err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if _, err := cdfs.EOBExtraCDF(TransformSize4x4, CoeffPlaneY, EOBCoefContexts); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("bad eob extra err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if _, err := cdfs.DCSignCDF(CoeffPlaneTypes, 0); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("bad dc sign err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if _, err := RecEOBPosition(len(eobGroupStart), 0); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad eob pos err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := CoeffPlaneTypeForPlane(3); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad plane err=%v want %v", err, ErrInvalidDecodeState)
	}

	var nilState *DecodeState
	if _, err := nilState.ReadTXBSkip(&cdfs, TXBSkipRequest{Size: TransformSize4x4}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil state err=%v want %v", err, ErrInvalidDecodeState)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := state.ReadEOB(&cdfs, EOBRequest{Size: TransformSize4x4, Plane: CoeffPlaneTypes}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad eob req err=%v want %v", err, ErrInvalidDecodeState)
	}

	txSize, err := TransformSize4x4.TransformSize()
	if err != nil {
		t.Fatal(err)
	}
	scan, scratch := coeffScanAndScratch(t, TransformSize4x4, txSize, transform.Class2D)
	if _, err := state.ReadCoefficientsTXB(&cdfs, TXBDecodeRequest{
		Size:           TransformSize4x4,
		Plane:          CoeffPlaneY,
		Class:          transform.Class2D,
		TXBSkipContext: 0,
	}, nil, scan, scratch); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("short coeff txb err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.ReadCoefficientsTXB(&cdfs, TXBDecodeRequest{
		Size:           TransformSize4x4,
		Plane:          CoeffPlaneY,
		Class:          transform.Class(99),
		TXBSkipContext: 0,
	}, make([]int16, len(scan)), scan, scratch); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad class txb err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestCoeffAllocs(t *testing.T) {
	var cdfs CoeffCDFs
	var state DecodeState
	payload := []byte{0x00}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := cdfs.InitDefault(0); err != nil {
			t.Fatal(err)
		}
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		if _, err := state.ReadTXBSkip(&cdfs, TXBSkipRequest{Size: TransformSize4x4, Context: 0}); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("coeff primitive decode allocated: %f", allocs)
	}
}

func TestReadCoefficientsTXBAllocs(t *testing.T) {
	txSize, err := TransformSize4x4.TransformSize()
	if err != nil {
		t.Fatal(err)
	}
	scan, scratch := coeffScanAndScratch(t, TransformSize4x4, txSize, transform.Class2D)
	coeffs := make([]int16, len(scan))
	payload := []byte{0x00}
	req := TXBDecodeRequest{
		Size:            TransformSize4x4,
		Plane:           CoeffPlaneY,
		Class:           transform.Class2D,
		TXBSkipContext:  0,
		DCSignContext:   0,
		EOBMultiContext: 0,
	}
	var cdfs CoeffCDFs
	var state DecodeState

	allocs := testing.AllocsPerRun(1000, func() {
		if err := cdfs.InitDefault(0); err != nil {
			t.Fatal(err)
		}
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		result, err := state.ReadCoefficientsTXB(&cdfs, req, coeffs, scan, scratch)
		if err != nil {
			t.Fatal(err)
		}
		if result.EOB != 1 || coeffs[0] != 1 {
			t.Fatalf("txb result=%+v coeff[0]=%d want eob=1 coeff=1", result, coeffs[0])
		}
	})
	if allocs != 0 {
		t.Fatalf("txb coefficient decode allocated: %f", allocs)
	}
}

func FuzzReadCoeffPrimitives(f *testing.F) {
	f.Add([]byte{0x00}, uint8(TransformSize4x4), uint8(0), uint8(0), uint8(0), uint8(0))
	f.Add([]byte{0xff}, uint8(TransformSize16x16), uint8(1), uint8(2), uint8(7), uint8(20))
	f.Add([]byte{0xa5, 0x5a}, uint8(TransformSize64x64), uint8(0), uint8(1), uint8(8), uint8(40))

	f.Fuzz(func(t *testing.T, payload []byte, rawSize uint8, rawPlane uint8, rawEOBCtx uint8, rawCoeffCtx uint8, q uint8) {
		if len(payload) == 0 || len(payload) > 64 {
			return
		}
		size := TransformSize(rawSize % uint8(transformSizeCount))
		plane := CoeffPlaneType(rawPlane % CoeffPlaneTypes)
		var cdfs CoeffCDFs
		if err := cdfs.InitDefault(q); err != nil {
			t.Fatal(err)
		}
		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		if _, err := state.ReadTXBSkip(&cdfs, TXBSkipRequest{
			Size:    size,
			Context: int(rawCoeffCtx % TXBSkipContexts),
		}); err != nil {
			t.Fatalf("ReadTXBSkip err=%v", err)
		}
		if _, err := state.ReadEOB(&cdfs, EOBRequest{
			Size:            size,
			Plane:           plane,
			EOBMultiContext: int(rawEOBCtx % maxEOBFlagContexts),
		}); err != nil {
			t.Fatalf("ReadEOB err=%v", err)
		}
		if _, err := state.ReadCoeffBaseEOB(&cdfs, CoeffTokenRequest{
			Size:    size,
			Plane:   plane,
			Context: int(rawCoeffCtx % EOBBaseContexts),
		}); err != nil {
			t.Fatalf("ReadCoeffBaseEOB err=%v", err)
		}
		if _, err := state.ReadCoeffBase(&cdfs, CoeffTokenRequest{
			Size:    size,
			Plane:   plane,
			Context: int(rawCoeffCtx % CoeffBaseContexts),
		}); err != nil {
			t.Fatalf("ReadCoeffBase err=%v", err)
		}
		if _, err := state.ReadCoeffBR(&cdfs, CoeffTokenRequest{
			Size:    size,
			Plane:   plane,
			Context: int(rawCoeffCtx % CoeffBRContexts),
		}); err != nil {
			t.Fatalf("ReadCoeffBR err=%v", err)
		}
		if _, err := state.ReadDCSign(&cdfs, plane, int(rawCoeffCtx%3)); err != nil {
			t.Fatalf("ReadDCSign err=%v", err)
		}
	})
}

func FuzzReadCoefficientsTXB(f *testing.F) {
	f.Add([]byte{0x00}, uint8(TransformSize4x4), uint8(transform.Class2D), uint8(CoeffPlaneY), uint8(0), uint8(0), uint8(0), uint8(0))
	f.Add([]byte{0xff}, uint8(TransformSize8x8), uint8(transform.ClassHoriz), uint8(CoeffPlaneUV), uint8(1), uint8(1), uint8(1), uint8(21))
	f.Add([]byte{0xa5, 0x5a}, uint8(TransformSize16x16), uint8(transform.ClassVert), uint8(CoeffPlaneY), uint8(6), uint8(2), uint8(1), uint8(121))
	f.Add([]byte{0x4d, 0x21, 0xc3, 0x7e}, uint8(TransformSize64x64), uint8(transform.Class2D), uint8(CoeffPlaneUV), uint8(12), uint8(2), uint8(1), uint8(255))

	f.Fuzz(func(t *testing.T, payload []byte, rawSize uint8, rawClass uint8, rawPlane uint8, rawSkipCtx uint8, rawDCSignCtx uint8, rawEOBCtx uint8, q uint8) {
		if len(payload) == 0 || len(payload) > 128 {
			return
		}
		size := TransformSize(rawSize % uint8(transformSizeCount))
		class := transform.Class(rawClass % 3)
		plane := CoeffPlaneType(rawPlane % CoeffPlaneTypes)
		txSize, err := size.TransformSize()
		if err != nil {
			t.Fatal(err)
		}
		scan, scratch := coeffScanAndScratch(t, size, txSize, class)
		coeffs := make([]int16, len(scan))

		var cdfs CoeffCDFs
		if err := cdfs.InitDefault(q); err != nil {
			t.Fatal(err)
		}
		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		result, err := state.ReadCoefficientsTXB(&cdfs, TXBDecodeRequest{
			Size:            size,
			Plane:           plane,
			Class:           class,
			TXBSkipContext:  int(rawSkipCtx % TXBSkipContexts),
			DCSignContext:   int(rawDCSignCtx % 3),
			EOBMultiContext: int(rawEOBCtx % maxEOBFlagContexts),
		}, coeffs, scan, scratch)
		if err != nil {
			if errors.Is(err, ErrInvalidDecodeState) {
				return
			}
			t.Fatalf("ReadCoefficientsTXB err=%v", err)
		}
		assertTXBDecodeInvariants(t, result, coeffs, scan)
	})
}

func readCoefficientsTXBForTest(t *testing.T, payload []byte, size TransformSize, class transform.Class, plane CoeffPlaneType, q uint8) (TXBDecodeResult, []int16) {
	t.Helper()
	txSize, err := size.TransformSize()
	if err != nil {
		t.Fatal(err)
	}
	scan, scratch := coeffScanAndScratch(t, size, txSize, class)
	coeffs := make([]int16, len(scan))
	var cdfs CoeffCDFs
	if err := cdfs.InitDefault(q); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	result, err := state.ReadCoefficientsTXB(&cdfs, TXBDecodeRequest{
		Size:            size,
		Plane:           plane,
		Class:           class,
		TXBSkipContext:  0,
		DCSignContext:   0,
		EOBMultiContext: 0,
	}, coeffs, scan, scratch)
	if err != nil {
		t.Fatal(err)
	}
	return result, coeffs
}

func coeffScanAndScratch(t *testing.T, size TransformSize, txSize transform.Size, class transform.Class) ([]int16, []uint8) {
	t.Helper()
	scanSize, err := transform.ScanSize(txSize)
	if err != nil {
		t.Fatal(err)
	}
	scan := make([]int16, scanSize.Width*scanSize.Height)
	inverse := make([]int16, len(scan))
	if err := transform.FillDefaultScan(scan, inverse, txSize, class); err != nil {
		t.Fatal(err)
	}
	scratchLen, err := CoeffLevelsScratchLen(size)
	if err != nil {
		t.Fatal(err)
	}
	return scan, make([]uint8, scratchLen)
}

func mustSetCoeffLevel(t *testing.T, levels []uint8, size TransformSize, coeffIndex int, level int) {
	t.Helper()
	if err := setCoeffLevel(levels, size, coeffIndex, level); err != nil {
		t.Fatal(err)
	}
}

func assertTXBDecodeInvariants(t *testing.T, result TXBDecodeResult, coeffs []int16, scan []int16) {
	t.Helper()
	if result.CulLevel < 0 || result.CulLevel > CoeffContextMask+(2<<CoeffContextBits) {
		t.Fatalf("cul_level=%d outside context range", result.CulLevel)
	}
	if result.AllZero {
		if result.EOB != 0 || result.MaxScanLine != 0 || result.CulLevel != 0 {
			t.Fatalf("all-zero result=%+v want zeroed result", result)
		}
		for i, coeff := range coeffs {
			if coeff != 0 {
				t.Fatalf("all-zero coeff[%d]=%d want 0", i, coeff)
			}
		}
		return
	}
	if result.EOB <= 0 || result.EOB > len(scan) {
		t.Fatalf("eob=%d outside scan len %d", result.EOB, len(scan))
	}
	if result.MaxScanLine < 0 || result.MaxScanLine >= len(coeffs) {
		t.Fatalf("max_scan_line=%d outside coeff len %d", result.MaxScanLine, len(coeffs))
	}
	last := int(scan[result.EOB-1])
	if coeffs[last] == 0 {
		t.Fatalf("last non-zero scan coeff[%d]=0", last)
	}
	for c := result.EOB; c < len(scan); c++ {
		pos := int(scan[c])
		if coeffs[pos] != 0 {
			t.Fatalf("coeff after eob scan=%d pos=%d value=%d want 0", c, pos, coeffs[pos])
		}
	}
}
