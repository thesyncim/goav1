package quantize

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestQuantLookupMatchesAV1Tables(t *testing.T) {
	tests := []struct {
		name     string
		bitDepth uint8
		qIndex   uint8
		wantDC   int32
		wantAC   int32
	}{
		{name: "8 bit lossless", bitDepth: 8, qIndex: 0, wantDC: 4, wantAC: 4},
		{name: "8 bit q2", bitDepth: 8, qIndex: 2, wantDC: 8, wantAC: 9},
		{name: "8 bit max", bitDepth: 8, qIndex: 255, wantDC: 1336, wantAC: 1828},
		{name: "10 bit max", bitDepth: 10, qIndex: 255, wantDC: 5347, wantAC: 7312},
		{name: "12 bit max", bitDepth: 12, qIndex: 255, wantDC: 21387, wantAC: 29247},
		// libaom av1-1-b8-05-mv.ivf frame 0 keyframe: BaseQIdx=37, all deltas 0.
		// libaom decodetxb dump records dequant[0]=38 dequant[1]=44 for every
		// block. q=37 -> dc_qlookup_QTX[37]=38, ac_qlookup_QTX[37]=44. Pinning
		// this rules out an off-by-one in the 8-bit lookup table when the
		// shared frame-1 regression fires.
		{name: "8 bit mv keyframe q37", bitDepth: 8, qIndex: 37, wantDC: 38, wantAC: 44},
		// libaom av1-1-b8-05-mv.ivf frame 1 displayed inter frame:
		// BaseQIdx=81, all deltas 0, no segmentation, no qmatrix. libaom
		// decodetxb dump shows dequant[0]=74 dequant[1]=88 for every block.
		// q=81 -> dc_qlookup_QTX[81]=74, ac_qlookup_QTX[81]=88. If a future
		// refactor breaks this, the MV strict-MD5 vector will fail and this
		// pin will surface a quantize-table-level regression.
		{name: "8 bit mv inter q81", bitDepth: 8, qIndex: 81, wantDC: 74, wantAC: 88},
	}
	for _, tt := range tests {
		dc, err := DCQuant(tt.qIndex, 0, tt.bitDepth)
		if err != nil {
			t.Fatalf("%s DCQuant err=%v", tt.name, err)
		}
		ac, err := ACQuant(tt.qIndex, 0, tt.bitDepth)
		if err != nil {
			t.Fatalf("%s ACQuant err=%v", tt.name, err)
		}
		if dc != tt.wantDC || ac != tt.wantAC {
			t.Fatalf("%s quant dc/ac=%d/%d want %d/%d", tt.name, dc, ac, tt.wantDC, tt.wantAC)
		}
	}
}

func TestQuantLookupClampsDelta(t *testing.T) {
	if got := ClampQIndex(10, -100); got != 0 {
		t.Fatalf("ClampQIndex low=%d want 0", got)
	}
	if got := ClampQIndex(250, 100); got != 255 {
		t.Fatalf("ClampQIndex high=%d want 255", got)
	}
	dc, err := DCQuant(10, -100, 8)
	if err != nil {
		t.Fatal(err)
	}
	ac, err := ACQuant(250, 100, 8)
	if err != nil {
		t.Fatal(err)
	}
	if dc != 4 || ac != 1828 {
		t.Fatalf("clamped quant dc/ac=%d/%d want 4/1828", dc, ac)
	}
}

func TestPlaneQuantizer(t *testing.T) {
	params := parser.QuantizationParams{
		BaseQIdx: 96,
		YDCDelta: -2,
		UDCDelta: 3,
		UACDelta: -1,
		VDCDelta: 4,
		VACDelta: 5,
	}
	y, err := PlaneQuantizer(params, params.BaseQIdx, 8, PlaneY)
	if err != nil {
		t.Fatal(err)
	}
	u, err := PlaneQuantizer(params, params.BaseQIdx, 8, PlaneU)
	if err != nil {
		t.Fatal(err)
	}
	v, err := PlaneQuantizer(params, params.BaseQIdx, 8, PlaneV)
	if err != nil {
		t.Fatal(err)
	}
	if y != (Quantizer{DC: 85, AC: 104}) {
		t.Fatalf("y=%+v want dc=85 ac=104", y)
	}
	if u != (Quantizer{DC: 92, AC: 102}) {
		t.Fatalf("u=%+v want dc=92 ac=102", u)
	}
	if v != (Quantizer{DC: 93, AC: 114}) {
		t.Fatalf("v=%+v want dc=93 ac=114", v)
	}
}

// TestPlaneQuantizerMatchesLibaomMVFrame1 pins the libaom-extracted dq_dc/dq_ac
// values for av1-1-b8-05-mv.ivf frame 1 (BaseQIdx=81, no deltas, no
// segmentation, no qmatrix). libaom's decodetxb dumps dequant[0]=74
// dequant[1]=88 for every block of that frame across Y, U, and V planes; this
// test pins the same values for PlaneY/U/V so any regression in the qindex ->
// dequant table chain surfaces here before reaching the MD5 oracle.
func TestPlaneQuantizerMatchesLibaomMVFrame1(t *testing.T) {
	params := parser.QuantizationParams{BaseQIdx: 81}
	for _, plane := range []Plane{PlaneY, PlaneU, PlaneV} {
		got, err := PlaneQuantizer(params, params.BaseQIdx, 8, plane)
		if err != nil {
			t.Fatalf("plane=%d PlaneQuantizer err=%v", plane, err)
		}
		if got != (Quantizer{DC: 74, AC: 88}) {
			t.Fatalf("plane=%d quantizer=%+v want {DC:74 AC:88}", plane, got)
		}
	}
}

// TestPlaneQuantizerMatchesLibaomMVKeyframe pins the libaom-extracted
// dq_dc/dq_ac values for av1-1-b8-05-mv.ivf frame 0 keyframe (BaseQIdx=37, no
// deltas). libaom dumps dequant[0]=38 dequant[1]=44 for every block of frame
// 0. Pairing this with the frame-1 pin documents the qindex range over which
// the MV vector exercises the dequant chain.
func TestPlaneQuantizerMatchesLibaomMVKeyframe(t *testing.T) {
	params := parser.QuantizationParams{BaseQIdx: 37}
	for _, plane := range []Plane{PlaneY, PlaneU, PlaneV} {
		got, err := PlaneQuantizer(params, params.BaseQIdx, 8, plane)
		if err != nil {
			t.Fatalf("plane=%d PlaneQuantizer err=%v", plane, err)
		}
		if got != (Quantizer{DC: 38, AC: 44}) {
			t.Fatalf("plane=%d quantizer=%+v want {DC:38 AC:44}", plane, got)
		}
	}
}

func TestTransformScaleMatchesLibaom(t *testing.T) {
	tests := []struct {
		name   string
		width  int
		height int
		want   uint8
	}{
		{name: "4x4", width: 4, height: 4, want: 0},
		{name: "8x32", width: 8, height: 32, want: 0},
		{name: "16x32", width: 16, height: 32, want: 1},
		{name: "32x32", width: 32, height: 32, want: 1},
		{name: "16x64", width: 16, height: 64, want: 1},
		{name: "32x64", width: 32, height: 64, want: 2},
		{name: "64x64", width: 64, height: 64, want: 2},
	}
	for _, tt := range tests {
		got, err := TransformScale(tt.width, tt.height)
		if err != nil {
			t.Fatalf("%s TransformScale err=%v", tt.name, err)
		}
		if got != tt.want {
			t.Fatalf("%s scale=%d want %d", tt.name, got, tt.want)
		}
	}
	if _, err := TransformScale(0, 4); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("zero width err=%v want %v", err, ErrInvalidQuantizer)
	}
}

func TestQuantizeFPNoQMatrixMatchesLibaomVector(t *testing.T) {
	q := FPQuantizer{
		Dequant: [2]int16{78, 93},
		Round:   [2]int16{39, 46},
		Quant:   [2]int16{840, 704},
	}
	coeff := []int32{-449, 624, -14, 24}
	qcoeff := make([]int32, 4)
	dqcoeff := make([]int32, 4)
	scan := []int16{0, 1, 2, 3}
	eob, err := QuantizeFPNoQMatrix(qcoeff, dqcoeff, coeff, scan, q)
	if err != nil {
		t.Fatal(err)
	}
	if eob != 2 {
		t.Fatalf("eob=%d want 2", eob)
	}
	wantQCoeff := []int32{-6, 7, 0, 0}
	wantDQCoeff := []int32{-468, 651, 0, 0}
	for i := range wantQCoeff {
		if qcoeff[i] != wantQCoeff[i] {
			t.Fatalf("qcoeff[%d]=%d want %d", i, qcoeff[i], wantQCoeff[i])
		}
		if dqcoeff[i] != wantDQCoeff[i] {
			t.Fatalf("dqcoeff[%d]=%d want %d", i, dqcoeff[i], wantDQCoeff[i])
		}
	}
}

func TestQuantizeFPNoQMatrixLogScaleAndEOB(t *testing.T) {
	q := FPQuantizer{
		Dequant:  [2]int16{64, 128},
		Round:    [2]int16{32, 64},
		Quant:    [2]int16{1024, 512},
		LogScale: 1,
	}
	coeff := []int32{4, -512, 0, 1024}
	qcoeff := []int32{99, 99, 99, 99}
	dqcoeff := []int32{99, 99, 99, 99}
	scan := []int16{0, 1, 2, 3}
	eob, err := QuantizeFPNoQMatrix(qcoeff, dqcoeff, coeff, scan, q)
	if err != nil {
		t.Fatal(err)
	}
	if eob != 4 {
		t.Fatalf("eob=%d want 4", eob)
	}
	wantQCoeff := []int32{0, -8, 0, 16}
	wantDQCoeff := []int32{0, -512, 0, 1024}
	for i := range wantQCoeff {
		if qcoeff[i] != wantQCoeff[i] || dqcoeff[i] != wantDQCoeff[i] {
			t.Fatalf("coeff[%d] q/dq=%d/%d want %d/%d", i, qcoeff[i], dqcoeff[i], wantQCoeff[i], wantDQCoeff[i])
		}
	}
}

func TestQuantizeFPNoQMatrixRejectsInvalidInputs(t *testing.T) {
	q := FPQuantizer{
		Dequant: [2]int16{78, 93},
		Round:   [2]int16{39, 46},
		Quant:   [2]int16{840, 704},
	}
	coeff := make([]int32, 4)
	qcoeff := make([]int32, 4)
	dqcoeff := make([]int32, 4)
	scan := []int16{0, 1, 2, 3}
	if _, err := QuantizeFPNoQMatrix(qcoeff[:3], dqcoeff, coeff, scan, q); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("short qcoeff err=%v want %v", err, ErrInvalidQuantizer)
	}
	if _, err := QuantizeFPNoQMatrix(qcoeff, dqcoeff[:3], coeff, scan, q); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("short dqcoeff err=%v want %v", err, ErrInvalidQuantizer)
	}
	if _, err := QuantizeFPNoQMatrix(qcoeff, dqcoeff, coeff[:3], scan, q); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("short coeff err=%v want %v", err, ErrInvalidQuantizer)
	}
	if _, err := QuantizeFPNoQMatrix(qcoeff, dqcoeff, coeff, []int16{0, 4}, q); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("invalid scan err=%v want %v", err, ErrInvalidQuantizer)
	}
	q.Dequant[0] = 0
	if _, err := QuantizeFPNoQMatrix(qcoeff, dqcoeff, coeff, scan, q); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("invalid quantizer err=%v want %v", err, ErrInvalidQuantizer)
	}
}

func TestQuantizeFPNoQMatrixAllocs(t *testing.T) {
	q := FPQuantizer{
		Dequant: [2]int16{78, 93},
		Round:   [2]int16{39, 46},
		Quant:   [2]int16{840, 704},
	}
	coeff := make([]int32, 16)
	qcoeff := make([]int32, 16)
	dqcoeff := make([]int32, 16)
	scan := []int16{0, 4, 1, 2, 5, 8, 12, 9, 6, 3, 7, 10, 13, 14, 11, 15}
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := QuantizeFPNoQMatrix(qcoeff, dqcoeff, coeff, scan, q); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("QuantizeFPNoQMatrix allocated: %f", allocs)
	}
}

func FuzzQuantizeFPNoQMatrix(f *testing.F) {
	f.Add(int32(-449), int32(624), uint16(78), uint16(93), uint8(0))
	f.Add(int32(1<<20-1), int32(-(1 << 20)), uint16(4), uint16(32767), uint8(1))

	f.Fuzz(func(t *testing.T, c0 int32, c1 int32, rawDC uint16, rawAC uint16, rawLogScale uint8) {
		dequantDC := int16(rawDC%32765 + 3)
		dequantAC := int16(rawAC%32765 + 3)
		logScale := rawLogScale & 1
		q := FPQuantizer{
			Dequant:  [2]int16{dequantDC, dequantAC},
			Round:    [2]int16{dequantDC / 2, dequantAC / 2},
			Quant:    [2]int16{int16((1 << 16) / int(dequantDC)), int16((1 << 16) / int(dequantAC))},
			LogScale: logScale,
		}
		coeff := []int32{c0, c1, 0, c0 - c1}
		qcoeff := make([]int32, len(coeff))
		dqcoeff := make([]int32, len(coeff))
		scan := []int16{0, 1, 2, 3}
		eob, err := QuantizeFPNoQMatrix(qcoeff, dqcoeff, coeff, scan, q)
		if err != nil {
			t.Fatalf("QuantizeFPNoQMatrix err=%v", err)
		}
		wantEOB := 0
		for i, rawRC := range scan {
			rc := int(rawRC)
			if qcoeff[rc] == 0 {
				if dqcoeff[rc] != 0 {
					t.Fatalf("dqcoeff[%d]=%d with zero qcoeff", rc, dqcoeff[rc])
				}
				continue
			}
			wantEOB = i + 1
			absQ := qcoeff[rc]
			negative := absQ < 0
			if negative {
				absQ = -absQ
			}
			idx := 0
			if rc != 0 {
				idx = 1
			}
			wantDQ := (absQ * int32(q.Dequant[idx])) >> q.LogScale
			if negative {
				wantDQ = -wantDQ
			}
			if dqcoeff[rc] != wantDQ {
				t.Fatalf("dqcoeff[%d]=%d want %d", rc, dqcoeff[rc], wantDQ)
			}
		}
		if eob != wantEOB {
			t.Fatalf("eob=%d want %d", eob, wantEOB)
		}
	})
}

func BenchmarkQuantLookup(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = DCQuant(uint8(i), int16(i&7)-3, 10)
		_, _ = ACQuant(uint8(i), int16(i&7)-3, 10)
	}
}

func BenchmarkDequantizeBlockEOB16Trusted(b *testing.B) {
	const width, height = 64, 64
	coeff := make([]int16, width*height)
	scan := make([]int16, width*height)
	for i := range scan {
		scan[i] = int16(i)
	}
	for i := 0; i < 16; i++ {
		coeff[i] = int16(i + 1)
	}
	dst := make([]int32, width*height)
	q := Quantizer{DC: 80, AC: 97}
	b.ReportAllocs()
	for b.Loop() {
		DequantizeBlockScaledBitDepthEOBTrusted(dst, height, coeff, height, scan, 16, width, height, q, 2, 8)
	}
}

func TestDequantizeDCCoeffBitDepthTrustedMatchesFull(t *testing.T) {
	const width, height = 4, 4
	q := Quantizer{DC: 91, AC: 137}
	for _, tc := range []struct {
		name     string
		coeff    int16
		txScale  uint8
		bitDepth uint8
	}{
		{name: "8 bit", coeff: 3, txScale: 0, bitDepth: 8},
		{name: "10 bit scaled", coeff: -7, txScale: 1, bitDepth: 10},
		{name: "12 bit scaled", coeff: 11, txScale: 2, bitDepth: 12},
	} {
		coeff := make([]int16, width*height)
		coeff[0] = tc.coeff
		want := make([]int32, width*height)
		if err := DequantizeBlockScaledBitDepth(want, height, coeff, height, width, height, q, tc.txScale, tc.bitDepth); err != nil {
			t.Fatalf("%s full dequant err=%v", tc.name, err)
		}
		if got := DequantizeDCCoeffBitDepthTrusted(tc.coeff, q, tc.txScale, tc.bitDepth); got != want[0] {
			t.Fatalf("%s dc=%d want %d", tc.name, got, want[0])
		}
	}
}
