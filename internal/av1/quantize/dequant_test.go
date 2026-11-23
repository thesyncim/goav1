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
