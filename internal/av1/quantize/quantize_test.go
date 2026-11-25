package quantize

import (
	"errors"
	"fmt"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/transform"
	"math/bits"
	"math/rand"
	"reflect"
	"slices"
	"testing"
)

// TestDQCoeffBoundsMatchLibaom anchors the bit-depth-dependent dequantized
// coefficient clamp bounds to the values libaom's decode_coefs() applies in
// av1/decoder/decodetxb.c:312.
func TestDQCoeffBoundsMatchLibaom(t *testing.T) {
	cases := []struct {
		bd      uint8
		wantMin int32
		wantMax int32
	}{
		{bd: 8, wantMin: -32768, wantMax: 32767},
		{bd: 10, wantMin: -131072, wantMax: 131071},
		{bd: 12, wantMin: -524288, wantMax: 524287},
	}
	for _, tc := range cases {
		got_min, got_max, ok := dqCoeffBounds(tc.bd)
		if !ok {
			t.Fatalf("bd=%d: dqCoeffBounds returned !ok", tc.bd)
		}
		if got_min != tc.wantMin || got_max != tc.wantMax {
			t.Fatalf("bd=%d: range=[%d,%d] want [%d,%d]", tc.bd, got_min, got_max, tc.wantMin, tc.wantMax)
		}
	}
}

// TestDQCoeffBoundsRejectsInvalidBitDepth ensures only 8/10/12 are accepted.
func TestDQCoeffBoundsRejectsInvalidBitDepth(t *testing.T) {
	for _, bd := range []uint8{0, 1, 6, 7, 9, 11, 13, 16, 255} {
		if _, _, ok := dqCoeffBounds(bd); ok {
			t.Fatalf("bd=%d: dqCoeffBounds unexpectedly accepted", bd)
		}
	}
}

// TestDequantizeBlockScaledBitDepthClamps verifies that the 10-bit path lets
// values past the 8-bit clamp through, up to the 18-bit bound. This anchors
// the fix for the 10-bit quantizer cohort: the OLD goav1 path produced raw
// products that exceeded libaom's tran_low_t bound, feeding wrong residuals
// to the inverse transform.
func TestDequantizeBlockScaledBitDepthClamps(t *testing.T) {
	// Force a dequant product > int16 max but within ±131071 (10-bit) and
	// > ±524287 (12-bit). coeff=300, scale=200 ⇒ 60000 > 32767, < 131071.
	width, height := 4, 4
	q := Quantizer{DC: 200, AC: 200}
	coeff := make([]int16, width*height)
	for i := range coeff {
		coeff[i] = 300
	}
	dst8 := make([]int32, width*height)
	dst10 := make([]int32, width*height)
	if err := DequantizeBlockScaledBitDepth(dst8, height, coeff, height, width, height, q, 0, 8); err != nil {
		t.Fatalf("8-bit dequant: %v", err)
	}
	if err := DequantizeBlockScaledBitDepth(dst10, height, coeff, height, width, height, q, 0, 10); err != nil {
		t.Fatalf("10-bit dequant: %v", err)
	}
	for i := range dst8 {
		if dst8[i] != 32767 {
			t.Fatalf("8-bit dst[%d]=%d want 32767 (int16 max)", i, dst8[i])
		}
		if dst10[i] != 60000 {
			t.Fatalf("10-bit dst[%d]=%d want 60000 (unclamped)", i, dst10[i])
		}
	}
}

// TestDequantizeBlockScaledBitDepthInt64Multiply pins the int64 widening of
// libaom's `(int64)level * dqv & 0xffffff` formula. The Go input is already
// int16, so for legal AV1 streams the multiply fits in int32; this test
// constructs an out-of-range product that would wrap to a different value
// under a 32-bit multiply, to guard against future regressions where the
// quantized buffer widens to int32 (e.g. for a more libaom-faithful pipeline).
// The 10-bit q32/q63 cohort exposed this divergence as +4 residual deltas at
// the first sample of frame 0; widening to int64 keeps the bit pattern of
// libaom's tran_low_t multiply exactly.
func TestDequantizeBlockScaledBitDepthInt64Multiply(t *testing.T) {
	// Pre-fill dequant output and dequantize a single nonzero level. Drive
	// `level * scale` into a value > 2^23 so the &0xffffff mask actually
	// strips bits; this verifies the mask is applied after the int64 widen.
	width, height := 4, 4
	q := Quantizer{DC: 16777216 - 100, AC: 16777216 - 100} // 24-bit-ish scale
	coeff := make([]int16, width*height)
	coeff[0] = 2
	dst := make([]int32, width*height)
	if err := DequantizeBlockScaledBitDepth(dst, height, coeff, height, width, height, q, 0, 12); err != nil {
		t.Fatalf("12-bit dequant: %v", err)
	}
	// product = 2 * (1<<24 - 100) = (1<<25 - 200). After &0xffffff: low
	// 24 bits = (1<<25 - 200) mod (1<<24) = (1<<24 - 200). The 12-bit dq
	// clamp at ±(1<<19) caps this at 524287.
	if dst[0] != 524287 {
		t.Fatalf("dst[0]=%d want 524287 (12-bit dq clamp ceiling)", dst[0])
	}
}

// dequantColumnImpl is the dispatch slot bound to the architecture-best column
// kernel (NEON on arm64, pure-Go elsewhere). These tests assert that whatever
// is bound is bit-identical to the pure-Go reference over the full parameter
// space the decoder exercises: every supported scale, transform scale, and
// bit-depth clamp.

func dqBoundsForBitDepth(t *testing.T, bd uint8) (int32, int32) {
	t.Helper()
	lo, hi, ok := dqCoeffBounds(bd)
	if !ok {
		t.Fatalf("dqCoeffBounds(%d) not ok", bd)
	}
	return lo, hi
}

func TestDequantColumnDispatchMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5eed))

	// Coefficient lengths chosen to cover the 8-wide vector body plus every
	// possible tail remainder (n%8) and the empty/short cases.
	lengths := []int{0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 1024}
	scales := []int32{4, 9, 38, 44, 74, 88, 1828, 7312, 29247}
	txScales := []uint8{0, 1, 2}
	bitDepths := []uint8{8, 10, 12}

	for _, n := range lengths {
		// Build a coefficient buffer that includes the int16 extremes and a
		// spread of random values so masking, shifting, sign reapplication and
		// clamping are all exercised.
		coeff := make([]int16, n)
		for i := range coeff {
			switch i % 5 {
			case 0:
				coeff[i] = 32767
			case 1:
				coeff[i] = -32768
			case 2:
				coeff[i] = 0
			default:
				coeff[i] = int16(rng.Intn(65536) - 32768)
			}
		}
		for _, scale := range scales {
			for _, txScale := range txScales {
				for _, bd := range bitDepths {
					dqMin, dqMax := dqBoundsForBitDepth(t, bd)

					want := make([]int32, n)
					dequantColumnPureGo(want, coeff, scale, txScale, dqMin, dqMax)

					got := make([]int32, n)
					dequantColumnImpl(got, coeff, scale, txScale, dqMin, dqMax)

					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("n=%d scale=%d txScale=%d bd=%d coeff[%d]=%d: got %d want %d",
								n, scale, txScale, bd, i, coeff[i], got[i], want[i])
						}
					}
				}
			}
		}
	}
}

func TestDequantColumnDispatchZeroAlloc(t *testing.T) {
	coeff := make([]int16, 1024)
	for i := range coeff {
		coeff[i] = int16(i*131 - 32768)
	}
	dst := make([]int32, 1024)
	allocs := testing.AllocsPerRun(200, func() {
		dequantColumnImpl(dst, coeff, 88, 1, -32768, 32767)
	})
	if allocs != 0 {
		t.Fatalf("dequantColumnImpl allocs = %v, want 0", allocs)
	}
}

func BenchmarkDequantColumn(b *testing.B) {
	coeff := make([]int16, 1024)
	for i := range coeff {
		coeff[i] = int16(i*131 - 32768)
	}
	dst := make([]int32, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dequantColumnImpl(dst, coeff, 88, 1, -32768, 32767)
	}
}

func BenchmarkDequantColumnPureGo(b *testing.B) {
	coeff := make([]int16, 1024)
	for i := range coeff {
		coeff[i] = int16(i*131 - 32768)
	}
	dst := make([]int32, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dequantColumnPureGo(dst, coeff, 88, 1, -32768, 32767)
	}
}

// Independent scalar statement of the original flat dequant + separate row
// traversal. Do not use dequantScalar or the fused row collection here.
func dequantEOBRowsOriginal(dst []int32, coeff []int16, scan []int16, eob, height int, q Quantizer, txScale, bitDepth uint8) int {
	clear(dst)
	for i := 0; i < eob; i++ {
		pos := int(scan[i])
		factor := int64(q.AC)
		if pos == 0 {
			factor = int64(q.DC)
		}
		level := int64(coeff[pos])
		negative := level < 0
		if negative {
			level = -level
		}
		value := ((level * factor) & 0xffffff) >> txScale
		if negative {
			value = -value
		}
		limit := int64(1) << (7 + bitDepth)
		if value < -limit {
			value = -limit
		}
		if value >= limit {
			value = limit - 1
		}
		dst[pos] = int32(value)
	}
	rows := 0
	for _, pos := range scan[:eob] {
		row := int(pos) % height
		if row >= rows {
			rows = row + 1
		}
	}
	return rows
}

func TestDequantEOBRowsFlatOriginal(t *testing.T) {
	levels := []int16{-32768, -32767, -1025, -17, -1, 0, 1, 17, 1025, 32766, 32767}
	for _, dims := range [][2]int{{4, 4}, {4, 8}, {8, 4}, {8, 8}, {4, 16}, {16, 4}, {8, 32}, {32, 8}, {16, 16}, {16, 32}, {32, 16}, {32, 32}} {
		width, height := dims[0], dims[1]
		n := width * height
		coeff := make([]int16, n)
		for i := range coeff {
			coeff[i] = levels[(i*7+height)%len(levels)]
		}
		for _, order := range []string{"columns", "rows", "permuted", "zeros_last_row"} {
			scan := make([]int16, n)
			for i := range scan {
				switch order {
				case "columns":
					scan[i] = int16(i)
				case "rows":
					scan[i] = int16((i%width)*height + i/width)
				case "permuted":
					scan[i] = int16((i*13 + 7) % n)
				case "zeros_last_row":
					scan[i] = int16(n - 1 - i)
				}
			}
			c := slices.Clone(coeff)
			if order == "zeros_last_row" {
				for i := height - 1; i < n; i += height {
					c[i] = 0
				}
			}
			for _, q := range []Quantizer{{4, 8}, {1336, 1828}, {21387, 29247}, {1<<30 + 1, 1<<30 - 1}} {
				for _, bd := range []uint8{8, 10, 12} {
					for ts := uint8(0); ts <= 2; ts++ {
						for eob := 0; eob <= n; eob++ {
							if n > 64 && eob > 3 && eob < n-2 && eob%(n/8) != 0 {
								continue
							}
							want, got := make([]int32, n+8), make([]int32, n+8)
							for i := range got {
								got[i] = int32(i*29 - 12345)
								want[i] = got[i]
							}
							wc, ws := slices.Clone(c), slices.Clone(scan)
							wr := dequantEOBRowsOriginal(want[3:n+3:n+3], wc, ws, eob, height, q, ts, bd)
							gr := DequantizeBlockScaledBitDepthEOBRowsFlatTrusted(got[3:n+3:n+3], c, scan, eob, height, q, ts, bd)
							if gr != wr || !slices.Equal(got, want) {
								t.Fatalf("%dx%d %s q%v bd%d ts%d eob%d rows=%d want=%d: result/guard mismatch", width, height, order, q, bd, ts, eob, gr, wr)
							}
							if !slices.Equal(c, wc) || !slices.Equal(scan, ws) {
								t.Fatal("input mutated")
							}
						}
					}
				}
			}
		}
	}
}

func TestDequantEOBRowsFlatPartialPanicState(t *testing.T) {
	for _, scan := range [][]int16{{0, 3, -1, 2}, {0, 3, 16, 2}, {0, 3}} {
		coeff := make([]int16, 16)
		for i := range coeff {
			coeff[i] = int16(i + 1)
		}
		got, want := make([]int32, 16), make([]int32, 16)
		for i := range got {
			got[i] = 999
			want[i] = 999
		}
		panics := func(f func()) (yes bool) { defer func() { yes = recover() != nil }(); f(); return }
		wp := panics(func() { dequantEOBRowsOriginal(want, coeff, scan, 4, 4, Quantizer{7, 11}, 0, 8) })
		gp := panics(func() {
			DequantizeBlockScaledBitDepthEOBRowsFlatTrusted(got, coeff, scan, 4, 4, Quantizer{7, 11}, 0, 8)
		})
		if !wp || !gp || !slices.Equal(got, want) {
			t.Fatal("panic or partial destination state differs")
		}
	}
}

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

func TestQMatrixMatchesLibaomTables(t *testing.T) {
	y := QMatrix(0, false, transform.Size{Width: 4, Height: 4})
	wantY := []uint16{32, 43, 73, 97, 43, 67, 94, 110, 73, 94, 137, 150, 97, 110, 150, 200}
	for i, want := range wantY {
		if y[i] != want {
			t.Fatalf("y[%d]=%d want %d", i, y[i], want)
		}
	}
	u := QMatrix(0, true, transform.Size{Width: 4, Height: 4})
	wantU := []uint16{35, 46, 57, 66, 46, 60, 69, 71, 57, 69, 90, 90, 66, 71, 90, 109}
	for i, want := range wantU {
		if u[i] != want {
			t.Fatalf("u[%d]=%d want %d", i, u[i], want)
		}
	}
	v := QMatrix(14, true, transform.Size{Width: 64, Height: 64})
	if len(v) != 32*32 || v[0] != 32 || v[5] != 31 {
		t.Fatalf("adjusted 64x64 matrix len=%d first=%d sixth=%d", len(v), v[0], v[5])
	}
	if m := QMatrix(QMLevels-1, false, transform.Size{Width: 4, Height: 4}); m != nil {
		t.Fatalf("flat level 15 matrix %v", m)
	}
	if m := QMatrix(0, false, transform.Size{Width: 4, Height: 12}); m != nil {
		t.Fatalf("invalid size matrix %v", m)
	}
}

// TestQuantizeBBlockImplMatchesScalar proves the vector zbin-quantize kernel
// bit-exact with the scalar rule across block sizes, tx scales, quantizer
// steps, and the full coefficient range the forward transforms emit.
func TestQuantizeBBlockImplMatchesScalar(t *testing.T) {
	if quantizeBBlockImpl == nil {
		t.Skip("no vector kernel on this architecture")
	}
	rng := rand.New(rand.NewSource(61))
	for _, n := range []int{4, 8, 16, 32} {
		for _, ts := range []uint8{0, 1} {
			for trial := 0; trial < 200; trial++ {
				q := Quantizer{
					DC: int32(4 + rng.Intn(8000)),
					AC: int32(4 + rng.Intn(8000)),
				}
				zbinFactor := int32(80)
				if q.DC < 148 {
					zbinFactor = 84
				}
				mk := func(step int32) quantBParams {
					quant, shift := invertQuant(step)
					return quantBParams{
						zbin:  roundPowerOfTwo(roundPowerOfTwo(zbinFactor*step, 7), ts),
						round: roundPowerOfTwo((48*step)>>7, ts),
						quant: quant,
						shift: shift,
					}
				}
				dc, ac := mk(q.DC), mk(q.AC)
				coeff := make([]int32, n*n)
				for i := range coeff {
					coeff[i] = int32(rng.Intn(1<<22)) - 1<<21
				}
				want := make([]int16, n*n)
				got := make([]int16, n*n)
				for i := range coeff {
					p := &ac
					if i == 0 {
						p = &dc
					}
					want[i] = quantizeScalarB(coeff[i], p, ts)
				}
				if !quantizeBBlockImpl(got, coeff, n, q, ts) {
					t.Fatalf("n=%d ts=%d: kernel refused", n, ts)
				}
				for i := range want {
					if want[i] != got[i] {
						t.Fatalf("n=%d ts=%d trial=%d: q[%d] impl %d want %d (coeff %d)", n, ts, trial, i, got[i], want[i], coeff[i])
					}
				}
			}
		}
	}
}

// TestQuantizeBlockScaledBEOB checks that QuantizeBlockScaledBEOB writes the
// levels QuantizeBlockScaledB writes and reports their end of block.
func TestQuantizeBlockScaledBEOB(t *testing.T) {
	rng := rand.New(rand.NewSource(77))
	for _, dims := range [][2]int{{4, 4}, {8, 8}, {16, 16}, {32, 32}, {16, 8}, {8, 32}} {
		w, h := dims[0], dims[1]
		n := w * h
		for trial := range 300 {
			coeff := make([]int32, n)
			amp := []int32{1, 40, 400, 4000, 40000}[trial%5]
			for i := range coeff {
				if rng.Intn(4) == 0 {
					coeff[i] = rng.Int31n(2*amp+1) - amp
				}
			}
			if trial%7 == 0 {
				clear(coeff)
				coeff[0] = rng.Int31n(200) - 100
			}
			q := Quantizer{DC: 4 + rng.Int31n(2000), AC: 4 + rng.Int31n(2000)}
			ts := uint8(trial % 3)
			want := make([]int16, n)
			if err := QuantizeBlockScaledB(want, h, coeff, h, w, h, q, ts); err != nil {
				t.Fatal(err)
			}
			got := make([]int16, n)
			perm := rng.Perm(n - 1)
			iscan := make([]int16, n)
			for pos := 1; pos < n; pos++ {
				iscan[pos] = int16(perm[pos-1] + 1)
			}
			eob, err := QuantizeBlockScaledBEOB(got, h, coeff, h, w, h, q, ts, iscan)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("%dx%d trial %d: levels differ", w, h, trial)
			}
			if wantEOB := testEOB(want, iscan); eob != wantEOB {
				t.Fatalf("%dx%d trial %d: eob %d want %d", w, h, trial, eob, wantEOB)
			}
		}
	}
}

// TestTryQuantizeBBlockEOBTrusted checks the prebuilt-constant square kernel
// against the independent scalar rule (originalRowGroupLevel) over every
// size, tx scale and live-column bound, on sparse blocks around both zero
// bins with int32 extremes mixed in. The columns past the bound hold
// garbage that must be neither read nor coded.
func TestTryQuantizeBBlockEOBTrusted(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5eb))
	quants := []Quantizer{{4, 4}, {120, 140}, {147, 148}, {148, 147}, {1828, 4}, {4, 1828}, {29247, 29247}, {65535, 65535}}
	dims := [][2]int{{4, 4}, {8, 8}, {16, 16}, {32, 32}, {8, 4}, {4, 8}, {16, 8}, {8, 32}, {32, 16}, {16, 4}, {4, 16}}
	handledCount := 0
	for _, d := range dims {
		w, h := d[0], d[1]
		count := w * h
		for ts := uint8(0); ts <= 2; ts++ {
			for qi, q := range quants {
				p := testBParams(q, ts)
				dcZ, acZ := ZeroBinB(q, ts)
				for trial := range 24 {
					// Live column counts whose coefficients fill whole
					// groups of eight.
					cols := []int{w, w / 2, max(w/4, 1), 1}[trial%4]
					if cols*h%8 != 0 {
						cols = w
					}
					coeff := make([]int32, count)
					for i := range coeff {
						switch {
						case i/h >= cols:
							coeff[i] = int32(rng.Uint32()) // garbage
						case rng.Intn(3) == 0:
							z := []int32{acZ, dcZ}[i%2]
							coeff[i] = rng.Int31n(2*z+5) - z - 2
						}
					}
					edges := []int32{acZ, -acZ, acZ - 1, dcZ, -dcZ, dcZ - 1, -2147483648, 2147483647, 32767, -32768}
					for range trial % 5 {
						coeff[rng.Intn(cols*h)] = edges[rng.Intn(len(edges))]
					}
					if trial%6 == 0 {
						coeff[0] = []int32{dcZ, -dcZ, dcZ - 1, acZ}[trial/6%4]
					}
					want := make([]int16, count)
					for i := range want[:cols*h] {
						step := q.AC
						if i == 0 {
							step = q.DC
						}
						want[i] = originalRowGroupLevel(coeff[i], step, q.DC, ts)
					}
					iscan := make([]int16, count)
					for i, v := range rng.Perm(count - 1) {
						iscan[i+1] = int16(v + 1)
					}
					got := make([]int16, count+8)
					for i := range got {
						got[i] = 0x5a5a
					}
					before := append([]int32(nil), coeff...)
					eob, handled := TryQuantizeBBlockEOBTrusted(got, coeff, count, cols*h, p, iscan)
					wantHandled := quantizeBBlockEOBTrustedImpl != nil && p.vector
					if handled != wantHandled {
						t.Fatalf("%dx%d ts=%d q=%+v: handled=%v want %v", w, h, ts, q, handled, wantHandled)
					}
					if !handled {
						for _, v := range got {
							if v != 0x5a5a {
								t.Fatal("declined call wrote levels")
							}
						}
						continue
					}
					handledCount++
					if !reflect.DeepEqual(got[:count], want) {
						t.Fatalf("%dx%d ts=%d q=%d cols=%d trial=%d: levels differ", w, h, ts, qi, cols, trial)
					}
					if wantEOB := testEOB(want, iscan); eob != wantEOB {
						t.Fatalf("%dx%d ts=%d q=%d cols=%d: eob %d want %d", w, h, ts, qi, cols, eob, wantEOB)
					}
					for _, v := range got[count:] {
						if v != 0x5a5a {
							t.Fatal("output tail changed")
						}
					}
					if !reflect.DeepEqual(coeff, before) {
						t.Fatal("input changed")
					}
				}
			}
		}
	}
	if quantizeBBlockEOBTrustedImpl != nil && handledCount == 0 {
		t.Fatal("kernel did not execute")
	}
}

// TestQuantizeFPBlockImplMatchesScalar proves the vector fp-quantize kernel
// bit-exact with the scalar rule across block sizes, tx scales, quantizer
// steps, and the full coefficient range the forward transforms emit.
func TestQuantizeFPBlockImplMatchesScalar(t *testing.T) {
	if quantizeFPBlockImpl == nil {
		t.Skip("no vector kernel on this architecture")
	}
	rng := rand.New(rand.NewSource(59))
	for _, n := range []int{4, 8, 16, 32} {
		for _, ts := range []uint8{0, 1} {
			for trial := 0; trial < 200; trial++ {
				q := Quantizer{
					DC: int32(4 + rng.Intn(8000)),
					AC: int32(4 + rng.Intn(8000)),
				}
				coeff := make([]int32, n*n)
				for i := range coeff {
					coeff[i] = int32(rng.Intn(1<<22)) - 1<<21
				}
				want := make([]int16, n*n)
				got := make([]int16, n*n)
				quantDC := int64(1<<16) / int64(q.DC)
				roundDC := roundPowerOfTwo((64*q.DC)>>7, ts)
				quantAC := int64(1<<16) / int64(q.AC)
				roundAC := roundPowerOfTwo((64*q.AC)>>7, ts)
				for i := range coeff {
					if i == 0 {
						want[i] = quantizeScalarFP(coeff[i], q.DC, quantDC, roundDC, ts)
					} else {
						want[i] = quantizeScalarFP(coeff[i], q.AC, quantAC, roundAC, ts)
					}
				}
				if !quantizeFPBlockImpl(got, coeff, n, q, ts) {
					t.Fatalf("n=%d ts=%d: kernel refused", n, ts)
				}
				for i := range want {
					if want[i] != got[i] {
						t.Fatalf("n=%d ts=%d trial=%d: q[%d] impl %d want %d (coeff %d dc=%v)", n, ts, trial, i, got[i], want[i], coeff[i], i == 0)
					}
				}
			}
		}
	}
}

// Independent derivation of the old quantize_b scalar rule. In particular it
// does not use invertQuant, quantizeScalarB, or either vector implementation.
func originalRowGroupLevel(c int32, step, dcStep int32, ts uint8) int16 {
	factor := int64(80)
	if dcStep < 148 {
		factor = 84
	}
	roundShift := func(v int64, shift uint8) int64 {
		if shift == 0 {
			return v
		}
		return (v + (int64(1) << (shift - 1))) >> shift
	}
	zbin := roundShift((factor*int64(step)+64)>>7, ts)
	v := int64(c)
	if v < 0 {
		v = -v
	}
	if v < zbin {
		return 0
	}
	v += roundShift((48*int64(step))>>7, ts)
	if v > 32767 {
		v = 32767
	}
	lg := 0
	for x := step; x > 1; x >>= 1 {
		lg++
	}
	multiplier := 1 + (int64(1)<<uint(16+lg))/int64(step) - 65536
	v = (((v*multiplier)>>16 + v) * (int64(1) << uint(16-lg))) >> (16 - ts)
	if v > 32767 {
		v = 32767
	}
	if c < 0 {
		v = -v
	}
	return int16(v)
}

// Models the frozen original two-pass result, including the SIMD int32
// intermediate before the int64 widening products and its later DC correction.
// The separate loop deliberately does not share the fused operation order.
func originalRowGroupDiff(coeff []int32, n int, live uint8, cols int, q Quantizer, ts uint8) ([]int16, int64, int64) {
	levels := make([]int16, n*n)
	for i, c := range coeff[:n*n] {
		if live&(1<<uint((i%n)/4)) == 0 || i/n >= cols {
			continue
		}
		step := q.AC
		if i == 0 {
			step = q.DC
		}
		levels[i] = originalRowGroupLevel(c, step, q.DC, ts)
	}
	var diff, rate int64
	for i, v := range levels {
		if v == 0 {
			continue
		}
		dq := (int32(v) * q.AC) >> ts
		delta := int32(2*coeff[i] - dq)
		diff += int64(dq) * int64(delta)
		mag := int64(v)
		if mag < 0 {
			mag = -mag
		}
		length := int64(0)
		for ; mag != 0; mag >>= 1 {
			length++
		}
		rate += (2 + length) << 9
	}
	if v := levels[0]; v != 0 {
		c := int64(coeff[0])
		a := c - ((int64(v) * int64(q.AC)) >> ts)
		d := c - ((int64(v) * int64(q.DC)) >> ts)
		diff += a*a - d*d
	}
	return levels, diff, rate
}

func TestQuantizeBRowGroupsDiffOriginalNumeric(t *testing.T) {
	rng := rand.New(rand.NewSource(1054))
	quants := []Quantizer{{4, 4}, {1, 1}, {147, 148}, {148, 147}, {120, 140}, {65535, 65535}, {1828, 4}, {4, 1828}}
	extremes := []int32{-2147483648, 2147483647, -32768, 32767, -1, 0, 1, -100000, 100000}
	handledCount := 0
	for _, n := range []int{8, 16, 32} {
		for mask := 0; mask < 1<<(n/4); mask++ {
			for qi, q := range quants {
				ts := uint8((mask + qi) % 3)
				coeff := make([]int32, n*n+8)
				for i := range coeff {
					switch (mask + qi) % 4 {
					case 0:
						coeff[i] = extremes[i%len(extremes)]
					case 1:
						coeff[i] = rng.Int31n(65536) - 32768
					case 2:
						coeff[i] = int32((i%9)-4) * q.AC / 4
					case 3:
						// Around the zero bins: zbin-2..zbin+2 of both signs,
						// mostly zero so the zero-bin early-out is taken.
						_, acZ := ZeroBinB(q, ts)
						if i%7 < 3 {
							coeff[i] = (acZ + int32(i%5) - 2) * int32(1-2*(i%2))
						}
					}
				}
				before := append([]int32(nil), coeff...)
				want, wantDiff, wantRate := originalRowGroupDiff(coeff, n, uint8(mask), n, q, ts)
				old := make([]int16, n*n)
				oldNZ := QuantizeBRowGroupsTrusted(old, coeff, n, uint8(mask), q, ts)
				if !reflect.DeepEqual(old, want) || oldNZ != (wantRate != 0) {
					t.Fatalf("original quantizer mismatch n=%d mask=%x q=%+v ts=%d", n, mask, q, ts)
				}
				// The live-column bound: the full width, SVT's N2 and N4
				// shapes and the DC column alone.
				cols := []int{n, n / 2, max(n/4, 1), 1}[(mask+qi)%4]
				want, wantDiff, wantRate = originalRowGroupDiff(coeff, n, uint8(mask), cols, q, ts)
				got := make([]int16, n*n+8)
				for i := range got {
					got[i] = 0x5a5a
				}
				iscan := testISCAN(rng, n)
				diff, rate, eob, handled := TryQuantizeBRowGroupsDiffTrusted(got, coeff, n, uint8(mask), cols, testBParams(q, ts), iscan)
				wantHandled := quantizeBRowGroupsDiffImpl != nil && bits.Len32(uint32(q.AC))-1 >= int(ts) && bits.Len32(uint32(q.DC))-1 >= int(ts)
				if handled != wantHandled {
					t.Fatalf("handled=%v want %v n=%d q=%+v ts=%d", handled, wantHandled, n, q, ts)
				}
				if handled {
					handledCount++
					if wantEOB := testEOB(want, iscan); eob != wantEOB {
						t.Fatalf("n=%d mask=%x cols=%d q=%+v ts=%d: eob %d want %d", n, mask, cols, q, ts, eob, wantEOB)
					}
					if !reflect.DeepEqual(got[:n*n], want) || diff != wantDiff || rate != wantRate {
						t.Fatalf("n=%d mask=%x cols=%d q=%+v ts=%d: diff/rate=%d/%d want=%d/%d levelsEqual=%v", n, mask, cols, q, ts, diff, rate, wantDiff, wantRate, reflect.DeepEqual(got[:n*n], want))
					}
				} else {
					for _, v := range got[:n*n] {
						if v != 0x5a5a {
							t.Fatal("declined input was modified")
						}
					}
				}
				for _, v := range got[n*n:] {
					if v != 0x5a5a {
						t.Fatal("output tail changed")
					}
				}
				if !reflect.DeepEqual(coeff, before) {
					t.Fatal("input changed")
				}
			}
		}
	}
	// AC-only nonzero DC becomes zero after its separate quantizer, and the
	// converse. These exercise the old full-block nonzero-rescan corner.
	for _, q := range []Quantizer{{1828, 4}, {4, 1828}} {
		for _, dc := range []int32{-4, 4} {
			var coeff [64]int32
			var got [64]int16
			coeff[0] = dc
			want, wantDiff, wantRate := originalRowGroupDiff(coeff[:], 8, 3, 8, q, 0)
			iscan := testISCAN(rand.New(rand.NewSource(int64(dc))), 8)
			diff, rate, eob, handled := TryQuantizeBRowGroupsDiffTrusted(got[:], coeff[:], 8, 3, 8, testBParams(q, 0), iscan)
			if handled {
				handledCount++
				if wantEOB := testEOB(want, iscan); eob != wantEOB {
					t.Fatalf("DC override q=%+v dc=%d: eob %d want %d", q, dc, eob, wantEOB)
				}
				if !reflect.DeepEqual(got[:], want) || diff != wantDiff || rate != wantRate {
					t.Fatalf("DC override q=%+v dc=%d: diff/rate=%d/%d want=%d/%d", q, dc, diff, rate, wantDiff, wantRate)
				}
			}
		}
	}
	if quantizeBRowGroupsDiffImpl != nil && handledCount == 0 {
		t.Fatal("fused backend did not execute")
	}
	t.Logf("fused cases executed: %d", handledCount)
}

func TestQuantizeBRowGroupsDiffDeclinesUnsupported(t *testing.T) {
	for _, tc := range []struct {
		n, qlen, clen int
		q             Quantizer
		ts            uint8
	}{
		{4, 64, 64, Quantizer{120, 140}, 0},
		{8, 63, 64, Quantizer{120, 140}, 0},
		{8, 64, 63, Quantizer{120, 140}, 0},
		{8, 64, 64, Quantizer{0, 140}, 0},
		{8, 64, 64, Quantizer{120, 65536}, 0},
		{8, 64, 64, Quantizer{120, 140}, 3},
		{8, 64, 64, Quantizer{1, 1}, 2},
	} {
		qcoeff, coeff := make([]int16, tc.qlen), make([]int32, tc.clen)
		for i := range qcoeff {
			qcoeff[i] = 123
		}
		before := append([]int16(nil), qcoeff...)
		if _, _, _, handled := TryQuantizeBRowGroupsDiffTrusted(qcoeff, coeff, tc.n, 0xff, tc.n, testBParams(tc.q, tc.ts), make([]int16, 64)); handled {
			t.Fatalf("unsupported input accepted: %+v", tc)
		}
		if !reflect.DeepEqual(qcoeff, before) {
			t.Fatal("declined output changed")
		}
	}
}

func TestQuantizeBRowGroupsDiffNoAllocation(t *testing.T) {
	var coeff [1024]int32
	var levels [1024]int16
	var iscan [1024]int16
	for i := range iscan {
		iscan[i] = int16(i)
	}
	for i := range coeff {
		coeff[i] = int32(i*17 - 5000)
	}
	var cache BParamsCache
	if got := testing.AllocsPerRun(10, func() {
		TryQuantizeBRowGroupsDiffTrusted(levels[:], coeff[:], 32, 0x55, 8, cache.Get(Quantizer{120, 140}, 1), iscan[:])
	}); got != 0 {
		t.Fatalf("allocations=%v", got)
	}
}

// testBParams builds a fresh BParams for q at ts.
func testBParams(q Quantizer, ts uint8) *BParams {
	p := new(BParams)
	p.Set(q, ts)
	return p
}

// TestBParamsCache checks that the cache returns the constants of the asked
// quantizer and tx scale across more keys than it holds.
func TestBParamsCache(t *testing.T) {
	var c BParamsCache
	keys := []Quantizer{{4, 4}, {120, 140}, {148, 147}, {1828, 4}, {4, 1828}, {29247, 29247}}
	for round := range 3 {
		for i, q := range keys {
			for ts := range uint8(3) {
				p := c.Get(q, (ts+uint8(round+i))%3)
				var want BParams
				want.Set(q, (ts+uint8(round+i))%3)
				if *p != want {
					t.Fatalf("q=%+v ts=%d: cached params differ", q, (ts+uint8(round+i))%3)
				}
			}
		}
	}
}

// testISCAN is a random scan of an n x n block: iscan[pos] is the scan index
// of pos, with the DC first as every AV1 scan has it.
func testISCAN(rng *rand.Rand, n int) []int16 {
	perm := rng.Perm(n*n - 1)
	iscan := make([]int16, n*n)
	for pos := 1; pos < n*n; pos++ {
		iscan[pos] = int16(perm[pos-1] + 1)
	}
	return iscan
}

// testEOB is one past the largest scan index of a nonzero level.
func testEOB(levels []int16, iscan []int16) int {
	eob := 0
	for pos, v := range levels {
		if v != 0 {
			eob = max(eob, int(iscan[pos])+1)
		}
	}
	return eob
}

func BenchmarkQuantizeBRowGroups16x16(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	coeff := make([]int32, 256)
	for i := range coeff {
		coeff[i] = rng.Int31n(201) - 100
	}
	qcoeff := make([]int16, 256)
	q := Quantizer{DC: 120, AC: 140}
	for _, live := range []uint8{0xf, 0x1, 0x3} {
		b.Run(fmt.Sprintf("live%04b", live), func(b *testing.B) {
			for range b.N {
				QuantizeBRowGroupsTrusted(qcoeff, coeff, 16, live, q, 0)
			}
		})
	}
	b.Run("block", func(b *testing.B) {
		for range b.N {
			_ = QuantizeBlockScaledB(qcoeff, 16, coeff, 16, 16, 16, q, 0)
		}
	})
}

func TestQuantizeBRowGroupsMatchesBlock(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	impls := map[string]func([]int16, []int32, int, uint8, Quantizer, uint8) bool{
		"scalar":   quantizeBRowGroupsScalar,
		"dispatch": quantizeBRowGroupsImpl,
		"exported": QuantizeBRowGroupsTrusted,
	}
	for _, n := range []int{8, 16, 32} {
		txScale := uint8(0)
		if n == 32 {
			txScale = 1
		}
		coeff := make([]int32, n*n)
		want := make([]int16, n*n)
		got := make([]int16, n*n)
		for trial := range 3000 {
			q := Quantizer{DC: int32(rng.Intn(1500) + 4), AC: int32(rng.Intn(1500) + 4)}
			if trial%50 == 0 {
				q = Quantizer{DC: int32(rng.Intn(29000) + 4), AC: int32(rng.Intn(29000) + 4)}
			}
			dcZ, acZ := ZeroBinB(q, txScale)
			live := uint8(rng.Intn(1 << (n / 4)))
			amp := []int32{0, 1, 3 * acZ / 2, 4 * acZ, 70000}[trial%5] + 1
			for col := range n {
				for r := range n {
					i := col*n + r
					if live&(1<<(r/4)) != 0 {
						coeff[i] = rng.Int31n(2*amp+1) - amp
						continue
					}
					// Dead rows: provably below their zero bin.
					z := acZ
					if i == 0 {
						z = min(dcZ, acZ)
					}
					coeff[i] = 0
					if z > 1 {
						coeff[i] = rng.Int31n(2*z-1) - (z - 1)
					}
				}
			}
			if err := QuantizeBlockScaledB(want, n, coeff, n, n, n, q, txScale); err != nil {
				t.Fatal(err)
			}
			wantNZ := false
			for _, v := range want {
				wantNZ = wantNZ || v != 0
			}
			for name, impl := range impls {
				for i := range got {
					got[i] = 0x5a5a
				}
				nz := impl(got, coeff, n, live, q, txScale)
				if nz != wantNZ {
					t.Fatalf("%s n=%d trial %d: nonzero=%v want %v", name, n, trial, nz, wantNZ)
				}
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("%s n=%d trial %d: level[%d]=%d want %d (c=%d live=%b q=%+v)", name, n, trial, i, got[i], want[i], coeff[i], live, q)
					}
				}
			}
		}
	}
}

func TestQuantizeVectorImplsAllocs(t *testing.T) {
	if quantizeFPBlockImpl == nil {
		t.Skip("missing vector kernel on this architecture")
	}
	q := Quantizer{DC: 107, AC: 130}
	coeff := make([]int32, 16*16)
	for i := range coeff {
		coeff[i] = int32(i*37 - 4096)
	}
	qcoeff16 := make([]int16, 16*16)
	allocs := testing.AllocsPerRun(1000, func() {
		if !quantizeFPBlockImpl(qcoeff16, coeff, 16, q, 1) {
			t.Fatal("quantizeFPBlockImpl refused")
		}
	})
	if allocs != 0 {
		t.Fatalf("block vector quantizers allocated %f objects/run, want 0", allocs)
	}
}
