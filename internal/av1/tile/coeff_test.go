package tile

import (
	"errors"
	"slices"
	"testing"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestCoeffQContextMatchesLibaom(t *testing.T) {
	tests := []struct {
		q    uint8
		want uint8
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

// TestCoeffCDFsInitDefaultHighQIndexMatchesLibaom pins the q-context=3 (high
// qindex, baseQIndex > 120) coefficient CDFs against libaom's exact values
// from av1/common/token_cdfs.h. This is the q-context selected for the 10-bit
// quantizer_32 (baseQIdx=128) and quantizer_63 (baseQIdx=255) test vectors;
// regressions in the q-context-3 CDF initialization would silently break
// entropy decode for high-qindex 10-bit streams while leaving the
// q-context-0 default cohort intact. The expected values are stored AS the
// inverse-CDF form (CDFProbTop minus the cumulative-probability constants in
// token_cdfs.h), matching what CoeffCDFs.InitDefault writes into entropy.CDF.
//
// See decodetxb-side note: bit-exact decode of the q32/q63 10-bit first-block
// luma 8x8 DCT_DCT TXB was verified against libaom-NEON, confirming the
// q-context-3 CDFs are correctly wired through the high-qindex branch of
// CoeffQContext.
func TestCoeffCDFsInitDefaultHighQIndexMatchesLibaom(t *testing.T) {
	// baseQIdx=128 (10-bit q32) and 255 (10-bit q63) both map to q-context 3.
	if got := CoeffQContext(128); got != 3 {
		t.Fatalf("CoeffQContext(128)=%d want 3 (10-bit q32 baseQIdx)", got)
	}
	if got := CoeffQContext(255); got != 3 {
		t.Fatalf("CoeffQContext(255)=%d want 3 (10-bit q63 baseQIdx)", got)
	}
	if got := CoeffQContext(121); got != 3 {
		t.Fatalf("CoeffQContext(121)=%d want 3 (q-ctx 3 lower bound)", got)
	}

	var cdfs CoeffCDFs
	if err := cdfs.InitDefault(128); err != nil {
		t.Fatal(err)
	}

	// Each `want` is the inverse-CDF form expected after InitCDF. The
	// cumulative-probability constants are sourced from libaom's
	// av1/common/token_cdfs.h at q-context index 3 (the fourth row of each
	// outer table). For AOM_CDF4(a0,a1,a2) the storage is
	// {CDFProbTop-a0, CDFProbTop-a1, CDFProbTop-a2, 0, 0}.
	tests := []struct {
		name string
		cdf  *entropy.CDF
		want []uint16
	}{
		// av1_default_dc_sign_cdfs[3][PLANE_Y][0] = AOM_CDF2(128*125=16000).
		// q-context 3 dc-sign matches q-context 0; the entire table is
		// replicated across all four q-context indices in libaom.
		{name: "dc sign y ctx0 qctx3", cdf: &cdfs.DCSign[CoeffPlaneY][0], want: []uint16{16768, 0, 0}},

		// av1_default_txb_skip_cdfs[3][TX_4X4][0] = AOM_CDF2(26887).
		// Stored as {32768-26887, 32768-32768, 0} = {5881, 0, 0}.
		{name: "txb skip 4x4 ctx0 qctx3", cdf: &cdfs.TXBSkip[0][0], want: []uint16{5881, 0, 0}},

		// av1_default_coeff_base_multi_cdfs[3][TX_4X4][PLANE_Y][0] =
		// AOM_CDF4(7062, 16472, 22319). q-context 3 lives in the high
		// qindex regime where the DC-only base CDF skews coarser than
		// q-context 0.
		{name: "coeff base 4x4 y ctx0 qctx3", cdf: &cdfs.CoeffBase[0][CoeffPlaneY][0], want: []uint16{25706, 16296, 10449, 0, 0}},

		// av1_default_coeff_lps_multi_cdfs[3][TX_4X4][PLANE_Y][0] =
		// AOM_CDF4(18315, 24289, 27551). Used by the base-range reader
		// when level > NUM_BASE_LEVELS at q-ctx 3.
		{name: "coeff br 4x4 y ctx0 qctx3", cdf: &cdfs.CoeffBR[0][CoeffPlaneY][0], want: []uint16{14453, 8479, 5217, 0, 0}},

		// av1_default_coeff_base_eob_multi_cdfs[3][TX_4X4][PLANE_Y][0] =
		// AOM_CDF3(22497, 31198). The eob-base reader used at the last
		// scan position; mis-selecting this CDF would skew the very
		// first non-zero level read for every high-qindex block.
		{name: "coeff base eob 4x4 y ctx0 qctx3", cdf: &cdfs.CoeffBaseEOB[0][CoeffPlaneY][0], want: []uint16{10271, 1570, 0, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertEntropyCDFValues(t, tt.cdf.Values(), tt.want)
		})
	}

	// Also pin the q-context=3 CDFs for baseQIdx=255 so a future regression
	// that special-cases the upper qindex bound does not silently diverge
	// from baseQIdx=128.
	var q255 CoeffCDFs
	if err := q255.InitDefault(255); err != nil {
		t.Fatal(err)
	}
	if got, want := q255.TXBSkip[0][0].Values(), cdfs.TXBSkip[0][0].Values(); !cdfValuesEqual(got, want) {
		t.Fatalf("baseQIdx=255 TXBSkip diverges from baseQIdx=128: got=%v want=%v", got, want)
	}
	if got, want := q255.CoeffBase[0][CoeffPlaneY][0].Values(), cdfs.CoeffBase[0][CoeffPlaneY][0].Values(); !cdfValuesEqual(got, want) {
		t.Fatalf("baseQIdx=255 CoeffBase diverges from baseQIdx=128: got=%v want=%v", got, want)
	}
}

func cdfValuesEqual(a, b []uint16) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
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
	mustSetCoeffLevel(t, levels, TransformSize4x4, 1, MaxBaseBRRange+9)
	mustSetCoeffLevel(t, levels, TransformSize4x4, 4, MaxBaseBRRange+7)
	mustSetCoeffLevel(t, levels, TransformSize4x4, 5, MaxBaseBRRange+5)
	if got, err := CoeffBRContext(levels, TransformSize4x4, transform.Class2D, 0); err != nil || got != 6 {
		t.Fatalf("CoeffBRContext saturated 2d dc=%d err=%v want 6", got, err)
	}
	if got, err := CoeffBRContext(levels, TransformSize4x4, transform.Class2D, 1); err != nil || got != 13 {
		t.Fatalf("CoeffBRContext saturated 2d non-dc=%d err=%v want 13", got, err)
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
		// libaom indexes av1_nz_map_ctx_offset[tx_size] by the *unadjusted*
		// tx_size, so 32x64 / 64x32 transforms (whose scan layout reduces to
		// the 32x32 square after av1_get_adjusted_tx_size) still receive the
		// row<2/+11 or col<2/+16 rectangular bias. The first-row / first-col
		// positions therefore land on the rectangular branch even though the
		// adjusted scan grid is square; the canonical libaom table at
		// av1_nz_map_ctx_offset_32x64[1] = 11 and
		// av1_nz_map_ctx_offset_32x64[2] = 6 reflects the unadjusted bias.
		{name: "32x64 early row col 0", size: TransformSize32x64, class: transform.Class2D, index: 1, want: 11},
		{name: "32x64 early row col 1", size: TransformSize32x64, class: transform.Class2D, index: 32, want: 11},
		{name: "32x64 middle band col 0", size: TransformSize32x64, class: transform.Class2D, index: 2, want: 6},
		{name: "32x64 outer band col 0", size: TransformSize32x64, class: transform.Class2D, index: 4, want: 21},
		{name: "64x32 early col row 0", size: TransformSize64x32, class: transform.Class2D, index: 32, want: 16},
		{name: "64x32 early col row 1", size: TransformSize64x32, class: transform.Class2D, index: 33, want: 16},
		{name: "64x32 middle band row 0", size: TransformSize64x32, class: transform.Class2D, index: 64, want: 6},
		{name: "64x32 outer band row 0", size: TransformSize64x32, class: transform.Class2D, index: 128, want: 21},
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

func TestCoeffInitLevelsMatchesLibaomEncodeTxbInitLevel(t *testing.T) {
	for rawSize := range transformSizeCount {
		txSize, err := rawSize.TransformSize()
		if err != nil {
			t.Fatal(err)
		}
		scanSize, err := transform.ScanSize(txSize)
		if err != nil {
			t.Fatal(err)
		}
		scanWidth := int(scanSize.Width)
		scanHeight := int(scanSize.Height)
		maxEOB := scanWidth * scanHeight
		coeffs := make([]int16, maxEOB)
		for i := range maxEOB {
			switch i % 7 {
			case 0:
				coeffs[i] = -32768
			case 1:
				coeffs[i] = -128
			case 2:
				coeffs[i] = -127
			case 3:
				coeffs[i] = -1
			default:
				coeffs[i] = int16((i * 37) - 300)
			}
		}
		scratchLen, err := CoeffLevelsScratchLen(rawSize)
		if err != nil {
			t.Fatal(err)
		}
		levels := make([]uint8, scratchLen)
		for i := range levels {
			levels[i] = 0xff
		}
		if err := CoeffInitLevels(coeffs, rawSize, levels); err != nil {
			t.Fatal(err)
		}

		stride := scanHeight + txPadHorizontal
		for col := 0; col < scanWidth; col++ {
			for row := 0; row < scanHeight; row++ {
				idx := col*scanHeight + row
				if got, want := levels[col*stride+row], coeffAbsClamp127(coeffs[idx]); got != want {
					t.Fatalf("size=%d level[%d,%d]=%d want %d", rawSize, row, col, got, want)
				}
			}
			for row := scanHeight; row < scanHeight+txPadHorizontal; row++ {
				if got := levels[col*stride+row]; got != 0 {
					t.Fatalf("size=%d horizontal pad[%d,%d]=%d want 0", rawSize, row, col, got)
				}
			}
		}
		for col := scanWidth; col < scanWidth+txPadHorizontal; col++ {
			for row := range stride {
				if got := levels[col*stride+row]; got != 0 {
					t.Fatalf("size=%d bottom pad[%d,%d]=%d want 0", rawSize, row, col, got)
				}
			}
		}
	}
}

func TestCoeffGeometryPositionTablesFitScratch(t *testing.T) {
	for size := range transformSizeCount {
		geo := coeffGeometryTable[size]
		if !geo.valid {
			continue
		}
		posSlice := coeffPosTable[size]
		if len(posSlice) < int(geo.maxEOB) {
			t.Fatalf("size=%d pos table len=%d maxEOB=%d", size, len(posSlice), geo.maxEOB)
		}
		stride := int(geo.stride)
		scratchLen := int(geo.scratchLen)
		for pos := 0; pos < int(geo.maxEOB); pos++ {
			p := posSlice[pos]
			padded := int(p.padded)
			if padded < 0 || padded >= scratchLen {
				t.Fatalf("size=%d pos=%d padded=%d scratch=%d", size, pos, padded, scratchLen)
			}
			maxLower := maxInt(padded+4, padded+4*stride)
			maxBR := maxInt(padded+4, padded+2*stride+2)
			if maxLower >= scratchLen {
				t.Fatalf("size=%d pos=%d lower max=%d scratch=%d", size, pos, maxLower, scratchLen)
			}
			if maxBR >= scratchLen {
				t.Fatalf("size=%d pos=%d br max=%d scratch=%d", size, pos, maxBR, scratchLen)
			}
		}
	}
}

func TestCoeffNZMapContextsMatchesLibaomEncodeTxb(t *testing.T) {
	rnd := newCoeffContextRandom(0x1532a5)
	for isInter := range 2 {
		_ = isInter
		for rawType := range uint8(transform.TypeCount) {
			typ := transform.Type(rawType)
			class, err := typ.Class()
			if err != nil {
				t.Fatal(err)
			}
			for size := range transformSizeCount {
				if !libaomEncodeTXBTypeValid(size, typ) {
					continue
				}
				txSize, err := size.TransformSize()
				if err != nil {
					t.Fatal(err)
				}
				scanSize, err := transform.ScanSize(txSize)
				if err != nil {
					t.Fatal(err)
				}
				maxEOB := int(scanSize.Width) * int(scanSize.Height)
				scan := make([]int16, maxEOB)
				inverse := make([]int16, maxEOB)
				if err := transform.FillDefaultScan(scan, inverse, txSize, class); err != nil {
					t.Fatal(err)
				}
				levels := make([]uint8, mustCoeffLevelsScratchLen(t, size))
				for range 3 {
					for _, eob := range coeffNZMapEOBCases(maxEOB) {
						for i := range levels {
							levels[i] = 0
						}
						contexts := make([]int8, maxEOB)
						want := make([]int8, maxEOB)
						for c := range eob {
							pos := int(scan[c])
							mustSetCoeffLevel(t, levels, size, pos, int(rnd.u8()&0x7f))
							contexts[pos] = int8(rnd.u16() >> 9)
							want[pos] = contexts[pos]
						}
						copy(want, contexts)
						for c := range eob {
							pos := int(scan[c])
							var ctx int
							if c == eob-1 {
								ctx, err = transform.LowerLevelsCtxEOB(scanSize, c)
							} else {
								ctx, err = CoeffLowerLevelsContext(levels, size, class, pos)
							}
							if err != nil {
								t.Fatal(err)
							}
							want[pos] = int8(ctx)
						}
						if err := CoeffNZMapContexts(levels, size, class, scan, eob, contexts); err != nil {
							t.Fatal(err)
						}
						for i := range maxEOB {
							if contexts[i] != want[i] {
								t.Fatalf("size=%d type=%d eob=%d context[%d]=%d want %d", size, typ, eob, i, contexts[i], want[i])
							}
						}
					}
				}
			}
		}
	}
}

func TestCoeffTXBLevelHelpersDoNotAllocate(t *testing.T) {
	var coeffs [32 * 32]int16
	var levels [(32 + txPadHorizontal) * (32 + txPadHorizontal)]uint8
	var scan [32 * 32]int16
	var inverse [32 * 32]int16
	var contexts [32 * 32]int8
	for i := range coeffs {
		coeffs[i] = int16(i - 512)
	}
	if err := transform.FillDefaultScan(scan[:], inverse[:], transform.Size{Width: 32, Height: 32}, transform.Class2D); err != nil {
		t.Fatal(err)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := CoeffInitLevels(coeffs[:], TransformSize32x32, levels[:]); err != nil {
			t.Fatal(err)
		}
		if err := CoeffNZMapContexts(levels[:], TransformSize32x32, transform.Class2D, scan[:], 1024, contexts[:]); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("coeff txb helpers allocated: %f", allocs)
	}
}

func TestCoeffNZMapContextsRejectsInvalidScanPosition(t *testing.T) {
	size := TransformSize16x16
	geo := coeffGeometryTable[size]
	levels := make([]uint8, int(geo.scratchLen))
	contexts := make([]int8, int(geo.maxEOB))
	scan := append([]int16(nil), coeffScanTable[size][transform.Class2D]...)
	scan[2] = int16(geo.maxEOB)
	if err := CoeffNZMapContexts(levels, size, transform.Class2D, scan, 3, contexts); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("large scan position err=%v want %v", err, ErrInvalidDecodeState)
	}
	scan[2] = -1
	if err := CoeffNZMapContexts(levels, size, transform.Class2D, scan, 3, contexts); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("negative scan position err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func indexOfNonZero(levels []uint8) int {
	for i, v := range levels {
		if v != 0 {
			return i
		}
	}
	return -1
}

var coeffBenchmarkSink int

func BenchmarkCoeffCDFsInitDefault(b *testing.B) {
	var cdfs CoeffCDFs
	q := uint8(0)
	sum := 0
	b.ReportAllocs()

	for b.Loop() {
		if err := cdfs.InitDefault(q); err != nil {
			b.Fatal(err)
		}
		sum += int(cdfs.TXBSkip[0][0].Values()[0])
		q += 73
	}
	coeffBenchmarkSink = sum
}

func TestTracked2DPackedLevelContextFormulas(t *testing.T) {
	pack := func(level int) uint8 {
		clipped := min(level, 3)
		return uint8(level | clipped<<6)
	}
	for level := 0; level <= 15; level++ {
		packed := pack(level)
		if got := int(packed & 0x3f); got != level {
			t.Fatalf("packed level %d raw bits=%d", level, got)
		}
		if got, want := int(packed>>6), min(level, 3); got != want {
			t.Fatalf("packed level %d clipped bits=%d want=%d", level, got, want)
		}
	}
	clip3 := func(v int) int { return min(v, 3) }
	for a := 0; a <= 15; a++ {
		for b := 0; b <= 15; b++ {
			for c := 0; c <= 15; c++ {
				packedH := int(pack(a)) + int(pack(b)) + int(pack(c))
				gotBR := ((packedH & 63) + 1) >> 1
				if gotBR > 6 {
					gotBR = 6
				}
				wantBR := (a + b + c + 1) >> 1
				if wantBR > 6 {
					wantBR = 6
				}
				if gotBR != wantBR {
					t.Fatalf("BR context (%d,%d,%d)=%d want=%d", a, b, c, gotBR, wantBR)
				}
			}
		}
	}
	for a := 0; a <= 15; a++ {
		for b := 0; b <= 15; b++ {
			for c := 0; c <= 15; c++ {
				for d := 0; d <= 15; d++ {
					for e := 0; e <= 15; e++ {
						packedSum := int(pack(a)) + int(pack(b)) + int(pack(c)) + int(pack(d)) + int(pack(e))
						got := (packedSum + 64) >> 7
						if got > 4 {
							got = 4
						}
						clippedSum := clip3(a) + clip3(b) + clip3(c) + clip3(d) + clip3(e)
						want := (clippedSum + 1) >> 1
						if want > 4 {
							want = 4
						}
						if got != want {
							t.Fatalf("low context (%d,%d,%d,%d,%d)=%d want=%d", a, b, c, d, e, got, want)
						}
					}
				}
			}
		}
	}
}

func coeffScanAndScratch(t testing.TB, size TransformSize, txSize transform.Size, class transform.Class) ([]int16, []uint8) {
	t.Helper()
	scanSize, err := transform.ScanSize(txSize)
	if err != nil {
		t.Fatal(err)
	}
	scan := make([]int16, int(scanSize.Width)*int(scanSize.Height))
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

func mustCoeffLevelsScratchLen(t *testing.T, size TransformSize) int {
	t.Helper()
	n, err := CoeffLevelsScratchLen(size)
	if err != nil {
		t.Fatal(err)
	}
	return n
}

func libaomEncodeTXBTypeValid(size TransformSize, typ transform.Type) bool {
	up, err := TransformSizeSquareUp(size)
	if err != nil {
		return false
	}
	set := ExtTXSetAll16
	if up > TransformSize32x32 {
		set = ExtTXSetDCTOnly
	} else if up == TransformSize32x32 {
		set = ExtTXSetDCTIDTX
	}
	ok, err := ExtTXTypeAllowed(set, typ)
	return err == nil && ok
}

func coeffNZMapEOBCases(maxEOB int) []int {
	candidates := [...]int{1, 2, maxEOB / 8, maxEOB/8 + 1, maxEOB / 4, maxEOB/4 + 1, maxEOB / 2, maxEOB}
	out := make([]int, 0, len(candidates))
	for _, eob := range candidates {
		if eob <= 0 || eob > maxEOB {
			continue
		}
		seen := slices.Contains(out, eob)
		if !seen {
			out = append(out, eob)
		}
	}
	return out
}

type coeffContextRandom struct {
	state uint32
}

func newCoeffContextRandom(seed uint32) *coeffContextRandom {
	return &coeffContextRandom{state: seed}
}

func (r *coeffContextRandom) next() uint32 {
	r.state = r.state*1664525 + 1013904223
	return r.state
}

func (r *coeffContextRandom) u8() uint8 {
	return uint8(r.next() >> 24)
}

func (r *coeffContextRandom) u16() uint16 {
	return uint16(r.next() >> 16)
}

func TestCoeffGeometryTxScaleMatchesQuantizeTransformScale(t *testing.T) {
	valid := 0
	for size := TransformSize(0); size < transformSizeCount; size++ {
		dims, ok := size.Dimensions()
		if !ok {
			t.Fatalf("transform size %d has no dimensions", size)
		}
		shape, err := size.TransformSize()
		if err != nil {
			t.Fatalf("transform size %d: %v", size, err)
		}
		want, err := quantize.TransformScale(int(shape.Width), int(shape.Height))
		if err != nil {
			t.Fatalf("TransformScale(%dx%d): %v", shape.Width, shape.Height, err)
		}
		geo, ok := coeffGeo(size)
		if !ok {
			t.Fatalf("transform size %d has no coefficient geometry", size)
		}
		if geo.txCtx != (dims.Min+dims.Max+1)>>1 || geo.txCtx >= CoeffTxSizeContexts {
			t.Fatalf("transform size %d tx context=%d", size, geo.txCtx)
		}
		got := uint8(0)
		if geo.txCtx > 2 {
			got = geo.txCtx - 2
		}
		if got != want {
			t.Errorf("transform size %d (%dx%d): txCtx=%d scale=%d want %d", size, shape.Width, shape.Height, geo.txCtx, got, want)
		}
		valid++
	}
	if valid != 19 {
		t.Fatalf("validated %d transform sizes, want 19", valid)
	}
}

// TestCoeffCDFsEqualMatchesOperator pins Equal to the == it replaces: the
// default sets compare equal to a copy, and adapting any single CDF makes
// them differ.
func TestCoeffCDFsEqualMatchesOperator(t *testing.T) {
	for q := range CoeffQContexts {
		def := &defaultCoeffCDFs[q]
		cp := *def
		if !def.Equal(&cp) || *def != cp {
			t.Fatalf("q context %d: copy does not compare equal", q)
		}
		flat := unsafe.Slice((*entropy.CDF)(unsafe.Pointer(&cp)), coeffCDFCount)
		if uintptr(len(flat)) != coeffCDFCount {
			t.Fatalf("flat length %d want %d", len(flat), coeffCDFCount)
		}
		for i := 0; i < len(flat); i += 29 {
			save := flat[i]
			if err := flat[i].Update(0); err != nil {
				t.Fatalf("q context %d CDF %d: %v", q, i, err)
			}
			if flat[i] == save {
				t.Fatalf("q context %d CDF %d: Update left the CDF unchanged", q, i)
			}
			if def.Equal(&cp) || *def == cp {
				t.Fatalf("q context %d CDF %d: modified copy compares equal", q, i)
			}
			flat[i] = save
		}
		last := len(flat) - 1
		save := flat[last]
		flat[last].Update(1)
		if def.Equal(&cp) {
			t.Fatalf("q context %d: last CDF change not detected", q)
		}
		flat[last] = save
		if !def.Equal(&cp) {
			t.Fatalf("q context %d: restored copy does not compare equal", q)
		}
	}
}

func TestCoeffContextsMatchLibaom(t *testing.T) {
	ctxTests := []struct {
		size TransformSize
		want uint8
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
		want uint8
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

	if got, err := CoeffPlaneTypeForPlane(2); err != nil || got != CoeffPlaneUV {
		t.Fatalf("CoeffPlaneTypeForPlane(2)=%d err=%v want UV", got, err)
	}
}

func TestEOBPositionTokenMatchesLibaom(t *testing.T) {
	tests := []struct {
		eob       int
		wantToken int
		wantExtra int
	}{
		{eob: 0, wantToken: 0, wantExtra: 0},
		{eob: 1, wantToken: 1, wantExtra: 0},
		{eob: 2, wantToken: 2, wantExtra: 0},
		{eob: 3, wantToken: 3, wantExtra: 0},
		{eob: 4, wantToken: 3, wantExtra: 1},
		{eob: 5, wantToken: 4, wantExtra: 0},
		{eob: 8, wantToken: 4, wantExtra: 3},
		{eob: 9, wantToken: 5, wantExtra: 0},
		{eob: 16, wantToken: 5, wantExtra: 7},
		{eob: 17, wantToken: 6, wantExtra: 0},
		{eob: 32, wantToken: 6, wantExtra: 15},
		{eob: 33, wantToken: 7, wantExtra: 0},
		{eob: 64, wantToken: 7, wantExtra: 31},
		{eob: 65, wantToken: 8, wantExtra: 0},
		{eob: 128, wantToken: 8, wantExtra: 63},
		{eob: 129, wantToken: 9, wantExtra: 0},
		{eob: 256, wantToken: 9, wantExtra: 127},
		{eob: 257, wantToken: 10, wantExtra: 0},
		{eob: 512, wantToken: 10, wantExtra: 255},
		{eob: 513, wantToken: 11, wantExtra: 0},
		{eob: 1024, wantToken: 11, wantExtra: 511},
	}
	for _, tt := range tests {
		token, extra, err := EOBPositionToken(tt.eob)
		if err != nil {
			t.Fatal(err)
		}
		if token != tt.wantToken || extra != tt.wantExtra {
			t.Fatalf("EOBPositionToken(%d)=(%d,%d) want (%d,%d)", tt.eob, token, extra, tt.wantToken, tt.wantExtra)
		}
	}

	for eob := 0; eob <= 1024; eob++ {
		token, extra, err := EOBPositionToken(eob)
		if err != nil {
			t.Fatalf("EOBPositionToken(%d): %v", eob, err)
		}
		if got := int(eobGroupStart[token]) + extra; got != eob {
			t.Fatalf("eob round trip %d -> token %d extra %d -> %d", eob, token, extra, got)
		}
		if extra >= 1<<eobOffsetBits[token] {
			t.Fatalf("eob %d extra=%d offsetBits=%d", eob, extra, eobOffsetBits[token])
		}
	}

	if _, _, err := EOBPositionToken(-1); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("negative eob err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, _, err := EOBPositionToken(1025); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("large eob err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestCoeffEOB1TX64x64ContextsMatchLibaomMVFrame1(t *testing.T) {
	txsCtx, err := CoeffTransformSizeContext(TransformSize64x64)
	if err != nil {
		t.Fatal(err)
	}
	if txsCtx != 4 {
		t.Fatalf("CoeffTransformSizeContext(TX_64X64)=%d want 4", txsCtx)
	}

	eobMultiSize, err := EOBMultiSize(TransformSize64x64)
	if err != nil {
		t.Fatal(err)
	}
	if eobMultiSize != 6 {
		t.Fatalf("EOBMultiSize(TX_64X64)=%d want 6 (eob_flag_cdf1024)", eobMultiSize)
	}

	scanSize, err := transform.ScanSize(transform.Size{Width: 64, Height: 64})
	if err != nil {
		t.Fatal(err)
	}
	if scanSize.Width != 32 || scanSize.Height != 32 {
		t.Fatalf("ScanSize(TX_64X64)=%+v want 32x32 (av1_get_adjusted_tx_size)", scanSize)
	}

	if got, err := transform.LowerLevelsCtxEOB(scanSize, 0); err != nil || got != 0 {
		t.Fatalf("LowerLevelsCtxEOB(32x32, c=0)=%d err=%v want 0", got, err)
	}

	if got, err := CoeffBRContextEOB(TransformSize64x64, transform.Class2D, 0); err != nil || got != 0 {
		t.Fatalf("CoeffBRContextEOB(TX_64X64, 2D, pos=0)=%d err=%v want 0", got, err)
	}

	// qindex=81 lives in q-context 2 (60 < q <= 120). A regression in the
	// q-context selection would silently swap the CDF cohort the mv vector
	// reads coefficients with for every frame past the keyframe.
	if got := CoeffQContext(81); got != 2 {
		t.Fatalf("CoeffQContext(81)=%d want 2 (mv vector inter-frame qctx)", got)
	}

	// The eob=1 decode path stores level into levels[get_padded_idx(0, bhl)],
	// which for TX_64X64 (height=32, bhl=5) is just index 0 (since
	// (0 >> 5) << TX_PAD_HOR_LOG2 = 0). A regression where pos=0 lands on
	// a padded index would silently corrupt subsequent CoeffLowerLevelsContext
	// reads when a non-eob-only block coexists with the DC entry.
	scratchLen, err := CoeffLevelsScratchLen(TransformSize64x64)
	if err != nil {
		t.Fatal(err)
	}
	levels := make([]uint8, scratchLen)
	if err := setCoeffLevel(levels, TransformSize64x64, 0, 3); err != nil {
		t.Fatal(err)
	}
	if levels[0] != 3 {
		t.Fatalf("setCoeffLevel(TX_64X64, pos=0, 3) stored level at padded idx %d want 0",
			indexOfNonZero(levels))
	}
}
