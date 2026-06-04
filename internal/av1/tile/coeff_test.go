package tile

import (
	"errors"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
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
		roundTrip, err := RecEOBPosition(token, extra)
		if err != nil {
			t.Fatal(err)
		}
		if roundTrip != tt.eob {
			t.Fatalf("RecEOBPosition(%d,%d)=%d want %d", token, extra, roundTrip, tt.eob)
		}
	}

	for eob := 0; eob <= 1024; eob++ {
		token, extra, err := EOBPositionToken(eob)
		if err != nil {
			t.Fatalf("EOBPositionToken(%d): %v", eob, err)
		}
		got, err := RecEOBPosition(token, extra)
		if err != nil {
			t.Fatalf("RecEOBPosition(%d,%d): %v", token, extra, err)
		}
		if got != eob {
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

// TestCoeffEOB1TX64x64ContextsMatchLibaomMVFrame1 pins the entropy-decode
// context invariants for the canonical inter-frame eob=1 single-DC TX_64X64
// block. libaom's mv test vector at frame 1 mi=(0,0) plane=0 produces a
// 64x64 luma TXB with tx_class=2D, eob=1, level=3 (sign=neg), dq_dc=74 →
// dq=-55 after the 64x64 av1_get_tx_scale shift. The decode reads:
//   - TXBSkip CDF at txs_ctx=4 (CoeffTransformSizeContext for TX_64X64).
//   - EOB CDF1024 at plane=Y eob_multi_ctx=0 (2D class) → eob=1.
//   - LowerLevelsCtxEOB(scanSize=32x32, c=0) → 0 (DC of last-coeff branch).
//   - CoeffBaseEOB CDF at txs_ctx=4, plane=Y, ctx=0 → level=3 (base=3
//     forces the BR loop).
//   - CoeffBRContextEOB(TX_64X64, Class2D, pos=0) → 0 (DC of BR branch).
//   - CoeffBR CDF at AOMMIN(txs_ctx, TX_32X32)=3, plane=Y, ctx=0 (read at
//     least once, but for level=3 yields k=0 which terminates the loop and
//     leaves level at 3 unchanged).
//   - DCSign CDF at plane=Y, dc_sign_ctx=DCSignContext → sign=neg.
//
// Pinning these context values guards against regressions in the
// rectangular-bias rework, the CoeffBR txs-clamp, or the EOB-branch
// context functions that would silently mis-route the entropy reads for
// a single-coefficient 64x64 block.
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

	// CoeffBR CDF is clamped to TX_32X32 (txs_ctx=3) for tx sizes >= 32x32;
	// verify the CDF lookup does the AOMMIN clamp documented in libaom's
	// read_coeffs_txb (coeff_br_cdf[AOMMIN(txs_ctx, TX_32X32)]).
	var cdfs CoeffCDFs
	if err := cdfs.InitDefault(81); err != nil {
		t.Fatal(err)
	}
	brCDF, err := cdfs.CoeffBRCDF(TransformSize64x64, CoeffPlaneY, 0)
	if err != nil {
		t.Fatal(err)
	}
	brCDF32, err := cdfs.CoeffBRCDF(TransformSize32x32, CoeffPlaneY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !cdfValuesEqual(brCDF.Values(), brCDF32.Values()) {
		t.Fatalf("CoeffBRCDF(TX_64X64) diverges from TX_32X32 clamp: got=%v want=%v",
			brCDF.Values(), brCDF32.Values())
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
	if got, err := coeffLevel(levels, TransformSize64x64, 0); err != nil || got != 3 {
		t.Fatalf("coeffLevel(TX_64X64, 0)=%d err=%v want 3", got, err)
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

// TestCoeffTraceStubIsNoOp guards the goav1_coeff_trace build tag's no-op
// stub. The trace helpers are wired into ReadCoefficientsTXB hot paths to
// capture (c, pos, base, golomb, level, sign) tuples for cross-validation
// against libaom's read_coeffs_txb. The stub variant must never observe
// or mutate state so the default build retains zero overhead.
func TestCoeffTraceStubIsNoOp(t *testing.T) {
	coeffTraceBlock(0, 0, 0, int(TransformSize4x4), 0, 0)
	coeffTraceCoeff(0, 0, 0, 0, 0, 0)
	coeffTraceCulLevel(0)

	// Should also leave decode behaviour intact: re-run the single-DC
	// canonical block and confirm coeff[0]==1 and cul_level==17 still hold
	// after the trace points fire.
	result, coeffs := readCoefficientsTXBForTest(t, []byte{0x00}, TransformSize4x4, transform.Class2D, CoeffPlaneY, 0)
	if result.AllZero || result.EOB != 1 || coeffs[0] != 1 || result.CulLevel != 17 {
		t.Fatalf("trace stub disturbed decode: result=%+v coeff[0]=%d", result, coeffs[0])
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
		if err := state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
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

func BenchmarkReadCoefficientsTXB8x8Class2D(b *testing.B) {
	size := TransformSize8x8
	class := transform.Class2D
	txSize, err := size.TransformSize()
	if err != nil {
		b.Fatal(err)
	}
	scan, scratch := coeffScanAndScratch(b, size, txSize, class)
	coeffs := make([]int16, len(scan))
	payload := coeffBenchmarkTXBPayload(b, size, class, scan, scratch)
	req := TXBDecodeRequest{
		Size:            size,
		Plane:           CoeffPlaneY,
		Class:           class,
		TXBSkipContext:  0,
		DCSignContext:   0,
		EOBMultiContext: 0,
	}
	var cdfs CoeffCDFs
	if err := cdfs.InitDefault(0); err != nil {
		b.Fatal(err)
	}
	var state DecodeState
	sum := 0
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if err := state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
			b.Fatal(err)
		}
		result, err := state.ReadCoefficientsTXB(&cdfs, req, coeffs, scan, scratch)
		if err != nil {
			b.Fatal(err)
		}
		sum += int(result.EOB) + int(result.CulLevel)
	}
	coeffBenchmarkSink = sum
}

func coeffBenchmarkTXBPayload(b *testing.B, size TransformSize, class transform.Class, scan []int16, scratch []uint8) []byte {
	b.Helper()
	coeffs := make([]int16, len(scan))
	req := TXBDecodeRequest{
		Size:            size,
		Plane:           CoeffPlaneY,
		Class:           class,
		TXBSkipContext:  0,
		DCSignContext:   0,
		EOBMultiContext: 0,
	}
	for seed := 0; seed < 4096; seed++ {
		payload := []byte{byte(seed), byte(seed*73 + 19), byte(seed ^ 0xa5), byte(seed*29 + 7), 0x5a}
		clear(coeffs)
		clear(scratch)
		var cdfs CoeffCDFs
		if err := cdfs.InitDefault(0); err != nil {
			b.Fatal(err)
		}
		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
			b.Fatal(err)
		}
		result, err := state.ReadCoefficientsTXB(&cdfs, req, coeffs, scan, scratch)
		if err == nil && !result.AllZero && result.EOB > 8 {
			return payload
		}
	}
	b.Fatal("could not find nontrivial coefficient benchmark payload")
	return nil
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
		if err := state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
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

func TestReadCoefficientsTXBSkipClearPolicy(t *testing.T) {
	txSize, err := TransformSize4x4.TransformSize()
	if err != nil {
		t.Fatal(err)
	}
	scan, scratch := coeffScanAndScratch(t, TransformSize4x4, txSize, transform.Class2D)
	coeffs := make([]int16, len(scan))
	var cdfs CoeffCDFs
	if err := cdfs.InitDefault(0); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	req := TXBDecodeRequest{
		Size:          TransformSize4x4,
		Plane:         CoeffPlaneY,
		Class:         transform.Class2D,
		TXBSkipKnown:  true,
		TXBSkip:       true,
		DCSignContext: 0,
	}

	for i := range coeffs {
		coeffs[i] = 7
	}
	result, err := state.ReadCoefficientsTXB(&cdfs, req, coeffs, scan, scratch)
	if err != nil {
		t.Fatal(err)
	}
	assertTXBDecodeInvariants(t, result, coeffs, scan)

	for i := range coeffs {
		coeffs[i] = 7
	}
	req.SkipAllZeroCoeffClear = true
	result, err = state.ReadCoefficientsTXB(&cdfs, req, coeffs, scan, scratch)
	if err != nil {
		t.Fatal(err)
	}
	if !result.AllZero {
		t.Fatalf("result=%+v want all zero", result)
	}
	for i, coeff := range coeffs {
		if coeff != 7 {
			t.Fatalf("coeff[%d]=%d want preserved sentinel", i, coeff)
		}
	}
}

func TestReadCoefficientsTXBTrackedLevelDirtyClearsScratch(t *testing.T) {
	txSize, err := TransformSize8x8.TransformSize()
	if err != nil {
		t.Fatal(err)
	}
	scan, levelWindow := coeffScanAndScratch(t, TransformSize8x8, txSize, transform.Class2D)
	var levelArena [maxCoeffScratchLen]uint8
	levels := levelArena[:len(levelWindow)]
	coeffs := make([]int16, len(scan))
	stalePadded := len(levelArena) - 1
	if stalePadded < 0 || stalePadded > int(^uint16(0)>>1) {
		t.Fatalf("stale padded index=%d out of test range", stalePadded)
	}

	var levelDirty [maxCoeffScanLen]int16
	var levelDirtyLen uint16
	req := TXBDecodeRequest{
		Size:              TransformSize8x8,
		Plane:             CoeffPlaneY,
		Class:             transform.Class2D,
		TXBSkipContext:    0,
		DCSignContext:     0,
		EOBMultiContext:   0,
		levelDirtyPos:     &levelDirty,
		levelDirtyLen:     &levelDirtyLen,
		levelDirtyScratch: &levelArena,
	}

	for seed := 0; seed < 256; seed++ {
		for i := range coeffs {
			coeffs[i] = 0
		}
		levelArena[stalePadded] = 77
		levelDirty[0] = int16(stalePadded)
		levelDirtyLen = 1

		var cdfs CoeffCDFs
		if err := cdfs.InitDefault(0); err != nil {
			t.Fatal(err)
		}
		var state DecodeState
		payload := []byte{byte(seed), byte(seed*73 + 19), byte(seed ^ 0xa5), 0x5a}
		if err := state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		result, err := state.ReadCoefficientsTXB(&cdfs, req, coeffs, scan, levels)
		if err != nil || result.AllZero || result.EOB <= 1 {
			continue
		}
		if levelArena[stalePadded] != 0 {
			t.Fatalf("stale padded level %d survived dirty clear: %d", stalePadded, levelArena[stalePadded])
		}
		if levelDirtyLen == 0 {
			t.Fatal("levelDirtyLen=0 after nonzero eob decode")
		}
		for i := 0; i < int(levelDirtyLen); i++ {
			padded := int(levelDirty[i])
			if padded < 0 || padded >= len(levels) {
				t.Fatalf("levelDirty[%d]=%d out of range len=%d", i, padded, len(levels))
			}
			if levels[padded] == 0 {
				t.Fatalf("levelDirty[%d]=%d points at zero level", i, padded)
			}
		}
		return
	}
	t.Fatal("test payload search did not produce an eob>1 TXB")
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
		if err := state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		if _, err := state.ReadTXBSkip(&cdfs, TXBSkipRequest{
			Size:    size,
			Context: rawCoeffCtx % TXBSkipContexts,
		}); err != nil {
			t.Fatalf("ReadTXBSkip err=%v", err)
		}
		if _, err := state.ReadEOB(&cdfs, EOBRequest{
			Size:            size,
			Plane:           plane,
			EOBMultiContext: rawEOBCtx % maxEOBFlagContexts,
		}); err != nil {
			t.Fatalf("ReadEOB err=%v", err)
		}
		if _, err := state.ReadCoeffBaseEOB(&cdfs, CoeffTokenRequest{
			Size:    size,
			Plane:   plane,
			Context: rawCoeffCtx % EOBBaseContexts,
		}); err != nil {
			t.Fatalf("ReadCoeffBaseEOB err=%v", err)
		}
		if _, err := state.ReadCoeffBase(&cdfs, CoeffTokenRequest{
			Size:    size,
			Plane:   plane,
			Context: rawCoeffCtx % CoeffBaseContexts,
		}); err != nil {
			t.Fatalf("ReadCoeffBase err=%v", err)
		}
		if _, err := state.ReadCoeffBR(&cdfs, CoeffTokenRequest{
			Size:    size,
			Plane:   plane,
			Context: rawCoeffCtx % CoeffBRContexts,
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
		if err := state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		result, err := state.ReadCoefficientsTXB(&cdfs, TXBDecodeRequest{
			Size:            size,
			Plane:           plane,
			Class:           class,
			TXBSkipContext:  rawSkipCtx % TXBSkipContexts,
			DCSignContext:   rawDCSignCtx % 3,
			EOBMultiContext: rawEOBCtx % maxEOBFlagContexts,
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
	if err := state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
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

func assertTXBDecodeInvariants(t *testing.T, result TXBDecodeResult, coeffs []int16, scan []int16) {
	t.Helper()
	if result.CulLevel > CoeffContextMask+(2<<CoeffContextBits) {
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
	if result.EOB == 0 || int(result.EOB) > len(scan) {
		t.Fatalf("eob=%d outside scan len %d", result.EOB, len(scan))
	}
	if int(result.MaxScanLine) >= len(coeffs) {
		t.Fatalf("max_scan_line=%d outside coeff len %d", result.MaxScanLine, len(coeffs))
	}
	last := int(scan[int(result.EOB)-1])
	if coeffs[last] == 0 {
		t.Fatalf("last non-zero scan coeff[%d]=0", last)
	}
	for c := int(result.EOB); c < len(scan); c++ {
		pos := int(scan[c])
		if coeffs[pos] != 0 {
			t.Fatalf("coeff after eob scan=%d pos=%d value=%d want 0", c, pos, coeffs[pos])
		}
	}
}
