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
	for rawSize := TransformSize(0); rawSize < transformSizeCount; rawSize++ {
		txSize, err := rawSize.TransformSize()
		if err != nil {
			t.Fatal(err)
		}
		scanSize, err := transform.ScanSize(txSize)
		if err != nil {
			t.Fatal(err)
		}
		maxEOB := scanSize.Width * scanSize.Height
		coeffs := make([]int16, maxEOB)
		for i := 0; i < maxEOB; i++ {
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

		stride := scanSize.Height + txPadHorizontal
		for col := 0; col < scanSize.Width; col++ {
			for row := 0; row < scanSize.Height; row++ {
				idx := col*scanSize.Height + row
				if got, want := levels[col*stride+row], coeffAbsClamp127(coeffs[idx]); got != want {
					t.Fatalf("size=%d level[%d,%d]=%d want %d", rawSize, row, col, got, want)
				}
			}
			for row := scanSize.Height; row < scanSize.Height+txPadHorizontal; row++ {
				if got := levels[col*stride+row]; got != 0 {
					t.Fatalf("size=%d horizontal pad[%d,%d]=%d want 0", rawSize, row, col, got)
				}
			}
		}
		for col := scanSize.Width; col < scanSize.Width+txPadHorizontal; col++ {
			for row := 0; row < stride; row++ {
				if got := levels[col*stride+row]; got != 0 {
					t.Fatalf("size=%d bottom pad[%d,%d]=%d want 0", rawSize, row, col, got)
				}
			}
		}
	}
}

func TestCoeffNZMapContextsMatchesLibaomEncodeTxb(t *testing.T) {
	rnd := newCoeffContextRandom(0x1532a5)
	for isInter := 0; isInter < 2; isInter++ {
		_ = isInter
		for rawType := uint8(0); rawType < uint8(transform.TypeCount); rawType++ {
			typ := transform.Type(rawType)
			class, err := typ.Class()
			if err != nil {
				t.Fatal(err)
			}
			for size := TransformSize(0); size < transformSizeCount; size++ {
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
				maxEOB := scanSize.Width * scanSize.Height
				scan := make([]int16, maxEOB)
				inverse := make([]int16, maxEOB)
				if err := transform.FillDefaultScan(scan, inverse, txSize, class); err != nil {
					t.Fatal(err)
				}
				levels := make([]uint8, mustCoeffLevelsScratchLen(t, size))
				for repeat := 0; repeat < 3; repeat++ {
					for _, eob := range coeffNZMapEOBCases(maxEOB) {
						for i := range levels {
							levels[i] = 0
						}
						contexts := make([]int8, maxEOB)
						want := make([]int8, maxEOB)
						for c := 0; c < eob; c++ {
							pos := int(scan[c])
							mustSetCoeffLevel(t, levels, size, pos, int(rnd.u8()&0x7f))
							contexts[pos] = int8(rnd.u16() >> 9)
							want[pos] = contexts[pos]
						}
						copy(want, contexts)
						for c := 0; c < eob; c++ {
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
						for i := 0; i < maxEOB; i++ {
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
		seen := false
		for _, existing := range out {
			if existing == eob {
				seen = true
				break
			}
		}
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
