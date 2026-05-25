package tile

import (
	"errors"
	"math/bits"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestTransformTypeOrdinalsMatchLibaomAndDav1d(t *testing.T) {
	typeCases := []struct {
		typ  transform.Type
		want uint8
	}{
		{transform.TypeDCTDCT, 0},
		{transform.TypeADSTDCT, 1},
		{transform.TypeDCTADST, 2},
		{transform.TypeADSTADST, 3},
		{transform.TypeFlipADSTDCT, 4},
		{transform.TypeDCTFlipADST, 5},
		{transform.TypeFlipADSTFlipADST, 6},
		{transform.TypeADSTFlipADST, 7},
		{transform.TypeFlipADSTADST, 8},
		{transform.TypeIDTX, 9},
		{transform.TypeVDCT, 10},
		{transform.TypeHDCT, 11},
		{transform.TypeVADST, 12},
		{transform.TypeHADST, 13},
		{transform.TypeVFlipADST, 14},
		{transform.TypeHFlipADST, 15},
	}
	for _, tt := range typeCases {
		if uint8(tt.typ) != tt.want {
			t.Fatalf("transform type ordinal=%d want %d", tt.typ, tt.want)
		}
	}

	setCases := []struct {
		set  ExtTXSetType
		want uint8
	}{
		{ExtTXSetDCTOnly, 0},
		{ExtTXSetDCTIDTX, 1},
		{ExtTXSetDTT4IDTX, 2},
		{ExtTXSetDTT4IDTX1DDCT, 3},
		{ExtTXSetDTT9IDTX1DDCT, 4},
		{ExtTXSetAll16, 5},
	}
	for _, tt := range setCases {
		if uint8(tt.set) != tt.want {
			t.Fatalf("ext tx set ordinal=%d want %d", tt.set, tt.want)
		}
	}
}

func TestTransformSizeSquareMapsMatchLibaom(t *testing.T) {
	tests := []struct {
		name string
		size TransformSize
		want TransformSize
	}{
		{name: "4x4", size: TransformSize4x4, want: TransformSize4x4},
		{name: "64x64", size: TransformSize64x64, want: TransformSize64x64},
		{name: "4x8", size: TransformSize4x8, want: TransformSize4x4},
		{name: "8x4", size: TransformSize8x4, want: TransformSize4x4},
		{name: "8x16", size: TransformSize8x16, want: TransformSize8x8},
		{name: "16x8", size: TransformSize16x8, want: TransformSize8x8},
		{name: "16x32", size: TransformSize16x32, want: TransformSize16x16},
		{name: "32x16", size: TransformSize32x16, want: TransformSize16x16},
		{name: "32x64", size: TransformSize32x64, want: TransformSize32x32},
		{name: "64x32", size: TransformSize64x32, want: TransformSize32x32},
		{name: "4x16", size: TransformSize4x16, want: TransformSize4x4},
		{name: "16x4", size: TransformSize16x4, want: TransformSize4x4},
		{name: "8x32", size: TransformSize8x32, want: TransformSize8x8},
		{name: "32x8", size: TransformSize32x8, want: TransformSize8x8},
		{name: "16x64", size: TransformSize16x64, want: TransformSize16x16},
		{name: "64x16", size: TransformSize64x16, want: TransformSize16x16},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := TransformSizeSquare(tt.size)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("TransformSizeSquare(%d)=%d want %d", tt.size, got, tt.want)
			}
		})
	}
}

func TestTransformSizeSquareUpMapsMatchLibaom(t *testing.T) {
	tests := []struct {
		name string
		size TransformSize
		want TransformSize
	}{
		{name: "4x4", size: TransformSize4x4, want: TransformSize4x4},
		{name: "64x64", size: TransformSize64x64, want: TransformSize64x64},
		{name: "4x8", size: TransformSize4x8, want: TransformSize8x8},
		{name: "8x4", size: TransformSize8x4, want: TransformSize8x8},
		{name: "8x16", size: TransformSize8x16, want: TransformSize16x16},
		{name: "16x8", size: TransformSize16x8, want: TransformSize16x16},
		{name: "16x32", size: TransformSize16x32, want: TransformSize32x32},
		{name: "32x16", size: TransformSize32x16, want: TransformSize32x32},
		{name: "32x64", size: TransformSize32x64, want: TransformSize64x64},
		{name: "64x32", size: TransformSize64x32, want: TransformSize64x64},
		{name: "4x16", size: TransformSize4x16, want: TransformSize16x16},
		{name: "16x4", size: TransformSize16x4, want: TransformSize16x16},
		{name: "8x32", size: TransformSize8x32, want: TransformSize32x32},
		{name: "32x8", size: TransformSize32x8, want: TransformSize32x32},
		{name: "16x64", size: TransformSize16x64, want: TransformSize64x64},
		{name: "64x16", size: TransformSize64x16, want: TransformSize64x64},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := TransformSizeSquareUp(tt.size)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("TransformSizeSquareUp(%d)=%d want %d", tt.size, got, tt.want)
			}
		})
	}
}

func TestExtTXSetTypeMatchesLibaom(t *testing.T) {
	tests := []struct {
		name    string
		size    TransformSize
		inter   bool
		reduced bool
		want    ExtTXSetType
	}{
		{name: "64x64 inter", size: TransformSize64x64, inter: true, want: ExtTXSetDCTOnly},
		{name: "64x64 intra", size: TransformSize64x64, want: ExtTXSetDCTOnly},
		{name: "32x32 inter", size: TransformSize32x32, inter: true, want: ExtTXSetDCTIDTX},
		{name: "32x32 intra", size: TransformSize32x32, want: ExtTXSetDCTOnly},
		{name: "16x16 inter reduced", size: TransformSize16x16, inter: true, reduced: true, want: ExtTXSetDCTIDTX},
		{name: "16x16 intra reduced", size: TransformSize16x16, reduced: true, want: ExtTXSetDTT4IDTX},
		{name: "8x8 inter full", size: TransformSize8x8, inter: true, want: ExtTXSetAll16},
		{name: "16x16 inter full", size: TransformSize16x16, inter: true, want: ExtTXSetDTT9IDTX1DDCT},
		{name: "8x8 intra full", size: TransformSize8x8, want: ExtTXSetDTT4IDTX1DDCT},
		{name: "16x16 intra full", size: TransformSize16x16, want: ExtTXSetDTT4IDTX},
		{name: "64x16 inter full", size: TransformSize64x16, inter: true, want: ExtTXSetDCTOnly},
		{name: "8x32 inter full", size: TransformSize8x32, inter: true, want: ExtTXSetDCTIDTX},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ExtTXSetTypeFor(tt.size, tt.inter, tt.reduced)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("ExtTXSetTypeFor(%d,%v,%v)=%d want %d", tt.size, tt.inter, tt.reduced, got, tt.want)
			}
		})
	}
}

func TestExtTXSetIndexMatchesLibaom(t *testing.T) {
	tests := []struct {
		name    string
		size    TransformSize
		inter   bool
		reduced bool
		want    int
	}{
		{name: "dct only intra", size: TransformSize32x32, want: 0},
		{name: "dct idtx inter", size: TransformSize32x32, inter: true, want: 3},
		{name: "reduced intra dtt4 idtx", size: TransformSize16x16, reduced: true, want: 2},
		{name: "full intra dtt4 idtx one-d dct", size: TransformSize8x8, want: 1},
		{name: "full inter dtt9 idtx one-d dct", size: TransformSize16x16, inter: true, want: 2},
		{name: "full inter all16", size: TransformSize8x8, inter: true, want: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ExtTXSetIndex(tt.size, tt.inter, tt.reduced)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("ExtTXSetIndex(%d,%v,%v)=%d want %d", tt.size, tt.inter, tt.reduced, got, tt.want)
			}
		})
	}
}

func TestExtTXTypeCountAndFlagsMatchLibaom(t *testing.T) {
	tests := []struct {
		name string
		set  ExtTXSetType
		flag uint16
	}{
		{name: "DCT only", set: ExtTXSetDCTOnly, flag: 0x0001},
		{name: "DCT IDTX", set: ExtTXSetDCTIDTX, flag: 0x0201},
		{name: "DTT4 IDTX", set: ExtTXSetDTT4IDTX, flag: 0x020f},
		{name: "DTT4 IDTX one-d DCT", set: ExtTXSetDTT4IDTX1DDCT, flag: 0x0e0f},
		{name: "DTT9 IDTX one-d DCT", set: ExtTXSetDTT9IDTX1DDCT, flag: 0x0fff},
		{name: "ALL16", set: ExtTXSetAll16, flag: 0xffff},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := extTXTypeUsedFlag[tt.set]; got != tt.flag {
				t.Fatalf("flag=0x%04x want 0x%04x", got, tt.flag)
			}
			count, err := ExtTXTypeCount(tt.set)
			if err != nil {
				t.Fatal(err)
			}
			if count != bits.OnesCount16(tt.flag) {
				t.Fatalf("count=%d popcount=%d", count, bits.OnesCount16(tt.flag))
			}
			for raw := uint8(0); raw < uint8(transform.TypeCount); raw++ {
				typ := transform.Type(raw)
				got, err := ExtTXTypeAllowed(tt.set, typ)
				if err != nil {
					t.Fatal(err)
				}
				want := tt.flag&(1<<typ) != 0
				if got != want {
					t.Fatalf("allowed set=%d type=%d got %v want %v", tt.set, typ, got, want)
				}
			}
		})
	}
}

func TestExtTXTypeInvMatchesLibaom(t *testing.T) {
	sequences := map[ExtTXSetType][]transform.Type{
		ExtTXSetDCTOnly:       {transform.TypeDCTDCT},
		ExtTXSetDCTIDTX:       {transform.TypeIDTX, transform.TypeDCTDCT},
		ExtTXSetDTT4IDTX:      {transform.TypeIDTX, transform.TypeDCTDCT, transform.TypeADSTADST, transform.TypeADSTDCT, transform.TypeDCTADST},
		ExtTXSetDTT4IDTX1DDCT: {transform.TypeIDTX, transform.TypeDCTDCT, transform.TypeVDCT, transform.TypeHDCT, transform.TypeADSTADST, transform.TypeADSTDCT, transform.TypeDCTADST},
		ExtTXSetDTT9IDTX1DDCT: {transform.TypeIDTX, transform.TypeVDCT, transform.TypeHDCT, transform.TypeDCTDCT, transform.TypeADSTDCT, transform.TypeDCTADST, transform.TypeFlipADSTDCT, transform.TypeDCTFlipADST, transform.TypeADSTADST, transform.TypeFlipADSTFlipADST, transform.TypeADSTFlipADST, transform.TypeFlipADSTADST},
		ExtTXSetAll16:         {transform.TypeIDTX, transform.TypeVDCT, transform.TypeHDCT, transform.TypeVADST, transform.TypeHADST, transform.TypeVFlipADST, transform.TypeHFlipADST, transform.TypeDCTDCT, transform.TypeADSTDCT, transform.TypeDCTADST, transform.TypeFlipADSTDCT, transform.TypeDCTFlipADST, transform.TypeADSTADST, transform.TypeFlipADSTFlipADST, transform.TypeADSTFlipADST, transform.TypeFlipADSTADST},
	}
	for set, want := range sequences {
		count, err := ExtTXTypeCount(set)
		if err != nil {
			t.Fatal(err)
		}
		if count != len(want) {
			t.Fatalf("set %d count=%d want %d", set, count, len(want))
		}
		for symbol, wantType := range want {
			got, err := ExtTXTypeFromSymbol(set, symbol)
			if err != nil {
				t.Fatal(err)
			}
			if got != wantType {
				t.Fatalf("ExtTXTypeFromSymbol(%d,%d)=%d want %d", set, symbol, got, wantType)
			}
		}
	}

	tests := []struct {
		name   string
		set    ExtTXSetType
		symbol int
		want   transform.Type
	}{
		{name: "DCT only", set: ExtTXSetDCTOnly, symbol: 0, want: transform.TypeDCTDCT},
		{name: "DCT IDTX symbol0", set: ExtTXSetDCTIDTX, symbol: 0, want: transform.TypeIDTX},
		{name: "DCT IDTX symbol1", set: ExtTXSetDCTIDTX, symbol: 1, want: transform.TypeDCTDCT},
		{name: "DTT4 IDTX symbol0", set: ExtTXSetDTT4IDTX, symbol: 0, want: transform.TypeIDTX},
		{name: "DTT4 IDTX symbol2", set: ExtTXSetDTT4IDTX, symbol: 2, want: transform.TypeADSTADST},
		{name: "DTT4 IDTX symbol3", set: ExtTXSetDTT4IDTX, symbol: 3, want: transform.TypeADSTDCT},
		{name: "DTT4 IDTX symbol4", set: ExtTXSetDTT4IDTX, symbol: 4, want: transform.TypeDCTADST},
		{name: "DTT4 one-d symbol2", set: ExtTXSetDTT4IDTX1DDCT, symbol: 2, want: transform.TypeVDCT},
		{name: "DTT4 one-d symbol3", set: ExtTXSetDTT4IDTX1DDCT, symbol: 3, want: transform.TypeHDCT},
		{name: "DTT9 one-d symbol11", set: ExtTXSetDTT9IDTX1DDCT, symbol: 11, want: transform.TypeFlipADSTADST},
		{name: "ALL16 symbol7", set: ExtTXSetAll16, symbol: 7, want: transform.TypeDCTDCT},
		{name: "ALL16 symbol15", set: ExtTXSetAll16, symbol: 15, want: transform.TypeFlipADSTADST},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ExtTXTypeFromSymbol(tt.set, tt.symbol)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("ExtTXTypeFromSymbol(%d,%d)=%d want %d", tt.set, tt.symbol, got, tt.want)
			}
		})
	}
}

func TestExtTXSymbolMapMatchesLibaom(t *testing.T) {
	for set := ExtTXSetType(0); set < ExtTXSetTypes; set++ {
		count, err := ExtTXTypeCount(set)
		if err != nil {
			t.Fatal(err)
		}
		seen := uint16(0)
		for symbol := 0; symbol < count; symbol++ {
			typ, err := ExtTXTypeFromSymbol(set, symbol)
			if err != nil {
				t.Fatal(err)
			}
			seen |= 1 << typ
			got, err := ExtTXSymbolForType(set, typ)
			if err != nil {
				t.Fatal(err)
			}
			if got != symbol {
				t.Fatalf("ExtTXSymbolForType(%d,%d)=%d want %d", set, typ, got, symbol)
			}
		}
		if seen != extTXTypeUsedFlag[set] {
			t.Fatalf("set %d inverse symbols flag=0x%04x want 0x%04x", set, seen, extTXTypeUsedFlag[set])
		}
	}
}

func TestTransformTypeCDFsInitDefaultMatchesLibaom(t *testing.T) {
	var cdfs TransformTypeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}

	if got := cdfs.Inter[1][0].Symbols(); got != 16 {
		t.Fatalf("inter set 1 symbols=%d want 16", got)
	}
	assertEntropyCDFValues(t, cdfs.Inter[1][0].Values(), []uint16{
		28310, 27208, 25073, 23059, 19438, 17979, 15231, 12502,
		11264, 9920, 8834, 7294, 5041, 3853, 2137, 0, 0,
	})

	if got := cdfs.Inter[2][2].Symbols(); got != 12 {
		t.Fatalf("inter set 2 symbols=%d want 12", got)
	}
	assertEntropyCDFValues(t, cdfs.Inter[2][2].Values(), []uint16{
		31998, 30347, 27543, 19861, 16949, 13841, 11207, 8679,
		6173, 4242, 2239, 0, 0,
	})

	if got := cdfs.Inter[3][3].Symbols(); got != 2 {
		t.Fatalf("inter set 3 symbols=%d want 2", got)
	}
	assertEntropyCDFValues(t, cdfs.Inter[3][3].Values(), []uint16{32020, 0, 0})

	if got := cdfs.Intra[1][0][IntraModeDC].Symbols(); got != 7 {
		t.Fatalf("intra set 1 symbols=%d want 7", got)
	}
	assertEntropyCDFValues(t, cdfs.Intra[1][0][IntraModeDC].Values(), []uint16{
		31233, 24733, 23307, 20017, 9301, 4943, 0, 0,
	})

	if got := cdfs.Intra[1][1][IntraModeVertical].Symbols(); got != 7 {
		t.Fatalf("intra set 1 size1 symbols=%d want 7", got)
	}
	assertEntropyCDFValues(t, cdfs.Intra[1][1][IntraModeVertical].Values(), []uint16{
		32442, 23972, 18136, 17689, 13496, 5282, 0, 0,
	})

	if got := cdfs.Intra[2][2][IntraModeDC].Symbols(); got != 5 {
		t.Fatalf("intra set 2 symbols=%d want 5", got)
	}
	assertEntropyCDFValues(t, cdfs.Intra[2][2][IntraModeDC].Values(), []uint16{31641, 19954, 9996, 5285, 0, 0})

	if got := cdfs.Intra[2][0][IntraModePaeth].Symbols(); got != 5 {
		t.Fatalf("intra set 2 uniform symbols=%d want 5", got)
	}
	assertEntropyCDFValues(t, cdfs.Intra[2][0][IntraModePaeth].Values(), []uint16{26214, 19661, 13107, 6554, 0, 0})
}

func TestReadIntraTransformTypeMatchesLibaomBranches(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	got, err := state.ReadIntraTransformType(nil, IntraTransformTypeRequest{Size: TransformSize16x16, Mode: IntraModeDC, SkipTransform: true})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeDCTDCT {
		t.Fatalf("skip tx type=%d want %d", got, transform.TypeDCTDCT)
	}

	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	got, err = state.ReadIntraTransformType(nil, IntraTransformTypeRequest{Size: TransformSize16x16, Mode: IntraModeDC, Lossless: true})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeDCTDCT {
		t.Fatalf("lossless tx type=%d want %d", got, transform.TypeDCTDCT)
	}

	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	got, err = state.ReadIntraTransformType(nil, IntraTransformTypeRequest{Size: TransformSize16x16, Mode: IntraModeDC, QIndexKnown: true})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeDCTDCT {
		t.Fatalf("qindex-zero tx type=%d want %d", got, transform.TypeDCTDCT)
	}

	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	got, err = state.ReadIntraTransformType(nil, IntraTransformTypeRequest{Size: TransformSize32x32, Mode: IntraModeDC})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeDCTDCT {
		t.Fatalf("dct-only tx type=%d want %d", got, transform.TypeDCTDCT)
	}
}

func TestReadIntraTransformTypeUsesLibaomCDF(t *testing.T) {
	var cdfs TransformTypeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}

	got, err := state.ReadIntraTransformType(&cdfs, IntraTransformTypeRequest{Size: TransformSize8x8, Mode: IntraModeDC})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeIDTX {
		t.Fatalf("zero payload intra tx type=%d want %d", got, transform.TypeIDTX)
	}
	if got := cdfs.Intra[1][1][IntraModeDC].Values()[7]; got != 1 {
		t.Fatalf("intra ext tx cdf count=%d want 1", got)
	}

	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	got, err = state.ReadIntraTransformType(&cdfs, IntraTransformTypeRequest{Size: TransformSize16x16, Mode: IntraModeD135})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeIDTX {
		t.Fatalf("zero payload intra set2 tx type=%d want %d", got, transform.TypeIDTX)
	}
	if got := cdfs.Intra[2][2][IntraModeD135].Values()[5]; got != 1 {
		t.Fatalf("intra set2 cdf count=%d want 1", got)
	}
}

func TestReadInterTransformTypeMatchesLibaomBranches(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	got, err := state.ReadInterTransformType(nil, InterTransformTypeRequest{Size: TransformSize16x16, SkipTransform: true})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeDCTDCT {
		t.Fatalf("skip tx type=%d want %d", got, transform.TypeDCTDCT)
	}

	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	got, err = state.ReadInterTransformType(nil, InterTransformTypeRequest{Size: TransformSize16x16, Lossless: true})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeDCTDCT {
		t.Fatalf("lossless tx type=%d want %d", got, transform.TypeDCTDCT)
	}

	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	got, err = state.ReadInterTransformType(nil, InterTransformTypeRequest{Size: TransformSize64x64})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeDCTDCT {
		t.Fatalf("dct-only tx type=%d want %d", got, transform.TypeDCTDCT)
	}
}

func TestReadInterTransformTypeUsesLibaomCDF(t *testing.T) {
	var cdfs TransformTypeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}

	got, err := state.ReadInterTransformType(&cdfs, InterTransformTypeRequest{Size: TransformSize32x32})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeIDTX {
		t.Fatalf("zero payload tx type=%d want %d", got, transform.TypeIDTX)
	}
	if got := cdfs.Inter[3][3].Values()[2]; got != 1 {
		t.Fatalf("inter ext tx cdf count=%d want 1", got)
	}
}

func TestInterCoeffTransformSelectorChromaReusesLumaMap(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var selector InterCoeffTransformSelector
	selector.ResetForColor(&state, nil, false, false, false, parser.ColorConfig{SubsamplingX: true, SubsamplingY: true})
	if err := selector.RecordCoeffTransform(CoeffTransformRequest{
		Plane: 0,
		Block: TransformBlock{X4: 4, Y4: 6, Size: TransformSize8x8},
	}, transform.TypeADSTDCT); err != nil {
		t.Fatal(err)
	}

	before := state.Reader.BitsRead()
	got, err := selector.SelectCoeffTransform(CoeffTransformRequest{
		Plane: 1,
		Block: TransformBlock{X4: 2, Y4: 3, Size: TransformSize4x4},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeADSTDCT {
		t.Fatalf("chroma tx type=%d want %d", got, transform.TypeADSTDCT)
	}
	if after := state.Reader.BitsRead(); after != before {
		t.Fatalf("chroma tx type read bits=%d want %d", after, before)
	}
	if _, err := selector.SelectCoeffTransform(CoeffTransformRequest{
		Plane: 2,
		Block: TransformBlock{X4: 7, Y4: 7, Size: TransformSize4x4},
	}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("missing tx type map err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestInterCoeffTransformSelectorChromaReusesOddSubsampledLumaMap(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var selector InterCoeffTransformSelector
	selector.ResetForColor(&state, nil, false, false, false, parser.ColorConfig{SubsamplingX: true, SubsamplingY: true})
	if err := selector.RecordCoeffTransform(CoeffTransformRequest{
		Plane: 0,
		Block: TransformBlock{X4: 8, Y4: 13, Size: TransformSize8x4},
	}, transform.TypeADSTDCT); err != nil {
		t.Fatal(err)
	}

	got, err := selector.SelectCoeffTransform(CoeffTransformRequest{
		Plane: 2,
		Block: TransformBlock{X4: 4, Y4: 6, Size: TransformSize4x4},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeADSTDCT {
		t.Fatalf("odd chroma tx type=%d want %d", got, transform.TypeADSTDCT)
	}
}

func TestInterCoeffTransformSelectorChromaFallsBackWhenMappedTypeUnsupported(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var selector InterCoeffTransformSelector
	selector.ResetForColor(&state, nil, false, false, false, parser.ColorConfig{SubsamplingX: true, SubsamplingY: true})
	if err := selector.RecordCoeffTransform(CoeffTransformRequest{
		Plane: 0,
		Block: TransformBlock{X4: 0, Y4: 0, Size: TransformSize4x4},
	}, transform.TypeFlipADSTDCT); err != nil {
		t.Fatal(err)
	}
	got, err := selector.SelectCoeffTransform(CoeffTransformRequest{
		Plane: 2,
		Block: TransformBlock{X4: 0, Y4: 0, Size: TransformSize32x32},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeDCTDCT {
		t.Fatalf("unsupported chroma tx type=%d want %d", got, transform.TypeDCTDCT)
	}
}

func TestInterCoeffTransformSelectorLosslessChromaDoesNotNeedLumaMap(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var selector InterCoeffTransformSelector
	selector.ResetForColor(&state, nil, false, false, true, parser.ColorConfig{SubsamplingX: true, SubsamplingY: true})
	before := state.Reader.BitsRead()
	got, err := selector.SelectCoeffTransform(CoeffTransformRequest{
		Plane: 1,
		Block: TransformBlock{X4: 10, Y4: 4, Size: TransformSize4x4},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeDCTDCT {
		t.Fatalf("lossless chroma tx type=%d want %d", got, transform.TypeDCTDCT)
	}
	if after := state.Reader.BitsRead(); after != before {
		t.Fatalf("lossless chroma tx type read bits=%d want %d", after, before)
	}
}

func TestIntraCoeffTransformSelectorReadsLumaAndDerivesChroma(t *testing.T) {
	var cdfs TransformTypeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var selector IntraCoeffTransformSelector
	selector.Reset(&state, &cdfs, false, false, false, 64, IntraModeDC, ChromaIntraModeVertical)
	got, err := selector.SelectCoeffTransform(CoeffTransformRequest{
		Plane: 0,
		Block: TransformBlock{Size: TransformSize8x8},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeIDTX {
		t.Fatalf("intra luma tx type=%d want %d", got, transform.TypeIDTX)
	}

	got, err = selector.SelectCoeffTransform(CoeffTransformRequest{
		Plane: 1,
		Block: TransformBlock{Size: TransformSize8x8},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeADSTDCT {
		t.Fatalf("intra chroma tx type=%d want %d", got, transform.TypeADSTDCT)
	}
}

func TestIntraCoeffTransformSelectorQIndexZeroDoesNotRead(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var selector IntraCoeffTransformSelector
	selector.Reset(&state, nil, false, false, false, 0, IntraModeD45, ChromaIntraModeD135)
	before := state.Reader.BitsRead()
	got, err := selector.SelectCoeffTransform(CoeffTransformRequest{
		Plane: 0,
		Block: TransformBlock{Size: TransformSize8x8},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeDCTDCT {
		t.Fatalf("qindex-zero luma tx type=%d want %d", got, transform.TypeDCTDCT)
	}
	got, err = selector.SelectCoeffTransform(CoeffTransformRequest{
		Plane: 1,
		Block: TransformBlock{Size: TransformSize8x8},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeDCTDCT {
		t.Fatalf("qindex-zero chroma tx type=%d want %d", got, transform.TypeDCTDCT)
	}
	if after := state.Reader.BitsRead(); after != before {
		t.Fatalf("qindex-zero tx type read bits=%d want %d", after, before)
	}
}

func TestIntraChromaTransformTypeMatchesDav1dUVMap(t *testing.T) {
	tests := []struct {
		mode ChromaIntraMode
		size TransformSize
		want transform.Type
	}{
		{ChromaIntraModeDC, TransformSize8x8, transform.TypeDCTDCT},
		{ChromaIntraModeVertical, TransformSize8x8, transform.TypeADSTDCT},
		{ChromaIntraModeHorizontal, TransformSize8x8, transform.TypeDCTADST},
		{ChromaIntraModeD135, TransformSize8x8, transform.TypeADSTADST},
		{ChromaIntraModeCFL, TransformSize8x8, transform.TypeDCTDCT},
		{ChromaIntraModeVertical, TransformSize32x32, transform.TypeDCTDCT},
	}
	for _, tt := range tests {
		got, err := IntraChromaTransformType(tt.mode, tt.size, false)
		if err != nil {
			t.Fatal(err)
		}
		if got != tt.want {
			t.Fatalf("mode=%d size=%d tx type=%d want %d", tt.mode, tt.size, got, tt.want)
		}
	}
	if _, err := IntraChromaTransformType(chromaIntraModeCount, TransformSize8x8, false); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad chroma tx type err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestTransformTypeRejectsInvalidInputs(t *testing.T) {
	if _, err := TransformSizeSquare(transformSizeCount); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad square err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := TransformSizeSquareUp(transformSizeCount); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad square-up err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := ExtTXSetTypeFor(transformSizeCount, true, false); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad set type err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := ExtTXTypeCount(ExtTXSetTypes); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad count err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := ExtTXTypeFromSymbol(ExtTXSetAll16, 16); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad symbol err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := ExtTXTypeAllowed(ExtTXSetTypes, transform.TypeDCTDCT); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad allowed set err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := ExtTXSymbolForType(ExtTXSetDCTOnly, transform.TypeIDTX); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("disallowed symbol err=%v want %v", err, ErrInvalidDecodeState)
	}

	var cdfs TransformTypeCDFs
	if _, err := cdfs.IntraCDF(0, TransformSize4x4, IntraModeDC, 1); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("bad intra cdf err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if _, err := cdfs.InterCDF(0, TransformSize4x4, 1); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("bad inter cdf err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := state.ReadInterTransformType(nil, InterTransformTypeRequest{Size: TransformSize32x32}); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("nil cdfs err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if _, err := state.ReadIntraTransformType(nil, IntraTransformTypeRequest{Size: TransformSize8x8, Mode: IntraModeDC}); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("nil intra cdfs err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	var nilState *DecodeState
	if _, err := nilState.ReadInterTransformType(&cdfs, InterTransformTypeRequest{Size: TransformSize32x32}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil state err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := nilState.ReadIntraTransformType(&cdfs, IntraTransformTypeRequest{Size: TransformSize8x8, Mode: IntraModeDC}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil intra state err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestTransformTypeTablesDoNotAllocate(t *testing.T) {
	allocs := testing.AllocsPerRun(1000, func() {
		set, err := ExtTXSetTypeFor(TransformSize16x16, true, false)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := ExtTXSetIndex(TransformSize16x16, true, false); err != nil {
			t.Fatal(err)
		}
		if _, err := ExtTXTypeFromSymbol(set, 3); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs=%f want 0", allocs)
	}
}
