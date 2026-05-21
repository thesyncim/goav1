package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
)

func TestCompoundBlendCDFsInitDefaultMatchesLibaom(t *testing.T) {
	var cdfs CompoundBlendCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		cdf  *entropy.CDF
		want []uint16
	}{
		{name: "compound index ctx0", cdf: &cdfs.CompoundIndex[0], want: []uint16{14524, 0, 0}},
		{name: "compound group ctx5", cdf: &cdfs.CompoundGroup[5], want: []uint16{10094, 0, 0}},
		{name: "interintra group0", cdf: &cdfs.InterIntra[0], want: []uint16{16384, 0, 0}},
		{name: "interintra mode group1", cdf: &cdfs.InterIntraModeCDF[1], want: []uint16{30893, 21686, 5436, 0, 0}},
		{name: "wedge interintra 8x8", cdf: &cdfs.WedgeInterIntra[BlockSize8x8], want: []uint16{12732, 0, 0}},
		{name: "compound type 16x16", cdf: &cdfs.CompoundType[BlockSize16x16], want: []uint16{22998, 0, 0}},
		{name: "wedge idx 32x8", cdf: &cdfs.WedgeIndex[BlockSize32x8], want: []uint16{31633, 31446, 31275, 30133, 30072, 30031, 29998, 11752, 9833, 7711, 5517, 3595, 2679, 1808, 835, 0, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertEntropyCDFValues(t, tt.cdf.Values(), tt.want)
		})
	}
}

func TestInterIntraAndWedgeGates(t *testing.T) {
	if !InterIntraAllowedBlockSize(BlockSize8x8) || !InterIntraAllowedBlockSize(BlockSize32x32) {
		t.Fatal("expected 8x8 and 32x32 inter-intra allowed")
	}
	for _, size := range []BlockSize{BlockSize32x8, BlockSize8x32, BlockSize64x64, BlockSize4x4} {
		if InterIntraAllowedBlockSize(size) {
			t.Fatalf("InterIntraAllowedBlockSize(%d)=true want false", size)
		}
	}
	base := InterIntraRequest{
		Size:                     BlockSize16x16,
		Mode:                     InterModeNearestMV,
		EnableInterIntraCompound: true,
		SkipMode:                 false,
		Compound:                 false,
	}
	if !InterIntraAllowed(base) {
		t.Fatal("base inter-intra request unexpectedly disallowed")
	}
	base.SkipMode = true
	if InterIntraAllowed(base) {
		t.Fatal("skip-mode inter-intra request allowed")
	}
	base.SkipMode = false
	base.Compound = true
	if InterIntraAllowed(base) {
		t.Fatal("compound inter-intra request allowed")
	}

	wedgeTests := []struct {
		size BlockSize
		want bool
	}{
		{size: BlockSize32x32, want: true},
		{size: BlockSize32x8, want: true},
		{size: BlockSize16x4, want: false},
		{size: BlockSize64x64, want: false},
	}
	for _, tt := range wedgeTests {
		if got := WedgeUsed(tt.size); got != tt.want {
			t.Fatalf("WedgeUsed(%d)=%v want %v", tt.size, got, tt.want)
		}
	}
	if !AnyMaskedCompoundUsed(BlockSize8x8) || AnyMaskedCompoundUsed(BlockSize8x4) {
		t.Fatal("masked compound block-size gate mismatch")
	}
}

func TestCompoundBlendContextsMatchLibaom(t *testing.T) {
	var ctx BlockModeContext
	req := CompoundBlendRequest{
		Size:             BlockSize16x16,
		MotionMode:       MotionModeTranslation,
		EnableOrderHint:  true,
		OrderHintBits:    4,
		CurrentOrderHint: 8,
		RefOrderHint:     [2]uint32{4, 12},
		HaveTop:          true,
		HaveLeft:         true,
	}

	ctx.AboveCompound[0] = 1
	ctx.AboveCompGroup[0] = 1
	ctx.LeftCompound[0] = 1
	ctx.LeftCompGroup[0] = 0
	if got, err := ctx.CompoundGroupIndexContext(req); err != nil || got != 1 {
		t.Fatalf("compound group ctx=%d err=%v want 1", got, err)
	}

	ctx = BlockModeContext{}
	ctx.AboveRef[0][0] = ReferenceFrameAltref
	ctx.LeftRef[0][0] = ReferenceFrameAltref
	if got, err := ctx.CompoundGroupIndexContext(req); err != nil || got != 5 {
		t.Fatalf("altref group ctx=%d err=%v want 5", got, err)
	}

	ctx = BlockModeContext{}
	ctx.AboveCompound[0] = 1
	ctx.AboveCompIndex[0] = 0
	ctx.LeftRef[0][0] = ReferenceFrameAltref
	if got, err := ctx.CompoundIndexContext(req); err != nil || got != 4 {
		t.Fatalf("equal-distance compound index ctx=%d err=%v want 4", got, err)
	}
	req.RefOrderHint = [2]uint32{7, 12}
	if got, err := ctx.CompoundIndexContext(req); err != nil || got != 1 {
		t.Fatalf("unequal-distance compound index ctx=%d err=%v want 1", got, err)
	}
}

func TestReadInterIntra(t *testing.T) {
	var cdfs CompoundBlendCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	req := InterIntraRequest{
		Size:                     BlockSize16x16,
		Mode:                     InterModeNearestMV,
		EnableInterIntraCompound: true,
	}

	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	result, err := state.ReadInterIntra(&cdfs, req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Enabled {
		t.Fatal("inter-intra enabled on zero payload; want false")
	}
	if got := cdfs.InterIntra[yModeSizeContext[req.Size]].Values()[2]; got != 1 {
		t.Fatalf("interintra cdf count=%d want 1", got)
	}

	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if err := state.Reset([]byte{0xff, 0xff, 0xff}, Job{Offset: 0, Size: 3}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	result, err = state.ReadInterIntra(&cdfs, req)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Enabled || !result.Mode.Valid() {
		t.Fatalf("inter-intra result=%+v want enabled valid mode", result)
	}
	if result.UseWedge && result.WedgeIndex >= MaxWedgeTypes {
		t.Fatalf("wedge index=%d outside max", result.WedgeIndex)
	}
}

func TestReadCompoundBlend(t *testing.T) {
	var cdfs CompoundBlendCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var ctx BlockModeContext
	req := CompoundBlendRequest{
		Size:                  BlockSize16x16,
		Compound:              true,
		MotionMode:            MotionModeTranslation,
		EnableMaskedCompound:  false,
		EnableDistWtdCompound: false,
		EnableOrderHint:       true,
		OrderHintBits:         4,
		CurrentOrderHint:      8,
		RefOrderHint:          [2]uint32{4, 12},
		HaveTop:               true,
		HaveLeft:              true,
	}

	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	result, err := state.ReadCompoundBlend(&cdfs, &ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Type != CompoundTypeAverage || result.CompoundIndex != 1 || result.CompoundGroupIndex != 0 {
		t.Fatalf("default compound blend=%+v want average idx=1 group=0", result)
	}

	req.EnableDistWtdCompound = true
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	result, err = state.ReadCompoundBlend(&cdfs, &ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Type != CompoundTypeDistWtd || result.CompoundIndex != 0 {
		t.Fatalf("dist-wtd result=%+v want dist-wtd idx=0", result)
	}

	req.EnableMaskedCompound = true
	req.EnableDistWtdCompound = false
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if err := state.Reset([]byte{0xff, 0xff, 0xff}, Job{Offset: 0, Size: 3}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	result, err = state.ReadCompoundBlend(&cdfs, &ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if result.CompoundGroupIndex > 1 || !result.Type.Valid() {
		t.Fatalf("masked compound result=%+v outside valid range", result)
	}
}

func TestMarkCompoundBlend(t *testing.T) {
	var ctx BlockModeContext
	refs := InterReferencesResult{
		Ref:      [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameBWD},
		Compound: true,
	}
	if err := ctx.MarkInter(BlockSize16x16, 2, 3, refs); err != nil {
		t.Fatal(err)
	}
	if ctx.AboveCompIndex[2] != 1 || ctx.LeftCompIndex[3] != 1 || ctx.AboveCompGroup[2] != 0 {
		t.Fatalf("MarkInter default comp state aboveIdx=%d leftIdx=%d group=%d want idx=1 group=0",
			ctx.AboveCompIndex[2], ctx.LeftCompIndex[3], ctx.AboveCompGroup[2])
	}
	blend := CompoundBlendResult{Type: CompoundTypeWedge, CompoundGroupIndex: 1, CompoundIndex: 0}
	if err := ctx.MarkCompoundBlend(BlockSize16x16, 2, 3, blend); err != nil {
		t.Fatal(err)
	}
	for x := 2; x < 6; x++ {
		if ctx.AboveCompGroup[x] != 1 || ctx.AboveCompIndex[x] != 0 {
			t.Fatalf("above slot %d group/index=%d/%d want 1/0", x, ctx.AboveCompGroup[x], ctx.AboveCompIndex[x])
		}
	}
	for y := 3; y < 7; y++ {
		if ctx.LeftCompGroup[y] != 1 || ctx.LeftCompIndex[y] != 0 {
			t.Fatalf("left slot %d group/index=%d/%d want 1/0", y, ctx.LeftCompGroup[y], ctx.LeftCompIndex[y])
		}
	}
}

func TestCompoundBlendRejectsInvalidInputs(t *testing.T) {
	var cdfs CompoundBlendCDFs
	if _, err := cdfs.CompoundIndexCDF(0); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("uninitialized compound index err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if _, err := cdfs.WedgeIndexCDF(BlockSize64x64); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("disallowed wedge err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	var nilState *DecodeState
	if _, err := nilState.ReadInterIntra(&cdfs, InterIntraRequest{Size: BlockSize8x8, Mode: InterModeNearestMV}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil inter-intra err=%v want %v", err, ErrInvalidDecodeState)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := state.ReadCompoundBlend(&cdfs, nil, CompoundBlendRequest{Size: BlockSize16x16, MotionMode: MotionModeTranslation}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil compound ctx err=%v want %v", err, ErrInvalidDecodeState)
	}
	var ctx BlockModeContext
	if _, err := state.ReadCompoundBlend(&cdfs, &ctx, CompoundBlendRequest{Size: BlockSize8x4, Compound: true, MotionMode: MotionModeTranslation}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad compound size err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := ctx.CompoundIndexContext(CompoundBlendRequest{Size: BlockSize16x16, MotionMode: MotionModeTranslation, EnableOrderHint: true}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad order hint ctx err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestCompoundBlendAllocs(t *testing.T) {
	var cdfs CompoundBlendCDFs
	var state DecodeState
	payload := []byte{0x00}
	req := InterIntraRequest{
		Size:                     BlockSize16x16,
		Mode:                     InterModeNearestMV,
		EnableInterIntraCompound: true,
	}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := cdfs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		if _, err := state.ReadInterIntra(&cdfs, req); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("inter-intra decode allocated: %f", allocs)
	}
}

func FuzzReadInterIntra(f *testing.F) {
	f.Add([]byte{0x00}, uint8(BlockSize16x16), uint8(InterModeNearestMV), true, false, false)
	f.Add([]byte{0xff, 0xff}, uint8(BlockSize8x8), uint8(InterModeNewMV), true, false, false)
	f.Add([]byte{0xa5, 0x5a}, uint8(BlockSize32x16), uint8(InterModeGlobalMV), true, false, false)

	f.Fuzz(func(t *testing.T, payload []byte, rawSize uint8, rawMode uint8, enabled bool, skip bool, compound bool) {
		if len(payload) == 0 || len(payload) > 64 {
			return
		}
		size := BlockSize(rawSize % uint8(blockSizeCount))
		mode := InterMode(rawMode % uint8(interModeCount))
		var cdfs CompoundBlendCDFs
		if err := cdfs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		result, err := state.ReadInterIntra(&cdfs, InterIntraRequest{
			Size:                     size,
			Mode:                     mode,
			EnableInterIntraCompound: enabled,
			SkipMode:                 skip,
			Compound:                 compound,
		})
		if err != nil {
			t.Fatalf("ReadInterIntra err=%v", err)
		}
		if result.Enabled {
			if !InterIntraAllowedBlockSize(size) || !result.Mode.Valid() {
				t.Fatalf("invalid enabled inter-intra size=%d result=%+v", size, result)
			}
			if result.UseWedge && result.WedgeIndex >= MaxWedgeTypes {
				t.Fatalf("invalid wedge index=%d", result.WedgeIndex)
			}
		}
	})
}

func FuzzReadCompoundBlend(f *testing.F) {
	f.Add([]byte{0x00}, uint8(BlockSize16x16), true, false, false, true, uint8(4), uint32(8), uint32(4), uint32(12), uint8(0), uint8(0))
	f.Add([]byte{0xff, 0xff, 0xff}, uint8(BlockSize8x8), true, true, true, true, uint8(5), uint32(16), uint32(9), uint32(23), uint8(1), uint8(1))
	f.Add([]byte{0xa5, 0x5a}, uint8(BlockSize32x8), true, false, true, false, uint8(0), uint32(0), uint32(0), uint32(0), uint8(0), uint8(1))

	f.Fuzz(func(t *testing.T, payload []byte, rawSize uint8, compound bool, masked bool, dist bool, orderHint bool, bits uint8, cur uint32, ref0 uint32, ref1 uint32, top uint8, left uint8) {
		if len(payload) == 0 || len(payload) > 64 {
			return
		}
		size := BlockSize(rawSize % uint8(blockSizeCount))
		if bits > 8 {
			bits = bits % 9
		}
		if orderHint && bits > 0 {
			limit := uint32(1) << bits
			cur %= limit
			ref0 %= limit
			ref1 %= limit
		}

		var cdfs CompoundBlendCDFs
		if err := cdfs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		var ctx BlockModeContext
		ctx.AboveRef[0][0] = ReferenceFrame(top % uint8(referenceFrameCount))
		ctx.LeftRef[0][0] = ReferenceFrame(left % uint8(referenceFrameCount))
		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		result, err := state.ReadCompoundBlend(&cdfs, &ctx, CompoundBlendRequest{
			Size:                  size,
			Compound:              compound,
			MotionMode:            MotionModeTranslation,
			EnableMaskedCompound:  masked,
			EnableDistWtdCompound: dist,
			EnableOrderHint:       orderHint,
			OrderHintBits:         bits,
			CurrentOrderHint:      cur,
			RefOrderHint:          [2]uint32{ref0, ref1},
			HaveTop:               true,
			HaveLeft:              true,
		})
		if err != nil {
			if compound && !compoundReferenceAllowed(size) {
				return
			}
			if orderHint && bits == 0 && dist {
				return
			}
			t.Fatalf("ReadCompoundBlend err=%v", err)
		}
		if !result.Type.Valid() || result.CompoundIndex > 1 || result.CompoundGroupIndex > 1 {
			t.Fatalf("invalid compound blend result=%+v", result)
		}
		if result.Type == CompoundTypeWedge && result.WedgeIndex >= MaxWedgeTypes {
			t.Fatalf("invalid wedge index=%d", result.WedgeIndex)
		}
	})
}
