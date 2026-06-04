package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestInterRefCDFsInitDefaultMatchesDav1dAndLibaom(t *testing.T) {
	var cdfs InterRefCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		cdf  *entropy.CDF
		want []uint16
	}{
		{name: "comp inter ctx0", cdf: &cdfs.CompInter[0], want: []uint16{5940, 0, 0}},
		{name: "comp ref type ctx4", cdf: &cdfs.CompRefType[4], want: []uint16{10293, 0, 0}},
		{name: "single ref p1 ctx0", cdf: &cdfs.SingleRef[0][0], want: []uint16{27871, 0, 0}},
		{name: "single ref p6 ctx2", cdf: &cdfs.SingleRef[5][2], want: []uint16{2464, 0, 0}},
		{name: "comp fwd ref p2 ctx1", cdf: &cdfs.CompFwdRef[2][1], want: []uint16{17608, 0, 0}},
		{name: "comp bwd ref p1 ctx2", cdf: &cdfs.CompBwdRef[1][2], want: []uint16{2279, 0, 0}},
		{name: "uni comp ref p ctx2", cdf: &cdfs.UniCompRef[0][2], want: []uint16{994, 0, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertEntropyCDFValues(t, tt.cdf.Values(), tt.want)
		})
	}
}

func TestInterReferenceContextsMatchDav1d(t *testing.T) {
	var ctx BlockModeContext
	req := InterReferenceRequest{Size: BlockSize16x16}
	if got, err := ctx.ReferenceModeContext(req); err != nil || got != 1 {
		t.Fatalf("empty comp ctx=%d err=%v want 1", got, err)
	}
	if got, err := ctx.CompoundReferenceTypeContext(req); err != nil || got != 2 {
		t.Fatalf("empty comp dir ctx=%d err=%v want 2", got, err)
	}
	if got, err := ctx.ReferenceContext(req); err != nil || got != 1 {
		t.Fatalf("empty ref ctx=%d err=%v want 1", got, err)
	}

	if err := ctx.MarkInter(BlockSize8x8, 0, 0, InterReferencesResult{
		Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone},
	}, true); err != nil {
		t.Fatal(err)
	}
	req.HaveTop = true
	req.HaveLeft = true
	if got, err := ctx.ReferenceModeContext(req); err != nil || got != 0 {
		t.Fatalf("single/single comp ctx=%d err=%v want 0", got, err)
	}
	if got, err := ctx.ReferenceContext(req); err != nil || got != 2 {
		t.Fatalf("forward-only ref ctx=%d err=%v want 2", got, err)
	}

	ctx.AboveRef[0][0] = ReferenceFrameLast
	ctx.AboveRef[1][0] = ReferenceFrameGolden
	ctx.AboveCompound[0] = 1
	ctx.LeftRef[0][0] = ReferenceFrameBWD
	ctx.LeftRef[1][0] = ReferenceFrameAltref
	ctx.LeftCompound[0] = 1
	if got, err := ctx.ReferenceModeContext(req); err != nil || got != 4 {
		t.Fatalf("compound/compound comp ctx=%d err=%v want 4", got, err)
	}
	if got, err := ctx.CompoundReferenceTypeContext(req); err != nil || got != 3 {
		t.Fatalf("unidir/unidir comp dir ctx=%d err=%v want 3", got, err)
	}
	if got, err := ctx.ForwardReferenceContext(req); err != nil || got != 1 {
		t.Fatalf("fwd ctx=%d err=%v want 1", got, err)
	}
	if got, err := ctx.BackwardReferenceContext(req); err != nil || got != 1 {
		t.Fatalf("bwd ctx=%d err=%v want 1", got, err)
	}

	if err := ctx.MarkIntra(BlockSize8x8, 0, 0, true, IntraModeDC); err != nil {
		t.Fatal(err)
	}
	if ctx.AboveRef[0][0] != ReferenceFrameNone || ctx.LeftRef[0][0] != ReferenceFrameNone || ctx.AboveCompound[0] != 0 {
		t.Fatalf("intra refs above=%d left=%d comp=%d", ctx.AboveRef[0][0], ctx.LeftRef[0][0], ctx.AboveCompound[0])
	}
	if got, err := ctx.ReferenceModeContext(InterReferenceRequest{Size: BlockSize16x16, HaveTop: true}); err != nil || got != 0 {
		t.Fatalf("intra edge comp ctx=%d err=%v want 0", got, err)
	}
}

func TestReadInterReferencesSingleAndForced(t *testing.T) {
	var cdfs InterRefCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var ctx BlockModeContext
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}

	result, err := state.ReadInterReferences(&cdfs, &ctx, InterReferenceRequest{
		Size:          BlockSize16x16,
		ReferenceMode: parser.ReferenceModeSelect,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Compound || result.Ref != [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone} {
		t.Fatalf("refs=%+v want LAST/NONE", result)
	}
	if got := cdfs.CompInter[1].Values()[2]; got != 1 {
		t.Fatalf("comp-inter cdf count=%d want 1", got)
	}
	if got := cdfs.SingleRef[0][1].Values()[2]; got != 1 {
		t.Fatalf("single p1 cdf count=%d want 1", got)
	}
	if err := ctx.MarkInter(BlockSize16x16, 0, 0, result, true); err != nil {
		t.Fatal(err)
	}
	if ctx.AboveRef[0][0] != ReferenceFrameLast || ctx.LeftRef[1][0] != ReferenceFrameNone {
		t.Fatalf("marked refs above=%d left1=%d", ctx.AboveRef[0][0], ctx.LeftRef[1][0])
	}

	result, err = state.ReadInterReferences(&cdfs, &ctx, InterReferenceRequest{
		Size:         BlockSize16x16,
		SkipMode:     true,
		SkipModeRefs: [2]ReferenceFrame{ReferenceFrameLast2, ReferenceFrameAltref},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Compound || result.Ref != [2]ReferenceFrame{ReferenceFrameLast2, ReferenceFrameAltref} {
		t.Fatalf("skip refs=%+v", result)
	}

	result, err = state.ReadInterReferences(&cdfs, &ctx, InterReferenceRequest{
		Size:                BlockSize16x16,
		SegmentationEnabled: true,
		Segment:             parser.SegmentData{RefFrame: 3},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Compound || result.Ref[0] != ReferenceFrameLast3 || result.Ref[1] != ReferenceFrameNone {
		t.Fatalf("segment refs=%+v", result)
	}
}

func TestReadInterReferencesCompound(t *testing.T) {
	var cdfs InterRefCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var ctx BlockModeContext
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}

	result, err := state.ReadInterReferences(&cdfs, &ctx, InterReferenceRequest{
		Size:          BlockSize16x16,
		ReferenceMode: parser.ReferenceModeCompound,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Compound || !result.Unidir || result.Ref != [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameLast2} {
		t.Fatalf("compound refs=%+v want LAST/LAST2 unidir", result)
	}
	if got := cdfs.CompRefType[2].Values()[2]; got != 1 {
		t.Fatalf("comp ref type count=%d want 1", got)
	}
	if got := cdfs.UniCompRef[0][1].Values()[2]; got != 1 {
		t.Fatalf("uni comp ref count=%d want 1", got)
	}
}

func TestInterReferencesRejectInvalidInputs(t *testing.T) {
	var cdfs InterRefCDFs
	if _, err := cdfs.SingleRefCDF(0, 0); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("uninitialized cdf err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if _, err := cdfs.CompFwdRefCDF(CompoundForwardReferenceBits, 0); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("bad cdf err=%v want %v", err, entropy.ErrInvalidCDF)
	}

	var ctx BlockModeContext
	if _, err := ctx.ReferenceModeContext(InterReferenceRequest{Size: blockSizeCount}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad ctx err=%v want %v", err, ErrInvalidDecodeState)
	}
	if err := ctx.MarkInter(BlockSize4x4, 0, 0, InterReferencesResult{}, true); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad mark err=%v want %v", err, ErrInvalidDecodeState)
	}

	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var nilState *DecodeState
	if _, err := nilState.ReadInterReferences(&cdfs, &ctx, InterReferenceRequest{Size: BlockSize16x16}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil state err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.ReadInterReferences(&cdfs, &ctx, InterReferenceRequest{
		Size:         BlockSize16x16,
		SkipMode:     true,
		SkipModeRefs: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone},
	}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad skip refs err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.ReadInterReferences(&cdfs, &ctx, InterReferenceRequest{
		Size:          BlockSize16x16,
		ReferenceMode: parser.ReferenceMode(99),
	}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad reference mode err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestInterReferencesAllocs(t *testing.T) {
	var cdfs InterRefCDFs
	var ctx BlockModeContext
	var state DecodeState
	payload := []byte{0x00}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := cdfs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		ctx = BlockModeContext{}
		if err := state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		result, err := state.ReadInterReferences(&cdfs, &ctx, InterReferenceRequest{
			Size:          BlockSize16x16,
			ReferenceMode: parser.ReferenceModeSelect,
		})
		if err != nil {
			t.Fatal(err)
		}
		if err := ctx.MarkInter(BlockSize16x16, 0, 0, result, true); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("inter reference decode allocated: %f", allocs)
	}
}

func FuzzReadInterReferences(f *testing.F) {
	f.Add([]byte{0x00}, uint8(BlockSize16x16), uint8(parser.ReferenceModeSelect), uint8(0), uint8(0), false, false)
	f.Add([]byte{0xff}, uint8(BlockSize32x32), uint8(parser.ReferenceModeCompound), uint8(3), uint8(4), true, true)
	f.Add([]byte{0xa5, 0x5a}, uint8(BlockSize8x8), uint8(parser.ReferenceModeSingle), uint8(7), uint8(9), true, false)

	f.Fuzz(func(t *testing.T, payload []byte, rawSize uint8, rawMode uint8, rawX uint8, rawY uint8, haveTop bool, haveLeft bool) {
		if len(payload) == 0 || len(payload) > 64 {
			return
		}
		size := BlockSize(rawSize % uint8(blockSizeCount))
		dims, ok := size.Dimensions()
		if !ok {
			t.Fatal("invalid generated block size")
		}
		x4 := int(rawX) % (MaxBlockModeSlots - int(dims.W4) + 1)
		y4 := int(rawY) % (MaxBlockModeSlots - int(dims.H4) + 1)
		mode := parser.ReferenceMode(rawMode % 3)

		var cdfs InterRefCDFs
		if err := cdfs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		var ctx BlockModeContext
		if haveTop {
			ctx.AboveRef[0][x4] = ReferenceFrameLast
			ctx.AboveRef[1][x4] = ReferenceFrameAltref
			ctx.AboveCompound[x4] = 1
		}
		if haveLeft {
			ctx.LeftRef[0][y4] = ReferenceFrameGolden
			ctx.LeftRef[1][y4] = ReferenceFrameNone
		}
		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		result, err := state.ReadInterReferences(&cdfs, &ctx, InterReferenceRequest{
			Size:          size,
			ReferenceMode: mode,
			X4:            uint8(x4),
			Y4:            uint8(y4),
			HaveTop:       haveTop,
			HaveLeft:      haveLeft,
		})
		if err != nil {
			t.Fatalf("ReadInterReferences err=%v size=%d mode=%d", err, size, mode)
		}
		if err := ctx.MarkInter(size, x4, y4, result, true); err != nil {
			t.Fatalf("MarkInter err=%v result=%+v", err, result)
		}
	})
}
