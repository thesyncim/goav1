package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestIntraModeCDFsInitDefaultMatchesDav1dAndLibaom(t *testing.T) {
	var cdfs IntraModeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		cdf  *entropy.CDF
		want []uint16
	}{
		{name: "intra ctx0", cdf: &cdfs.Intra[0], want: []uint16{31962, 0, 0}},
		{name: "intra ctx3", cdf: &cdfs.Intra[3], want: []uint16{6230, 0, 0}},
		{name: "intrabc", cdf: &cdfs.Intrabc, want: []uint16{2237, 0, 0}},
		{name: "ymode ctx0", cdf: &cdfs.YMode[0], want: []uint16{9967, 9279, 8475, 8012, 7167, 6645, 6162, 5350, 4823, 3540, 3083, 2419, 0, 0}},
		{name: "ymode ctx3", cdf: &cdfs.YMode[3], want: []uint16{12613, 11467, 9930, 9590, 9507, 9235, 9065, 7964, 7416, 6193, 5752, 4719, 0, 0}},
		{name: "keyframe ymode 0,0", cdf: &cdfs.KeyframeYMode[0][0], want: []uint16{17180, 15741, 13430, 12550, 12086, 11658, 10943, 9524, 8579, 4603, 3675, 2302, 0, 0}},
		{name: "keyframe ymode 4,4", cdf: &cdfs.KeyframeYMode[4][4], want: []uint16{25150, 24480, 22909, 22259, 17382, 14111, 9865, 3992, 3588, 1413, 966, 175, 0, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertEntropyCDFValues(t, tt.cdf.Values(), tt.want)
		})
	}
}

func TestIntraContextsMatchDav1dAndLibaom(t *testing.T) {
	var ctx BlockModeContext
	if got, err := ctx.IntraContext(0, 0, false, false); err != nil || got != 0 {
		t.Fatalf("empty intra ctx=%d err=%v", got, err)
	}

	if err := ctx.MarkIntra(BlockSize8x8, 2, 3, true, IntraModeVertical); err != nil {
		t.Fatal(err)
	}
	if got, err := ctx.IntraContext(2, 3, true, true); err != nil || got != 3 {
		t.Fatalf("both intra ctx=%d err=%v", got, err)
	}
	if got, err := ctx.IntraContext(2, 3, true, false); err != nil || got != 2 {
		t.Fatalf("top intra ctx=%d err=%v", got, err)
	}
	if got, err := ctx.IntraContext(2, 3, false, true); err != nil || got != 2 {
		t.Fatalf("left intra ctx=%d err=%v", got, err)
	}
	above, left, err := ctx.KeyframeYModeContext(2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if above != 1 || left != 1 {
		t.Fatalf("keyframe ctx above=%d left=%d want 1,1", above, left)
	}

	if err := ctx.MarkIntra(BlockSize4x4, 4, 4, true, IntraModeHorizontal); err != nil {
		t.Fatal(err)
	}
	above, left, err = ctx.KeyframeYModeContext(4, 4)
	if err != nil {
		t.Fatal(err)
	}
	if above != 2 || left != 2 {
		t.Fatalf("keyframe horizontal ctx above=%d left=%d want 2,2", above, left)
	}
}

func TestMarkIntraEntryLeavesInterRefsForReferenceReader(t *testing.T) {
	var ctx BlockModeContext
	if err := ctx.MarkInter(BlockSize8x8, 0, 0, InterReferencesResult{
		Ref: [2]ReferenceFrame{ReferenceFrameGolden, ReferenceFrameNone},
	}); err != nil {
		t.Fatal(err)
	}
	if err := ctx.MarkIntraEntry(BlockSize8x8, 0, 0, false, IntraModeDC); err != nil {
		t.Fatal(err)
	}
	if ctx.AboveIntra[0] != 0 || ctx.LeftIntra[0] != 0 {
		t.Fatalf("intra flags above=%d left=%d want 0,0", ctx.AboveIntra[0], ctx.LeftIntra[0])
	}
	if ctx.AboveRef[0][0] != ReferenceFrameGolden || ctx.LeftRef[0][0] != ReferenceFrameGolden {
		t.Fatalf("refs changed above=%d left=%d", ctx.AboveRef[0][0], ctx.LeftRef[0][0])
	}

	if err := ctx.MarkIntra(BlockSize8x8, 0, 0, true, IntraModeVertical); err != nil {
		t.Fatal(err)
	}
	if ctx.AboveRef[0][0] != ReferenceFrameNone || ctx.LeftRef[0][0] != ReferenceFrameNone ||
		ctx.AboveMode[0] != IntraModeVertical || ctx.LeftMode[0] != IntraModeVertical {
		t.Fatalf("intra mark refs/modes aboveRef=%d leftRef=%d aboveMode=%d leftMode=%d",
			ctx.AboveRef[0][0], ctx.LeftRef[0][0], ctx.AboveMode[0], ctx.LeftMode[0])
	}
}

func TestReadIntraFlag(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var cdfs IntraModeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var ctx BlockModeContext

	intra, err := state.ReadIntraFlag(&cdfs, &ctx, IntraFlagRequest{
		FrameType: parser.FrameTypeInter,
		Segment:   parser.SegmentData{RefFrame: -1},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !intra {
		t.Fatal("inter-frame intra flag=false want true for zero payload")
	}
	if got := cdfs.Intra[0].Values()[2]; got != 1 {
		t.Fatalf("intra cdf count=%d want 1", got)
	}

	before := cdfs.Intra[0]
	intra, err = state.ReadIntraFlag(&cdfs, &ctx, IntraFlagRequest{
		FrameType: parser.FrameTypeInter,
		SkipMode:  true,
		Segment:   parser.SegmentData{RefFrame: -1},
	})
	if err != nil {
		t.Fatal(err)
	}
	if intra || cdfs.Intra[0] != before {
		t.Fatalf("skip-mode intra=%v cdf changed=%v", intra, cdfs.Intra[0] != before)
	}
}

func TestReadIntraFlagSegmentAndIntrabcConditions(t *testing.T) {
	var cdfs IntraModeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var ctx BlockModeContext
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}

	intra, err := state.ReadIntraFlag(&cdfs, &ctx, IntraFlagRequest{
		FrameType:           parser.FrameTypeInter,
		SegmentationEnabled: true,
		Segment:             parser.SegmentData{RefFrame: 0},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !intra {
		t.Fatal("segment INTRA_FRAME should force intra")
	}
	intra, err = state.ReadIntraFlag(&cdfs, &ctx, IntraFlagRequest{
		FrameType:           parser.FrameTypeInter,
		SegmentationEnabled: true,
		Segment:             parser.SegmentData{RefFrame: 2},
	})
	if err != nil {
		t.Fatal(err)
	}
	if intra {
		t.Fatal("segment inter ref should force inter")
	}
	intra, err = state.ReadIntraFlag(&cdfs, &ctx, IntraFlagRequest{
		FrameType:    parser.FrameTypeKey,
		AllowIntrabc: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !intra {
		t.Fatal("zero intrabc flag should decode intra")
	}
	if got := cdfs.Intrabc.Values()[2]; got != 1 {
		t.Fatalf("intrabc cdf count=%d want 1", got)
	}
}

func TestReadLumaIntraMode(t *testing.T) {
	var cdfs IntraModeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var ctx BlockModeContext
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}

	mode, err := state.ReadLumaIntraMode(&cdfs, &ctx, LumaIntraModeRequest{
		FrameType: parser.FrameTypeInter,
		Size:      BlockSize16x16,
	})
	if err != nil {
		t.Fatal(err)
	}
	if mode != IntraModeDC {
		t.Fatalf("inter ymode=%d want DC", mode)
	}
	if got := cdfs.YMode[2].Values()[int(intraModeCount)]; got != 1 {
		t.Fatalf("ymode cdf count=%d want 1", got)
	}

	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	mode, err = state.ReadLumaIntraMode(&cdfs, &ctx, LumaIntraModeRequest{
		FrameType: parser.FrameTypeKey,
		Size:      BlockSize4x4,
	})
	if err != nil {
		t.Fatal(err)
	}
	if mode != IntraModeDC {
		t.Fatalf("keyframe ymode=%d want DC", mode)
	}
	if got := cdfs.KeyframeYMode[0][0].Values()[int(intraModeCount)]; got != 1 {
		t.Fatalf("keyframe ymode cdf count=%d want 1", got)
	}
}

func TestIntraRejectsInvalidInputs(t *testing.T) {
	var cdfs IntraModeCDFs
	if _, err := cdfs.IntraCDF(0); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("uninitialized intra cdf err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if _, err := cdfs.YModeCDF(LumaIntraModeContexts); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("bad ymode cdf err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if _, err := cdfs.KeyframeYModeCDF(KeyframeIntraModeContexts, 0); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("bad keyframe cdf err=%v want %v", err, entropy.ErrInvalidCDF)
	}

	var ctx BlockModeContext
	if err := ctx.MarkIntra(blockSizeCount, 0, 0, true, IntraModeDC); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad mark err=%v want %v", err, ErrInvalidDecodeState)
	}
	if err := ctx.MarkIntra(BlockSize4x4, 0, 0, true, intraModeCount); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad mode mark err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, _, err := ctx.KeyframeYModeContext(MaxBlockModeSlots, 0); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad keyframe ctx err=%v want %v", err, ErrInvalidDecodeState)
	}

	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var nilState *DecodeState
	if _, err := nilState.ReadIntraFlag(&cdfs, &ctx, IntraFlagRequest{}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil intra flag err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.ReadIntraFlag(&cdfs, nil, IntraFlagRequest{FrameType: parser.FrameTypeInter}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil intra ctx err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.ReadLumaIntraMode(&cdfs, &ctx, LumaIntraModeRequest{Size: blockSizeCount}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad luma mode err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestIntraModeAllocs(t *testing.T) {
	var cdfs IntraModeCDFs
	var ctx BlockModeContext
	var state DecodeState
	payload := []byte{0x00}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := cdfs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		ctx = BlockModeContext{}
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		intra, err := state.ReadIntraFlag(&cdfs, &ctx, IntraFlagRequest{
			FrameType: parser.FrameTypeInter,
			Segment:   parser.SegmentData{RefFrame: -1},
		})
		if err != nil {
			t.Fatal(err)
		}
		if intra {
			mode, err := state.ReadLumaIntraMode(&cdfs, &ctx, LumaIntraModeRequest{
				FrameType: parser.FrameTypeInter,
				Size:      BlockSize16x16,
			})
			if err != nil {
				t.Fatal(err)
			}
			if err := ctx.MarkIntra(BlockSize16x16, 0, 0, true, mode); err != nil {
				t.Fatal(err)
			}
		}
	})
	if allocs != 0 {
		t.Fatalf("intra mode decode allocated: %f", allocs)
	}
}

func FuzzReadIntraEntry(f *testing.F) {
	f.Add([]byte{0x00}, uint8(parser.FrameTypeInter), uint8(BlockSize16x16), uint8(0), uint8(0), false, false, false)
	f.Add([]byte{0xff}, uint8(parser.FrameTypeInter), uint8(BlockSize8x8), uint8(3), uint8(4), true, true, false)
	f.Add([]byte{0xa5, 0x5a}, uint8(parser.FrameTypeKey), uint8(BlockSize4x4), uint8(0), uint8(0), false, false, true)

	f.Fuzz(func(t *testing.T, payload []byte, rawFrameType uint8, rawSize uint8, rawX uint8, rawY uint8, haveTop bool, haveLeft bool, allowIntrabc bool) {
		if len(payload) == 0 || len(payload) > 64 {
			return
		}
		size := BlockSize(rawSize % uint8(blockSizeCount))
		dims, ok := size.Dimensions()
		if !ok {
			t.Fatal("invalid generated block size")
		}
		xLimit := MaxBlockModeSlots - int(dims.W4) + 1
		yLimit := MaxBlockModeSlots - int(dims.H4) + 1
		x4 := int(rawX) % xLimit
		y4 := int(rawY) % yLimit
		if x4 == 0 {
			haveLeft = false
		}
		if y4 == 0 {
			haveTop = false
		}

		frameType := parser.FrameType(rawFrameType & 3)
		var cdfs IntraModeCDFs
		if err := cdfs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		var ctx BlockModeContext
		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		intra, err := state.ReadIntraFlag(&cdfs, &ctx, IntraFlagRequest{
			FrameType:    frameType,
			AllowIntrabc: allowIntrabc,
			Segment:      parser.SegmentData{RefFrame: -1},
			X4:           x4,
			Y4:           y4,
			HaveTop:      haveTop,
			HaveLeft:     haveLeft,
		})
		if err != nil {
			t.Fatalf("ReadIntraFlag err=%v frameType=%d", err, frameType)
		}
		if !intra {
			return
		}
		mode, err := state.ReadLumaIntraMode(&cdfs, &ctx, LumaIntraModeRequest{
			FrameType: frameType,
			Size:      size,
			X4:        x4,
			Y4:        y4,
		})
		if err != nil {
			t.Fatalf("ReadLumaIntraMode err=%v frameType=%d size=%d", err, frameType, size)
		}
		if !mode.Valid() {
			t.Fatalf("mode=%d", mode)
		}
		if err := ctx.MarkIntra(size, x4, y4, true, mode); err != nil {
			t.Fatalf("MarkIntra err=%v", err)
		}
	})
}
