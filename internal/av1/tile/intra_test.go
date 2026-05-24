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
		{name: "uv no cfl dc", cdf: &cdfs.UVMode[0][IntraModeDC], want: []uint16{10137, 8616, 7390, 7107, 6782, 6248, 5713, 4845, 4524, 2709, 1827, 807, 0, 0}},
		{name: "uv cfl paeth", cdf: &cdfs.UVMode[1][IntraModePaeth], want: []uint16{29624, 27681, 25386, 25264, 25175, 25078, 24967, 24704, 24536, 23520, 22893, 22247, 3720, 0, 0}},
		{name: "angle delta vertical", cdf: &cdfs.AngleDelta[0], want: []uint16{30588, 27736, 25201, 9992, 5779, 2551, 0, 0}},
		{name: "cfl sign", cdf: &cdfs.CFLSign, want: []uint16{31350, 30645, 19428, 14363, 5796, 4425, 474, 0, 0}},
		{name: "cfl alpha ctx0", cdf: &cdfs.CFLAlpha[0], want: []uint16{25131, 12049, 1367, 287, 111, 80, 76, 72, 68, 64, 60, 56, 52, 48, 44, 0, 0}},
		{name: "filter intra 4x16", cdf: &cdfs.FilterIntra[BlockSize4x16], want: []uint16{19998, 0, 0}},
		{name: "filter intra mode", cdf: &cdfs.FilterIntraMode, want: []uint16{23819, 19992, 15557, 3210, 0, 0}},
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

func TestIntraEdgeSmoothNeighborMatchesLibaomLumaType(t *testing.T) {
	var ctx BlockModeContext
	if err := ctx.MarkIntra(BlockSize8x8, 2, 3, true, IntraModeSmoothVertical); err != nil {
		t.Fatal(err)
	}
	if got, err := ctx.IntraEdgeSmoothNeighbor(2, 3, true, false); err != nil || !got {
		t.Fatalf("top smooth=%v err=%v want true", got, err)
	}
	if err := ctx.MarkIntra(BlockSize4x4, 6, 7, true, IntraModeD203); err != nil {
		t.Fatal(err)
	}
	if got, err := ctx.IntraEdgeSmoothNeighbor(6, 7, true, true); err != nil || got {
		t.Fatalf("directional smooth=%v err=%v want false", got, err)
	}
	if err := ctx.MarkInter(BlockSize4x4, 8, 9, InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}); err != nil {
		t.Fatal(err)
	}
	if got, err := ctx.IntraEdgeSmoothNeighbor(8, 9, true, true); err != nil || got {
		t.Fatalf("inter neighbor smooth=%v err=%v want false", got, err)
	}
}

func TestChromaIntraEdgeSmoothNeighborMatchesLibaomUVType(t *testing.T) {
	var ctx BlockModeContext
	if err := ctx.MarkChromaIntra(BlockSize8x8, 2, 3, true, ChromaIntraModeSmoothHorizontal); err != nil {
		t.Fatal(err)
	}
	if got, err := ctx.ChromaIntraEdgeSmoothNeighbor(2, 3, true, false, false, false); err != nil || !got {
		t.Fatalf("top chroma smooth=%v err=%v want true", got, err)
	}
	if err := ctx.MarkChromaIntra(BlockSize4x4, 6, 7, true, ChromaIntraModeD45); err != nil {
		t.Fatal(err)
	}
	if got, err := ctx.ChromaIntraEdgeSmoothNeighbor(6, 7, true, true, false, false); err != nil || got {
		t.Fatalf("directional chroma smooth=%v err=%v want false", got, err)
	}
	if err := ctx.MarkInter(BlockSize4x4, 8, 9, InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}); err != nil {
		t.Fatal(err)
	}
	if got, err := ctx.ChromaIntraEdgeSmoothNeighbor(8, 9, true, true, false, false); err != nil || got {
		t.Fatalf("inter chroma neighbor smooth=%v err=%v want false", got, err)
	}
	if err := ctx.MarkChromaIntra(BlockSize4x4, 5, 4, true, ChromaIntraModeSmooth); err != nil {
		t.Fatal(err)
	}
	if got, err := ctx.ChromaIntraEdgeSmoothNeighbor(4, 5, true, false, true, true); err != nil || !got {
		t.Fatalf("subsampled chroma smooth=%v err=%v want true", got, err)
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

func TestReadIntraAngleDeltaAndChromaMode(t *testing.T) {
	var cdfs IntraModeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00, 0x00}, Job{Offset: 0, Size: 2}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}

	delta, err := state.ReadIntraAngleDelta(&cdfs, IntraAngleDeltaRequest{
		Size: BlockSize16x16,
		Mode: IntraModeD45,
	})
	if err != nil {
		t.Fatal(err)
	}
	if delta != -3 {
		t.Fatalf("angle delta=%d want -3", delta)
	}
	if got := cdfs.AngleDelta[IntraModeD45-IntraModeVertical].Values()[2*AngleDeltaMax+1]; got != 1 {
		t.Fatalf("angle cdf count=%d want 1", got)
	}

	mode, alpha, err := state.ReadChromaIntraMode(&cdfs, ChromaIntraModeRequest{
		Size:       BlockSize16x16,
		LumaMode:   IntraModeDC,
		CFLAllowed: false,
	})
	if err != nil {
		t.Fatal(err)
	}
	if mode != ChromaIntraModeDC || alpha != (CFLAlphaResult{}) {
		t.Fatalf("chroma mode=%d alpha=%+v want dc/no alpha", mode, alpha)
	}
	if got := cdfs.UVMode[0][IntraModeDC].Values()[int(chromaIntraModeCount-1)]; got != 1 {
		t.Fatalf("uv cdf count=%d want 1", got)
	}

	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	alpha, err = state.ReadCFLAlphas(&cdfs)
	if err != nil {
		t.Fatal(err)
	}
	if alpha.JointSign != 0 || alpha.Index != 0 {
		t.Fatalf("cfl alpha=%+v want sign=0 index=0", alpha)
	}
	if got := cdfs.CFLSign.Values()[CFLJointSigns]; got != 1 {
		t.Fatalf("cfl sign count=%d want 1", got)
	}
	if got := cdfs.CFLAlpha[0].Values()[CFLAlphabetSize]; got != 1 {
		t.Fatalf("cfl alpha count=%d want 1", got)
	}
}

func TestReadFilterIntraMode(t *testing.T) {
	var cdfs IntraModeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	mode, valid, err := state.ReadFilterIntraMode(&cdfs, FilterIntraRequest{
		EnableFilterIntra: true,
		Size:              BlockSize4x16,
		LumaMode:          IntraModeDC,
	})
	if err != nil {
		t.Fatal(err)
	}
	if valid || mode != 0 {
		t.Fatalf("filter intra mode=%d valid=%v want disabled", mode, valid)
	}
	if got := cdfs.FilterIntra[BlockSize4x16].Values()[2]; got != 1 {
		t.Fatalf("filter intra cdf count=%d want 1", got)
	}
	if got := cdfs.FilterIntraMode.Values()[FilterIntraModes]; got != 0 {
		t.Fatalf("filter intra mode cdf count=%d want 0", got)
	}

	foundEnabled := false
	for b := 0; b <= 255 && !foundEnabled; b++ {
		if err := cdfs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		if err := state.Reset([]byte{byte(b), 0xff}, Job{Offset: 0, Size: 2}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		mode, valid, err = state.ReadFilterIntraMode(&cdfs, FilterIntraRequest{
			EnableFilterIntra: true,
			Size:              BlockSize4x16,
			LumaMode:          IntraModeDC,
		})
		if err != nil {
			t.Fatal(err)
		}
		if valid {
			if !mode.Valid() {
				t.Fatalf("enabled filter intra invalid mode=%d", mode)
			}
			if got := cdfs.FilterIntraMode.Values()[FilterIntraModes]; got != 1 {
				t.Fatalf("filter intra mode cdf count=%d want 1", got)
			}
			foundEnabled = true
		}
	}
	if !foundEnabled {
		t.Fatal("no single-byte payload decoded enabled filter intra")
	}

	before := cdfs.FilterIntra[BlockSize4x16].Values()[2]
	mode, valid, err = state.ReadFilterIntraMode(&cdfs, FilterIntraRequest{
		EnableFilterIntra: false,
		Size:              BlockSize4x16,
		LumaMode:          IntraModeDC,
	})
	if err != nil || valid || mode != 0 {
		t.Fatalf("disabled filter intra mode=%d valid=%v err=%v", mode, valid, err)
	}
	if got := cdfs.FilterIntra[BlockSize4x16].Values()[2]; got != before {
		t.Fatalf("disabled gate consumed cdf count=%d want %d", got, before)
	}
}

func TestFilterIntraAllowedMatchesLibaom(t *testing.T) {
	tests := []struct {
		name string
		req  FilterIntraRequest
		want bool
	}{
		{name: "disabled sequence", req: FilterIntraRequest{Size: BlockSize4x16, LumaMode: IntraModeDC}, want: false},
		{name: "dc small block", req: FilterIntraRequest{EnableFilterIntra: true, Size: BlockSize4x16, LumaMode: IntraModeDC}, want: true},
		{name: "non dc mode", req: FilterIntraRequest{EnableFilterIntra: true, Size: BlockSize4x16, LumaMode: IntraModeVertical}, want: false},
		{name: "palette y present", req: FilterIntraRequest{EnableFilterIntra: true, Size: BlockSize4x16, LumaMode: IntraModeDC, PaletteYSize: 2}, want: false},
		{name: "too wide", req: FilterIntraRequest{EnableFilterIntra: true, Size: BlockSize64x16, LumaMode: IntraModeDC}, want: false},
		{name: "too high", req: FilterIntraRequest{EnableFilterIntra: true, Size: BlockSize16x64, LumaMode: IntraModeDC}, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := FilterIntraAllowed(tt.req)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("allowed=%v want %v", got, tt.want)
			}
		})
	}
	if _, err := FilterIntraAllowed(FilterIntraRequest{EnableFilterIntra: true, Size: blockSizeCount, LumaMode: IntraModeDC}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad size err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestChromaIntraCFLAllowedMatchesLibaom(t *testing.T) {
	color420 := parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}
	tests := []struct {
		name     string
		size     BlockSize
		color    parser.ColorConfig
		lossless bool
		want     bool
	}{
		{name: "32x32", size: BlockSize32x32, color: color420, want: true},
		{name: "64x16 too wide", size: BlockSize64x16, color: color420, want: false},
		{name: "16x64 too high", size: BlockSize16x64, color: color420, want: false},
		{name: "lossless 8x8 420 plane 4x4", size: BlockSize8x8, color: color420, lossless: true, want: true},
		{name: "lossless 16x16 420 plane 8x8", size: BlockSize16x16, color: color420, lossless: true, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ChromaIntraCFLAllowed(tt.size, tt.color, tt.lossless)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("allowed=%v want %v", got, tt.want)
			}
		})
	}
	if _, err := ChromaIntraCFLAllowed(BlockSize8x8, parser.ColorConfig{MonoChrome: true}, false); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("monochrome err=%v want %v", err, ErrInvalidDecodeState)
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
	if _, err := cdfs.UVModeCDF(false, intraModeCount); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("bad uv cdf err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if _, err := cdfs.AngleDeltaCDF(IntraModeDC); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("bad angle cdf err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if _, _, err := state.ReadChromaIntraMode(&cdfs, ChromaIntraModeRequest{Size: blockSizeCount, LumaMode: IntraModeDC}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad chroma mode err=%v want %v", err, ErrInvalidDecodeState)
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
