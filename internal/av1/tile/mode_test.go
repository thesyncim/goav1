package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestBlockModeCDFsInitDefaultMatchesDav1dAndLibaom(t *testing.T) {
	var cdfs BlockModeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		cdf  *entropy.CDF
		want []uint16
	}{
		{name: "skip ctx0", cdf: &cdfs.Skip[0], want: []uint16{1097, 0, 0}},
		{name: "skip ctx1", cdf: &cdfs.Skip[1], want: []uint16{16253, 0, 0}},
		{name: "skip ctx2", cdf: &cdfs.Skip[2], want: []uint16{28192, 0, 0}},
		{name: "skip mode ctx0", cdf: &cdfs.SkipMode[0], want: []uint16{147, 0, 0}},
		{name: "skip mode ctx1", cdf: &cdfs.SkipMode[1], want: []uint16{12060, 0, 0}},
		{name: "skip mode ctx2", cdf: &cdfs.SkipMode[2], want: []uint16{24641, 0, 0}},
		{name: "segment pred", cdf: &cdfs.SegmentPred[0], want: []uint16{16384, 0, 0}},
		{name: "segment id ctx0", cdf: &cdfs.SegmentID[0], want: []uint16{27146, 24875, 16675, 14535, 4959, 4395, 235, 0, 0}},
		{name: "segment id ctx1", cdf: &cdfs.SegmentID[1], want: []uint16{18494, 14538, 10211, 7833, 2788, 1917, 424, 0, 0}},
		{name: "segment id ctx2", cdf: &cdfs.SegmentID[2], want: []uint16{5241, 4281, 4045, 3878, 371, 121, 89, 0, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertEntropyCDFValues(t, tt.cdf.Values(), tt.want)
		})
	}
}

func TestBlockModeContextMatchesLibaomContexts(t *testing.T) {
	var ctx BlockModeContext
	if got, err := ctx.SkipContext(0, 0); err != nil || got != 0 {
		t.Fatalf("initial skip ctx=%d err=%v", got, err)
	}
	if got, err := ctx.SkipModeContext(0, 0); err != nil || got != 0 {
		t.Fatalf("initial skip-mode ctx=%d err=%v", got, err)
	}
	if got, err := ctx.SegmentPredContext(0, 0); err != nil || got != 0 {
		t.Fatalf("initial seg-pred ctx=%d err=%v", got, err)
	}

	result := BlockModeResult{SegmentPredicted: true, SkipMode: true, SkipTransform: true}
	if err := ctx.Mark(BlockSize16x8, 4, 6, result); err != nil {
		t.Fatal(err)
	}
	dims, _ := BlockSize16x8.Dimensions()
	for i := 0; i < int(dims.W4); i++ {
		if ctx.AboveSegmentPred[4+i] != 1 || ctx.AboveSkipMode[4+i] != 1 || ctx.AboveSkip[4+i] != 1 {
			t.Fatalf("above slot %d seg=%d skip_mode=%d skip=%d", i,
				ctx.AboveSegmentPred[4+i], ctx.AboveSkipMode[4+i], ctx.AboveSkip[4+i])
		}
	}
	for i := 0; i < int(dims.H4); i++ {
		if ctx.LeftSegmentPred[6+i] != 1 || ctx.LeftSkipMode[6+i] != 1 || ctx.LeftSkip[6+i] != 1 {
			t.Fatalf("left slot %d seg=%d skip_mode=%d skip=%d", i,
				ctx.LeftSegmentPred[6+i], ctx.LeftSkipMode[6+i], ctx.LeftSkip[6+i])
		}
	}
	if got, err := ctx.SkipContext(4, 6); err != nil || got != 2 {
		t.Fatalf("marked skip ctx=%d err=%v", got, err)
	}
}

func TestReadBlockModePrefix(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var cdfs BlockModeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var ctx BlockModeContext
	req := BlockModeRequest{
		Size: BlockSize16x16,
		SkipMode: parser.SkipModeParams{
			Allowed: true,
			Enabled: true,
		},
		CDEF:    parser.CDEFParams{Bits: 2, StrengthCount: 4},
		Segment: parser.SegmentData{RefFrame: -1},
	}

	result, err := state.ReadBlockModePrefix(&cdfs, &ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if result.SkipMode || result.SkipTransform || result.CDEFIndex != 0 {
		t.Fatalf("prefix=%+v", result)
	}
	assertEntropyCDFValues(t, cdfs.SkipMode[0].Values(), []uint16{138, 0, 1})
	assertEntropyCDFValues(t, cdfs.Skip[0].Values(), []uint16{1029, 0, 1})
}

func TestReadBlockModePrefixSegmentSkipForcesSkipAndSuppressesCDEF(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0xff}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var cdfs BlockModeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var ctx BlockModeContext
	req := BlockModeRequest{
		Size:                BlockSize16x16,
		SkipMode:            parser.SkipModeParams{Allowed: true, Enabled: true},
		CDEF:                parser.CDEFParams{Bits: 2, StrengthCount: 4},
		SegmentationEnabled: true,
		Segment:             parser.SegmentData{RefFrame: -1, Skip: true},
	}

	result, err := state.ReadBlockModePrefix(&cdfs, &ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if result.SkipMode || !result.SkipTransform || result.CDEFIndex != 0 {
		t.Fatalf("prefix=%+v", result)
	}
	if ctx.AboveSkip[0] != 1 || ctx.LeftSkip[0] != 1 {
		t.Fatalf("skip context above=%d left=%d", ctx.AboveSkip[0], ctx.LeftSkip[0])
	}
	assertEntropyCDFValues(t, cdfs.SkipMode[0].Values(), []uint16{147, 0, 0})
	assertEntropyCDFValues(t, cdfs.Skip[0].Values(), []uint16{1097, 0, 0})
}

func TestReadSkipModeConditionsPortLibaom(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0xff}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var cdfs BlockModeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var ctx BlockModeContext

	req := BlockModeRequest{
		Size:     BlockSize4x4,
		SkipMode: parser.SkipModeParams{Allowed: true, Enabled: true},
		Segment:  parser.SegmentData{RefFrame: -1},
	}
	skipMode, err := state.ReadSkipMode(&cdfs, &ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if skipMode {
		t.Fatal("4x4 block should imply skip_mode=0")
	}
	assertEntropyCDFValues(t, cdfs.SkipMode[0].Values(), []uint16{147, 0, 0})

	req.Size = BlockSize16x16
	req.SegmentationEnabled = true
	req.Segment = parser.SegmentData{RefFrame: 2}
	skipMode, err = state.ReadSkipMode(&cdfs, &ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if skipMode {
		t.Fatal("segment ref-frame feature should imply skip_mode=0")
	}
	assertEntropyCDFValues(t, cdfs.SkipMode[0].Values(), []uint16{147, 0, 0})
}

func TestReadCDEFIndex(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0xff}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	index, err := state.ReadCDEFIndex(parser.CDEFParams{Bits: 2, StrengthCount: 4}, false)
	if err != nil {
		t.Fatal(err)
	}
	if index != 3 {
		t.Fatalf("cdef index=%d want 3", index)
	}

	before := state.Reader.BitsRead()
	index, err = state.ReadCDEFIndex(parser.CDEFParams{Bits: 2, StrengthCount: 4}, true)
	if err != nil {
		t.Fatal(err)
	}
	if index != 0 || state.Reader.BitsRead() != before {
		t.Fatalf("skipped cdef index=%d bits before=%d after=%d", index, before, state.Reader.BitsRead())
	}
}

func TestReadCDEFIndexForBlockCachesPerLibaomUnit(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0xcf}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	params := parser.CDEFParams{Bits: 2, StrengthCount: 4}
	var ctx CDEFIndexContext
	ctx.Reset()

	index, err := state.ReadCDEFIndexForBlock(params, &ctx, BlockSize8x8, 0, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if index != 3 {
		t.Fatalf("first cdef index=%d want 3", index)
	}
	afterFirst := state.Reader.BitsRead()
	if !ctx.Read[0] || ctx.Index[0] != 3 {
		t.Fatalf("unit 0 cache read=%v index=%d want true,3", ctx.Read[0], ctx.Index[0])
	}

	index, err = state.ReadCDEFIndexForBlock(params, &ctx, BlockSize16x16, 8, 8, false)
	if err != nil {
		t.Fatal(err)
	}
	if index != 3 || state.Reader.BitsRead() != afterFirst {
		t.Fatalf("cached cdef index=%d bits=%d want index 3 bits %d", index, state.Reader.BitsRead(), afterFirst)
	}

	index, err = state.ReadCDEFIndexForBlock(params, &ctx, BlockSize8x8, 16, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if index != 0 || state.Reader.BitsRead() != afterFirst+2 {
		t.Fatalf("right-unit cdef index=%d bits=%d want index 0 bits %d", index, state.Reader.BitsRead(), afterFirst+2)
	}
}

func TestReadCDEFIndexForBlockSpanningBlockMarksCoveredUnits(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x80}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	params := parser.CDEFParams{Bits: 2, StrengthCount: 4}
	var ctx CDEFIndexContext
	ctx.Reset()

	index, err := state.ReadCDEFIndexForBlock(params, &ctx, BlockSize128x128, 0, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if index != 2 {
		t.Fatalf("spanning cdef index=%d want 2", index)
	}
	afterFirst := state.Reader.BitsRead()
	for unit, cached := range ctx.Index {
		if !ctx.Read[unit] || cached != 2 {
			t.Fatalf("unit %d read=%v cached=%d want true,2", unit, ctx.Read[unit], cached)
		}
	}

	index, err = state.ReadCDEFIndexForBlock(params, &ctx, BlockSize16x16, 16, 16, false)
	if err != nil {
		t.Fatal(err)
	}
	if index != 2 || state.Reader.BitsRead() != afterFirst {
		t.Fatalf("covered-unit cdef index=%d bits=%d want index 2 bits %d", index, state.Reader.BitsRead(), afterFirst)
	}
}

func TestSegmentPredictionAndID(t *testing.T) {
	cur := []uint8{
		0, 0, 0, 0,
		0, 2, 2, 0,
		0, 3, 4, 0,
		0, 0, 0, 0,
	}
	pred, ctx, err := PredictCurrentSegmentID(cur, 4, 2, 2, true, true)
	if err != nil {
		t.Fatal(err)
	}
	if pred != 2 || ctx != 1 {
		t.Fatalf("pred=%d ctx=%d want 2,1", pred, ctx)
	}
	pred, ctx, err = PredictCurrentSegmentID(cur, 4, 1, 0, false, true)
	if err != nil {
		t.Fatal(err)
	}
	if pred != 0 || ctx != 0 {
		t.Fatalf("left pred=%d ctx=%d", pred, ctx)
	}

	prev := []uint8{
		7, 7, 7, 7,
		7, 5, 3, 7,
		7, 4, 0, 7,
		7, 7, 7, 7,
	}
	minID, err := MinPreviousSegmentID(prev, 4, 1, 1, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if minID != 0 {
		t.Fatalf("previous min=%d want 0", minID)
	}

	dst := make([]uint8, 16)
	if err := FillSegmentID(dst, 4, 1, 1, 2, 2, 6); err != nil {
		t.Fatal(err)
	}
	for _, idx := range []int{5, 6, 9, 10} {
		if dst[idx] != 6 {
			t.Fatalf("dst[%d]=%d want 6", idx, dst[idx])
		}
	}
}

func TestReadSegmentPredictionAndID(t *testing.T) {
	var cdfs BlockModeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var ctx BlockModeContext
	if err := ctx.Mark(BlockSize4x4, 0, 0, BlockModeResult{SegmentPredicted: true}); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	predicted, err := state.ReadSegmentPrediction(&cdfs, &ctx, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if predicted {
		t.Fatal("segment prediction flag=true want false")
	}
	assertEntropyCDFValues(t, cdfs.SegmentPred[2].Values(), []uint16{15360, 0, 1})

	id, err := state.ReadSegmentID(&cdfs, 2, 0, 7, false)
	if err != nil {
		t.Fatal(err)
	}
	if id != 2 {
		t.Fatalf("segment id=%d want 2", id)
	}
	if got := cdfs.SegmentID[0].Values()[parser.MaxSegments]; got != 1 {
		t.Fatalf("segment cdf count=%d want 1", got)
	}

	before := cdfs.SegmentID[0]
	id, err = state.ReadSegmentID(&cdfs, 2, 0, 7, true)
	if err != nil {
		t.Fatal(err)
	}
	if id != 2 || cdfs.SegmentID[0] != before {
		t.Fatalf("skipped segment id=%d cdf changed=%v", id, cdfs.SegmentID[0] != before)
	}
}

func TestNegDeinterleaveSegmentIDMatchesLibaom(t *testing.T) {
	tests := []struct {
		diff int
		ref  int
		max  int
		want int
	}{
		{diff: 3, ref: 0, max: 8, want: 3},
		{diff: 2, ref: 7, max: 8, want: 5},
		{diff: 0, ref: 2, max: 8, want: 2},
		{diff: 1, ref: 2, max: 8, want: 3},
		{diff: 2, ref: 2, max: 8, want: 1},
		{diff: 5, ref: 2, max: 8, want: 5},
		{diff: 0, ref: 5, max: 8, want: 5},
		{diff: 1, ref: 5, max: 8, want: 6},
		{diff: 2, ref: 5, max: 8, want: 4},
		{diff: 5, ref: 5, max: 8, want: 2},
	}
	for _, tt := range tests {
		got, err := negDeinterleaveSegmentID(tt.diff, tt.ref, tt.max)
		if err != nil {
			t.Fatalf("%+v err=%v", tt, err)
		}
		if got != tt.want {
			t.Fatalf("%+v got=%d", tt, got)
		}
	}
}

func TestBlockModeRejectsInvalidInputs(t *testing.T) {
	var cdfs BlockModeCDFs
	if _, err := cdfs.SkipCDF(0); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("uninitialized SkipCDF err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if _, err := cdfs.SkipCDF(BlockModeContexts); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("bad SkipCDF err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if _, err := (*BlockModeCDFs)(nil).SkipCDF(0); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("nil SkipCDF err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	var ctx BlockModeContext
	if _, err := ctx.SkipContext(MaxBlockModeSlots, 0); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad ctx err=%v want %v", err, ErrInvalidDecodeState)
	}
	if err := ctx.Mark(blockSizeCount, 0, 0, BlockModeResult{}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad mark err=%v want %v", err, ErrInvalidDecodeState)
	}

	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var nilState *DecodeState
	if _, err := nilState.ReadCDEFIndex(parser.CDEFParams{}, false); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil CDEF err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.ReadCDEFIndex(parser.CDEFParams{Bits: 4}, false); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad CDEF err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.ReadBlockModePrefix(&cdfs, nil, BlockModeRequest{Size: BlockSize16x16}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil prefix ctx err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.ReadSegmentID(&cdfs, 8, 0, 7, false); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad segment pred err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, _, err := PredictCurrentSegmentID([]uint8{0}, 1, 0, 0, true, false); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad current seg err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := MinPreviousSegmentID([]uint8{0}, 1, 0, 0, 2, 1); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad previous seg err=%v want %v", err, ErrInvalidDecodeState)
	}
	if err := FillSegmentID([]uint8{0}, 1, 0, 0, 1, 1, 8); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad fill seg err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestBlockModeAllocs(t *testing.T) {
	var cdfs BlockModeCDFs
	var ctx BlockModeContext
	var state DecodeState
	payload := []byte{0x00}
	req := BlockModeRequest{
		Size:     BlockSize16x16,
		SkipMode: parser.SkipModeParams{Allowed: true, Enabled: true},
		Segment:  parser.SegmentData{RefFrame: -1},
	}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := cdfs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		ctx = BlockModeContext{}
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		if _, err := state.ReadBlockModePrefix(&cdfs, &ctx, req); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("block mode prefix allocated: %f", allocs)
	}
}

func FuzzReadBlockModePrefix(f *testing.F) {
	f.Add([]byte{0x00}, uint8(BlockSize16x16), uint8(0), uint8(0), uint8(0), false, false, false)
	f.Add([]byte{0xff}, uint8(BlockSize16x16), uint8(2), uint8(0), uint8(0), true, true, false)
	f.Add([]byte{0xa5, 0x5a}, uint8(BlockSize8x8), uint8(3), uint8(3), uint8(7), true, false, true)

	f.Fuzz(func(t *testing.T, payload []byte, rawSize uint8, rawX uint8, rawY uint8, rawCDEF uint8, skipModeEnabled bool, segEnabled bool, segSkip bool) {
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

		var cdfs BlockModeCDFs
		if err := cdfs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		var ctx BlockModeContext
		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		req := BlockModeRequest{
			Size:                size,
			SkipMode:            parser.SkipModeParams{Allowed: true, Enabled: skipModeEnabled},
			CDEF:                parser.CDEFParams{Bits: rawCDEF & maxCDEFBits},
			SegmentationEnabled: segEnabled,
			Segment:             parser.SegmentData{RefFrame: -1, Skip: segSkip},
			X4:                  uint8(int(rawX) % xLimit),
			Y4:                  uint8(int(rawY) % yLimit),
		}
		result, err := state.ReadBlockModePrefix(&cdfs, &ctx, req)
		if err != nil {
			t.Fatalf("ReadBlockModePrefix err=%v req=%+v", err, req)
		}
		if result.CDEFIndex >= parser.MaxCDEFStrengths {
			t.Fatalf("CDEFIndex=%d", result.CDEFIndex)
		}
		if segEnabled && segSkip && !result.SkipTransform {
			t.Fatalf("segment skip did not force skip: %+v", result)
		}
	})
}
