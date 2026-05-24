package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestInterpFilterCDFsInitDefaultMatchesLibaom(t *testing.T) {
	var cdfs InterpFilterCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		cdf  *entropy.CDF
		want []uint16
	}{
		{name: "ctx0", cdf: &cdfs.Switchable[0], want: []uint16{833, 48, 0, 0}},
		{name: "ctx3", cdf: &cdfs.Switchable[3], want: []uint16{4524, 160, 0, 0}},
		{name: "ctx11", cdf: &cdfs.Switchable[11], want: []uint16{5365, 132, 0, 0}},
		{name: "ctx15", cdf: &cdfs.Switchable[15], want: []uint16{17799, 11370, 0, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertEntropyCDFValues(t, tt.cdf.Values(), tt.want)
		})
	}
}

func TestSwitchableInterpContextMatchesLibaom(t *testing.T) {
	refs := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	base := InterpFilterRequest{
		FrameFilter: parser.InterpolationSwitchable,
		Size:        BlockSize16x16,
		References:  refs,
		Mode:        InterModeResult{Mode: InterModeNewMV},
		MotionMode:  MotionModeTranslation,
		X4:          0,
		Y4:          0,
	}
	var ctx BlockModeContext

	got, err := ctx.SwitchableInterpContext(base, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got != 3 {
		t.Fatalf("no-neighbor dir0 ctx=%d want 3", got)
	}
	got, err = ctx.SwitchableInterpContext(base, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got != 11 {
		t.Fatalf("no-neighbor dir1 ctx=%d want 11", got)
	}

	if err := ctx.MarkInter(BlockSize8x8, 0, 0, refs); err != nil {
		t.Fatal(err)
	}
	if err := ctx.MarkInterFilters(BlockSize8x8, 0, 0, refs, motion.InterpFilters{X: motion.InterpEightTapSmooth, Y: motion.InterpEightTapRegular}); err != nil {
		t.Fatal(err)
	}
	withLeft := base
	withLeft.HaveLeft = true
	got, err = ctx.SwitchableInterpContext(withLeft, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Fatalf("left regular dir0 ctx=%d want 0", got)
	}
	got, err = ctx.SwitchableInterpContext(withLeft, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got != 9 {
		t.Fatalf("left smooth dir1 ctx=%d want 9", got)
	}

	compound := base
	compound.References = InterReferencesResult{Compound: true, Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameAltref}}
	compound.Mode = InterModeResult{Compound: true, CompoundMode: CompoundInterModeNearestNearest}
	got, err = ctx.SwitchableInterpContext(compound, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got != 7 {
		t.Fatalf("compound no-neighbor dir0 ctx=%d want 7", got)
	}
	got, err = ctx.SwitchableInterpContext(compound, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got != 15 {
		t.Fatalf("compound no-neighbor dir1 ctx=%d want 15", got)
	}
}

func TestReadInterpFiltersPortsLibaomGates(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00, 0x00}, Job{Offset: 0, Size: 2}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	refs := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	req := InterpFilterRequest{
		FrameFilter: parser.InterpolationSwitchable,
		Size:        BlockSize16x16,
		References:  refs,
		Mode:        InterModeResult{Mode: InterModeNewMV},
		MotionMode:  MotionModeTranslation,
		X4:          0,
		Y4:          0,
	}

	skipReq := req
	skipReq.SkipMode = true
	filters, reads, err := state.ReadInterpFilters(nil, &BlockModeContext{}, skipReq)
	if err != nil {
		t.Fatal(err)
	}
	if filters != motion.RegularFilters || reads != 0 {
		t.Fatalf("skip filters=%+v reads=%d want regular/0", filters, reads)
	}

	fixedReq := req
	fixedReq.FrameFilter = parser.InterpolationSharp
	filters, reads, err = state.ReadInterpFilters(nil, &BlockModeContext{}, fixedReq)
	if err != nil {
		t.Fatal(err)
	}
	if filters != (motion.InterpFilters{X: motion.InterpMultiTapSharp, Y: motion.InterpMultiTapSharp}) || reads != 0 {
		t.Fatalf("fixed filters=%+v reads=%d", filters, reads)
	}

	_, _, err = state.ReadInterpFilters(nil, &BlockModeContext{}, req)
	if !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("switchable without CDF err=%v want %v", err, entropy.ErrInvalidCDF)
	}
}

func TestReadInterpFiltersConsumesSwitchableCDFs(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00, 0x00}, Job{Offset: 0, Size: 2}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var cdfs InterpFilterCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	req := InterpFilterRequest{
		FrameFilter:      parser.InterpolationSwitchable,
		EnableDualFilter: true,
		Size:             BlockSize16x16,
		References:       InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}},
		Mode:             InterModeResult{Mode: InterModeNewMV},
		MotionMode:       MotionModeTranslation,
		X4:               0,
		Y4:               0,
	}

	filters, reads, err := state.ReadInterpFilters(&cdfs, &BlockModeContext{}, req)
	if err != nil {
		t.Fatal(err)
	}
	if !switchableMotionFilter(filters.X) || !switchableMotionFilter(filters.Y) || reads != 2 {
		t.Fatalf("filters=%+v reads=%d", filters, reads)
	}
	if got := cdfs.Switchable[3].Values()[switchableFilterCount]; got != 1 {
		t.Fatalf("dir0 ctx count=%d want 1", got)
	}
	if got := cdfs.Switchable[11].Values()[switchableFilterCount]; got != 1 {
		t.Fatalf("dir1 ctx count=%d want 1", got)
	}
}
