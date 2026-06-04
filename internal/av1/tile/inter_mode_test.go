package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestInterModeCDFsInitDefaultMatchesDav1dAndLibaom(t *testing.T) {
	var cdfs InterModeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		cdf  *entropy.CDF
		want []uint16
	}{
		{name: "newmv ctx0", cdf: &cdfs.NewMV[0], want: []uint16{8733, 0, 0}},
		{name: "globalmv ctx1", cdf: &cdfs.GlobalMV[1], want: []uint16{31714, 0, 0}},
		{name: "refmv ctx2", cdf: &cdfs.RefMV[2], want: []uint16{14920, 0, 0}},
		{name: "drl ctx0", cdf: &cdfs.DRL[0], want: []uint16{19664, 0, 0}},
		{name: "compound ctx0", cdf: &cdfs.Compound[0], want: []uint16{25008, 18945, 16960, 15127, 13612, 12102, 5877, 0, 0}},
		{name: "compound ctx7", cdf: &cdfs.Compound[7], want: []uint16{19722, 9554, 8263, 6826, 5333, 4326, 3438, 0, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertEntropyCDFValues(t, tt.cdf.Values(), tt.want)
		})
	}
}

func TestAnalyzeInterModeContextMatchesLibaom(t *testing.T) {
	got, err := AnalyzeInterModeContext(0, false)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Fatalf("single ctx=%d want 0", got)
	}

	raw := uint16((4 << refMVOffset) | 6)
	got, err = AnalyzeInterModeContext(raw, true)
	if err != nil {
		t.Fatal(err)
	}
	if got != 7 {
		t.Fatalf("compound ctx=%d want 7", got)
	}

	raw = uint16((2 << refMVOffset) | 3)
	got, err = AnalyzeInterModeContext(raw, true)
	if err != nil {
		t.Fatal(err)
	}
	if got != 4 {
		t.Fatalf("compound ctx=%d want 4", got)
	}

	if _, err := AnalyzeInterModeContext(uint16(6), false); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad single ctx err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := AnalyzeInterModeContext(uint16(6<<refMVOffset), true); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad compound ctx err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestReadInterModes(t *testing.T) {
	var cdfs InterModeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}

	mode, err := state.ReadSingleInterMode(&cdfs, 0)
	if err != nil {
		t.Fatal(err)
	}
	if mode != InterModeNewMV {
		t.Fatalf("single mode=%d want NEWMV", mode)
	}
	if got := cdfs.NewMV[0].Values()[2]; got != 1 {
		t.Fatalf("newmv count=%d want 1", got)
	}

	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	compound, err := state.ReadCompoundInterMode(&cdfs, 0)
	if err != nil {
		t.Fatal(err)
	}
	if compound != CompoundInterModeNearestNearest {
		t.Fatalf("compound mode=%d want NEAREST_NEAREST", compound)
	}
	if got := cdfs.Compound[0].Values()[int(compoundInterModeCount)]; got != 1 {
		t.Fatalf("compound count=%d want 1", got)
	}
}

func TestReadBlockInterModeForcedPaths(t *testing.T) {
	var cdfs InterModeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset([]byte{0xff}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}

	result, err := state.ReadBlockInterMode(&cdfs, InterModeRequest{Compound: true, SkipMode: true})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Compound || result.CompoundMode != CompoundInterModeNearestNearest {
		t.Fatalf("skip mode result=%+v", result)
	}
	if cdfs.Compound[0].Values()[int(compoundInterModeCount)] != 0 {
		t.Fatal("skip-mode path updated compound cdf")
	}

	result, err = state.ReadBlockInterMode(&cdfs, InterModeRequest{
		SegmentationEnabled: true,
		Segment:             segmentGlobalMV(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Compound || result.Mode != InterModeGlobalMV {
		t.Fatalf("segment-global result=%+v", result)
	}
}

func TestReadDRLIndex(t *testing.T) {
	var cdfs InterModeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}

	index, err := state.ReadDRLIndex(&cdfs, DRLRequest{
		Mode:       InterModeNewMV,
		RefMVCount: 3,
		Contexts:   [3]uint8{0, 1, 2},
	})
	if err != nil {
		t.Fatal(err)
	}
	if index != 0 {
		t.Fatalf("newmv drl index=%d want 0", index)
	}
	if got := cdfs.DRL[0].Values()[2]; got != 1 {
		t.Fatalf("drl ctx0 count=%d want 1", got)
	}

	before := cdfs.DRL[1]
	index, err = state.ReadDRLIndex(&cdfs, DRLRequest{
		Mode:       InterModeNearMV,
		RefMVCount: 2,
		Contexts:   [3]uint8{0, 1, 2},
	})
	if err != nil {
		t.Fatal(err)
	}
	if index != 0 || cdfs.DRL[1] != before {
		t.Fatalf("near no-read index=%d cdf changed=%v", index, cdfs.DRL[1] != before)
	}

	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	before = cdfs.DRL[0]
	index, err = state.ReadDRLIndex(&cdfs, DRLRequest{
		Compound:     true,
		CompoundMode: CompoundInterModeNearestNew,
		RefMVCount:   3,
		Contexts:     [3]uint8{0, 1, 2},
	})
	if err != nil {
		t.Fatal(err)
	}
	if index != 0 || cdfs.DRL[0] != before {
		t.Fatalf("nearest-new drl index=%d cdf changed=%v", index, cdfs.DRL[0] != before)
	}

	index, err = state.ReadDRLIndex(&cdfs, DRLRequest{
		Compound:     true,
		CompoundMode: CompoundInterModeNewNew,
		RefMVCount:   3,
		Contexts:     [3]uint8{0, 1, 2},
	})
	if err != nil {
		t.Fatal(err)
	}
	if index != 0 || cdfs.DRL[0] == before {
		t.Fatalf("new-new drl index=%d cdf unchanged=%v", index, cdfs.DRL[0] == before)
	}
}

func TestInterModeRejectsInvalidInputs(t *testing.T) {
	var cdfs InterModeCDFs
	if _, err := cdfs.NewMVCDF(0); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("uninitialized cdf err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if _, err := cdfs.RefMVCDF(RefMVModeContexts); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("bad cdf err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if _, err := CompoundInterMode(compoundInterModeCount).Components(); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad compound mode err=%v want %v", err, ErrInvalidDecodeState)
	}

	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var nilState *DecodeState
	if _, err := nilState.ReadSingleInterMode(&cdfs, 0); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil single mode err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.ReadSingleInterMode(&cdfs, 6); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("bad single mode ctx err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if _, err := state.ReadBlockInterMode(&cdfs, InterModeRequest{Compound: false, SkipMode: true}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad skip inter mode err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.ReadDRLIndex(&cdfs, DRLRequest{Mode: interModeCount}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad drl mode err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.ReadDRLIndex(&cdfs, DRLRequest{Mode: InterModeNewMV, RefMVCount: 3, Contexts: [3]uint8{3}}); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("bad drl ctx err=%v want %v", err, entropy.ErrInvalidCDF)
	}
}

func TestInterModeAllocs(t *testing.T) {
	var cdfs InterModeCDFs
	var state DecodeState
	payload := []byte{0x00}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := cdfs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		if err := state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		result, err := state.ReadBlockInterMode(&cdfs, InterModeRequest{ModeContext: 0})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := state.ReadDRLIndex(&cdfs, DRLRequest{
			Mode:       result.Mode,
			RefMVCount: 3,
			Contexts:   [3]uint8{0, 1, 2},
		}); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("inter mode decode allocated: %f", allocs)
	}
}

func FuzzReadInterMode(f *testing.F) {
	f.Add([]byte{0x00}, uint16(0), false, false, uint8(3), uint8(0), uint8(1), uint8(2))
	f.Add([]byte{0xff}, uint16((4<<refMVOffset)|6), true, false, uint8(4), uint8(2), uint8(1), uint8(0))
	f.Add([]byte{0xa5, 0x5a}, uint16((2<<refMVOffset)|3), true, true, uint8(2), uint8(0), uint8(0), uint8(0))

	f.Fuzz(func(t *testing.T, payload []byte, rawModeContext uint16, compound bool, skipMode bool, rawRefMVCount uint8, rawCtx0 uint8, rawCtx1 uint8, rawCtx2 uint8) {
		if len(payload) == 0 || len(payload) > 64 {
			return
		}
		var cdfs InterModeCDFs
		if err := cdfs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		result, err := state.ReadBlockInterMode(&cdfs, InterModeRequest{
			Compound:    compound,
			SkipMode:    skipMode && compound,
			ModeContext: rawModeContext,
		})
		if err != nil {
			if errors.Is(err, ErrInvalidDecodeState) || errors.Is(err, entropy.ErrInvalidCDF) {
				return
			}
			t.Fatalf("ReadBlockInterMode err=%v", err)
		}
		drlReq := DRLRequest{
			Mode:         result.Mode,
			CompoundMode: result.CompoundMode,
			Compound:     result.Compound,
			RefMVCount:   rawRefMVCount % 5,
			Contexts:     [3]uint8{rawCtx0 % DRLModeContexts, rawCtx1 % DRLModeContexts, rawCtx2 % DRLModeContexts},
		}
		if _, err := state.ReadDRLIndex(&cdfs, drlReq); err != nil {
			t.Fatalf("ReadDRLIndex err=%v req=%+v", err, drlReq)
		}
	})
}

func segmentGlobalMV() parser.SegmentData {
	return parser.SegmentData{RefFrame: -1, GlobalMV: true}
}
