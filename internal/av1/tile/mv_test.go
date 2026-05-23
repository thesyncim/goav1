package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
)

func TestMVCDFsInitDefaultMatchesDav1dAndLibaom(t *testing.T) {
	var cdfs MVCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}

	assertEntropyCDFValues(t, cdfs.Joint.Values(), []uint16{28672, 21504, 13440, 0, 0})
	comp := cdfs.Components[0]
	assertEntropyCDFValues(t, comp.Classes.Values(), []uint16{4096, 1792, 910, 448, 217, 112, 28, 11, 6, 1, 0, 0})
	assertEntropyCDFValues(t, comp.Class0FP[0].Values(), []uint16{16384, 8192, 6144, 0, 0})
	assertEntropyCDFValues(t, comp.Class0FP[1].Values(), []uint16{20480, 11520, 8640, 0, 0})
	assertEntropyCDFValues(t, comp.FP.Values(), []uint16{24576, 15360, 11520, 0, 0})
	assertEntropyCDFValues(t, comp.Sign.Values(), []uint16{16384, 0, 0})
	assertEntropyCDFValues(t, comp.Class0HP.Values(), []uint16{12288, 0, 0})
	assertEntropyCDFValues(t, comp.HP.Values(), []uint16{16384, 0, 0})
	assertEntropyCDFValues(t, comp.Class0.Values(), []uint16{5120, 0, 0})
	assertEntropyCDFValues(t, comp.Bits[0].Values(), []uint16{15360, 0, 0})
	assertEntropyCDFValues(t, cdfs.Components[1].Classes.Values(), comp.Classes.Values())
}

func TestReadMVComponentDiffClass0Precision(t *testing.T) {
	var cdfs MVCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00, 0x00}, Job{Offset: 0, Size: 2}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}

	diff, result, err := state.ReadMVComponentDiff(&cdfs.Components[0], MVSubpelHigh)
	if err != nil {
		t.Fatal(err)
	}
	if diff != 1 || result != (MVComponentResult{Class0: true, Fraction: 0, HighPrecision: 0, Diff: 1}) {
		t.Fatalf("diff=%d result=%+v", diff, result)
	}
	if got := cdfs.Components[0].Class0HP.Values()[2]; got != 1 {
		t.Fatalf("class0 hp count=%d want 1", got)
	}

	if err := state.Reset([]byte{0x00, 0x00}, Job{Offset: 0, Size: 2}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	diff, result, err = state.ReadMVComponentDiff(&cdfs.Components[0], MVSubpelNone)
	if err != nil {
		t.Fatal(err)
	}
	if diff != 8 || result.Fraction != 3 || result.HighPrecision != 1 {
		t.Fatalf("integer diff=%d result=%+v", diff, result)
	}
}

func TestReadMotionVectorZeroJoint(t *testing.T) {
	var cdfs MVCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}

	ref := motion.Vector{Row: 3, Col: -5}
	mv, residual, err := state.ReadMotionVector(&cdfs, ref, MVSubpelLow)
	if err != nil {
		t.Fatal(err)
	}
	if mv != ref || residual.Joint != MVJointZero || residual.Diff != (motion.Vector{}) {
		t.Fatalf("mv=%+v residual=%+v", mv, residual)
	}
	if got := cdfs.Joint.Values()[MVJoints]; got != 1 {
		t.Fatalf("joint count=%d want 1", got)
	}
}

func TestReadInterMotionAssignsModesAndResiduals(t *testing.T) {
	var cdfs MVCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	refs := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	req := InterMotionRequest{
		References: refs,
		Mode:       InterModeResult{Mode: InterModeNewMV},
		ReferenceMVs: InterMVReferenceSet{
			Nearest:  [2]motion.Vector{{Row: 1, Col: 2}},
			Near:     [2]motion.Vector{{Row: 3, Col: 4}},
			Residual: [2]motion.Vector{{Row: 5, Col: 6}},
		},
		Precision: MVSubpelLow,
	}

	result, err := state.ReadInterMotion(&cdfs, req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Motion != (InterMotionResult{References: refs, Mode: req.Mode, MV: [2]motion.Vector{{Row: 5, Col: 6}}}) {
		t.Fatalf("motion=%+v", result.Motion)
	}
	if !result.ResidualValid[0] || result.Residuals[0].Joint != MVJointZero {
		t.Fatalf("residuals=%+v valid=%v", result.Residuals, result.ResidualValid)
	}

	req.Mode = InterModeResult{Mode: InterModeNearMV}
	result, err = state.ReadInterMotion(&cdfs, req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Motion.MV[0] != (motion.Vector{Row: 3, Col: 4}) || result.ResidualValid[0] {
		t.Fatalf("near motion=%+v residual=%v", result.Motion, result.ResidualValid)
	}
}

func TestReadInterMotionCompoundAndGlobal(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	refs := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameAltref}, Compound: true}
	mode := InterModeResult{Compound: true, CompoundMode: CompoundInterModeGlobalGlobal}
	result, err := state.ReadInterMotion(nil, InterMotionRequest{
		References: refs,
		Mode:       mode,
		GlobalMVs: [2]motion.Vector{
			{Row: 7, Col: -9},
			{Row: -11, Col: 13},
		},
		Precision: MVSubpelHigh,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Motion.MV != ([2]motion.Vector{{Row: 7, Col: -9}, {Row: -11, Col: 13}}) {
		t.Fatalf("global motion=%+v", result.Motion)
	}
}

func TestReadMotionVectorRejectsInvalidInputs(t *testing.T) {
	var state DecodeState
	var cdfs MVCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := (*DecodeState)(nil).ReadMotionVector(&cdfs, motion.Vector{}, MVSubpelLow); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil state err=%v want %v", err, ErrInvalidDecodeState)
	}
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := state.ReadMotionVector(nil, motion.Vector{}, MVSubpelLow); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("nil cdfs err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if _, err := state.ReadInterMotion(nil, InterMotionRequest{
		References: InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}},
		Mode:       InterModeResult{Mode: InterModeNearestMV, Compound: true},
		Precision:  MVSubpelLow,
	}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad inter motion err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func FuzzReadMotionVector(f *testing.F) {
	f.Add([]byte{0x00}, int16(0), int16(0), int8(MVSubpelLow))
	f.Add([]byte{0xff, 0xff}, int16(3), int16(-5), int8(MVSubpelHigh))
	f.Add([]byte{0xa5, 0x5a, 0x00}, int16(-8), int16(9), int8(MVSubpelNone))

	f.Fuzz(func(t *testing.T, payload []byte, row int16, col int16, rawPrecision int8) {
		if len(payload) == 0 || len(payload) > 32 {
			return
		}
		precision := MVSubpelPrecision(rawPrecision)
		if !precision.Valid() {
			precision = MVPrecision(rawPrecision&1 != 0, rawPrecision&2 != 0)
		}
		var cdfs MVCDFs
		if err := cdfs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		_, residual, err := state.ReadMotionVector(&cdfs, motion.Vector{Row: int32(row), Col: int32(col)}, precision)
		if err != nil {
			return
		}
		if !residual.Joint.Valid() || residual.Precision != precision {
			t.Fatalf("bad residual=%+v precision=%d", residual, precision)
		}
	})
}
