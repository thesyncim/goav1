package tile

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
)

// TestWriteInterSymbolsRoundTrip is the oracle gate for the single-reference
// inter writers: a mixed stream of inter modes, DRL indices, and motion
// vectors across all precisions coded with adapting CDFs must decode back
// exactly through ReadSingleInterMode/ReadDRLIndex/ReadMotionVector, decoder
// CDFs adapting in lockstep.
func TestWriteInterSymbolsRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(23))

	type op struct {
		kind int // 0 mode, 1 drl, 2 mv

		modeCtx uint16
		mode    InterMode

		drl    DRLRequest
		drlIdx int

		mv, ref   motion.Vector
		precision MVSubpelPrecision
	}

	const n = 6000
	ops := make([]op, 0, n)

	var encModes InterModeCDFs
	var encMV MVCDFs
	if err := encModes.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if err := encMV.InitDefault(); err != nil {
		t.Fatal(err)
	}
	w := entropy.NewWriter(make([]byte, 0, 1<<16))

	modes := []InterMode{InterModeNewMV, InterModeGlobalMV, InterModeNearestMV, InterModeNearMV}
	precisions := []MVSubpelPrecision{MVSubpelNone, MVSubpelLow, MVSubpelHigh}

	for range n {
		var o op
		switch rng.Intn(3) {
		case 0:
			// Pack a valid mode context: newmv bits 0..2, globalmv bit 3,
			// refmv bits 4..7, each within its CDF's context count.
			modeCtx := uint16(rng.Intn(6)) | uint16(rng.Intn(2))<<globalMVOffset | uint16(rng.Intn(6))<<refMVOffset
			o = op{kind: 0, modeCtx: modeCtx, mode: modes[rng.Intn(len(modes))]}
			if err := WriteSingleInterMode(&w, &encModes, o.modeCtx, o.mode); err != nil {
				t.Fatalf("WriteSingleInterMode ctx=%d mode=%d: %v", o.modeCtx, o.mode, err)
			}
		case 1:
			mode := modes[rng.Intn(len(modes))]
			count := uint8(1 + rng.Intn(4))
			req := DRLRequest{Mode: mode, RefMVCount: count}
			for i := range req.Contexts {
				req.Contexts[i] = uint8(rng.Intn(3))
			}
			maxIdx := 0
			switch {
			case req.usesNewMV():
				maxIdx = min(2, int(count)-1)
			case req.usesNearMV():
				maxIdx = max(0, min(2, int(count)-2))
			}
			idx := 0
			if maxIdx > 0 {
				idx = rng.Intn(maxIdx + 1)
			}
			o = op{kind: 1, drl: req, drlIdx: idx}
			if err := WriteDRLIndex(&w, &encModes, req, idx); err != nil {
				t.Fatalf("WriteDRLIndex mode=%d count=%d idx=%d: %v", mode, count, idx, err)
			}
		default:
			precision := precisions[rng.Intn(len(precisions))]
			ref := motion.Vector{Row: int16(rng.Intn(257) - 128), Col: int16(rng.Intn(257) - 128)}
			diffRow := randomMVDiff(rng, precision)
			diffCol := randomMVDiff(rng, precision)
			mv := motion.Vector{Row: ref.Row + int16(diffRow), Col: ref.Col + int16(diffCol)}
			o = op{kind: 2, mv: mv, ref: ref, precision: precision}
			if err := WriteMotionVector(&w, &encMV, mv, ref, precision); err != nil {
				t.Fatalf("WriteMotionVector mv=%v ref=%v prec=%d: %v", mv, ref, precision, err)
			}
		}
		ops = append(ops, o)
	}
	buf, err := w.Finish()
	if err != nil {
		t.Fatal(err)
	}

	var decModes InterModeCDFs
	var decMV MVCDFs
	if err := decModes.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if err := decMV.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset(buf, Job{Offset: 0, Size: uint32(len(buf))}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	for i, o := range ops {
		switch o.kind {
		case 0:
			got, err := state.ReadSingleInterMode(&decModes, o.modeCtx)
			if err != nil {
				t.Fatalf("op %d ReadSingleInterMode: %v", i, err)
			}
			if got != o.mode {
				t.Fatalf("op %d mode=%d want %d (ctx=%d)", i, got, o.mode, o.modeCtx)
			}
		case 1:
			got, err := state.ReadDRLIndex(&decModes, o.drl)
			if err != nil {
				t.Fatalf("op %d ReadDRLIndex: %v", i, err)
			}
			if got != o.drlIdx {
				t.Fatalf("op %d drl=%d want %d (mode=%d count=%d)", i, got, o.drlIdx, o.drl.Mode, o.drl.RefMVCount)
			}
		default:
			gotMV, _, err := state.ReadMotionVector(&decMV, o.ref, o.precision)
			if err != nil {
				t.Fatalf("op %d ReadMotionVector: %v", i, err)
			}
			if gotMV != o.mv {
				t.Fatalf("op %d mv=%v want %v (ref=%v prec=%d)", i, gotMV, o.mv, o.ref, o.precision)
			}
		}
	}
}

// TestWriteCompoundInterSymbolsRoundTrip gates the compound inter writer
// surface: block modes cover NEAREST_NEARESTMV..NEW_NEWMV, DRL indices cover
// NEW_NEWMV and all NEAR-containing compound modes, and motion writing covers
// one- and two-vector residual paths through ReadInterMotion.
func TestWriteCompoundInterSymbolsRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(31))

	type op struct {
		kind int // 0 block mode, 1 drl, 2 motion

		modeReq InterModeRequest
		mode    InterModeResult

		drl    DRLRequest
		drlIdx int

		motionReq InterMotionRequest
		motion    InterMotionResult
	}

	modes := []CompoundInterMode{
		CompoundInterModeNearestNearest,
		CompoundInterModeNearNear,
		CompoundInterModeNearestNew,
		CompoundInterModeNewNearest,
		CompoundInterModeNearNew,
		CompoundInterModeNewNear,
		CompoundInterModeGlobalGlobal,
		CompoundInterModeNewNew,
	}
	compoundRefs := []InterReferencesResult{
		{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameLast2}, Compound: true, Unidir: true},
		{Ref: [2]ReferenceFrame{ReferenceFrameBWD, ReferenceFrameAltref}, Compound: true, Unidir: true},
		{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameBWD}, Compound: true},
		{Ref: [2]ReferenceFrame{ReferenceFrameLast2, ReferenceFrameAltref2}, Compound: true},
		{Ref: [2]ReferenceFrame{ReferenceFrameLast3, ReferenceFrameAltref}, Compound: true},
		{Ref: [2]ReferenceFrame{ReferenceFrameGolden, ReferenceFrameBWD}, Compound: true},
	}
	precisions := []MVSubpelPrecision{MVSubpelNone, MVSubpelLow, MVSubpelHigh}

	const n = 8000
	ops := make([]op, 0, n)

	var encModes InterModeCDFs
	var encMV MVCDFs
	if err := encModes.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if err := encMV.InitDefault(); err != nil {
		t.Fatal(err)
	}
	w := entropy.NewWriter(make([]byte, 0, 1<<17))

	for range n {
		mode := modes[rng.Intn(len(modes))]
		modeResult := InterModeResult{Compound: true, CompoundMode: mode}
		var o op
		switch rng.Intn(3) {
		case 0:
			modeCtx := uint16(rng.Intn(6)) | uint16(rng.Intn(2))<<globalMVOffset | uint16(rng.Intn(6))<<refMVOffset
			req := InterModeRequest{Compound: true, ModeContext: modeCtx}
			o = op{kind: 0, modeReq: req, mode: modeResult}
			if err := WriteBlockInterMode(&w, &encModes, req, modeResult); err != nil {
				t.Fatalf("WriteBlockInterMode ctx=%d mode=%d: %v", modeCtx, mode, err)
			}
		case 1:
			count := uint8(1 + rng.Intn(4))
			req := DRLRequest{Compound: true, CompoundMode: mode, RefMVCount: count}
			for i := range req.Contexts {
				req.Contexts[i] = uint8(rng.Intn(DRLModeContexts))
			}
			maxIdx := 0
			switch {
			case req.usesNewMV():
				maxIdx = min(2, int(count)-1)
			case req.usesNearMV():
				maxIdx = max(0, min(2, int(count)-2))
			}
			idx := 0
			if maxIdx > 0 {
				idx = rng.Intn(maxIdx + 1)
			}
			o = op{kind: 1, drl: req, drlIdx: idx}
			if err := WriteDRLIndex(&w, &encModes, req, idx); err != nil {
				t.Fatalf("WriteDRLIndex compound mode=%d count=%d idx=%d: %v", mode, count, idx, err)
			}
		default:
			req := InterMotionRequest{
				References:   compoundRefs[rng.Intn(len(compoundRefs))],
				Mode:         modeResult,
				ReferenceMVs: randomInterMVReferenceSet(rng),
				GlobalMVs:    [2]motion.Vector{randomSmallVector(rng), randomSmallVector(rng)},
				Precision:    precisions[rng.Intn(len(precisions))],
			}
			result := randomInterMotionForRequest(rng, req)
			o = op{kind: 2, motionReq: req, motion: result}
			if err := WriteInterMotion(&w, &encMV, req, result); err != nil {
				t.Fatalf("WriteInterMotion mode=%d precision=%d: %v", mode, req.Precision, err)
			}
		}
		ops = append(ops, o)
	}
	buf, err := w.Finish()
	if err != nil {
		t.Fatal(err)
	}

	var decModes InterModeCDFs
	var decMV MVCDFs
	if err := decModes.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if err := decMV.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset(buf, Job{Offset: 0, Size: uint32(len(buf))}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	for i, o := range ops {
		switch o.kind {
		case 0:
			got, err := state.ReadBlockInterMode(&decModes, o.modeReq)
			if err != nil {
				t.Fatalf("op %d ReadBlockInterMode: %v", i, err)
			}
			if got != o.mode {
				t.Fatalf("op %d mode=%+v want %+v", i, got, o.mode)
			}
		case 1:
			got, err := state.ReadDRLIndex(&decModes, o.drl)
			if err != nil {
				t.Fatalf("op %d ReadDRLIndex: %v", i, err)
			}
			if got != o.drlIdx {
				t.Fatalf("op %d drl=%d want %d (mode=%d count=%d)", i, got, o.drlIdx, o.drl.CompoundMode, o.drl.RefMVCount)
			}
		default:
			got, err := state.ReadInterMotion(&decMV, o.motionReq)
			if err != nil {
				t.Fatalf("op %d ReadInterMotion: %v", i, err)
			}
			if got.Motion != o.motion {
				t.Fatalf("op %d motion=%+v want %+v", i, got.Motion, o.motion)
			}
			components, err := o.motionReq.Mode.CompoundMode.Components()
			if err != nil {
				t.Fatal(err)
			}
			if got.ResidualValid[0] != (components.First == InterModeNewMV) ||
				got.ResidualValid[1] != (components.Second == InterModeNewMV) {
				t.Fatalf("op %d residual valid=%v mode=%d", i, got.ResidualValid, o.motionReq.Mode.CompoundMode)
			}
		}
	}
}

// randomMVDiff returns a component residual representable at the given
// precision: full-pel multiples of 8 without subpel symbols, even residuals
// without the high-precision bit, anything otherwise. Zero is allowed (the
// joint absorbs it).
func randomMVDiff(rng *rand.Rand, precision MVSubpelPrecision) int32 {
	if rng.Intn(4) == 0 {
		return 0
	}
	var diff int32
	switch precision {
	case MVSubpelNone:
		diff = int32(1+rng.Intn(128)) * 8
	case MVSubpelLow:
		diff = int32(1+rng.Intn(512)) * 2
	default:
		diff = int32(1 + rng.Intn(1024))
	}
	if rng.Intn(2) == 0 {
		diff = -diff
	}
	return diff
}

func randomInterMVReferenceSet(rng *rand.Rand) InterMVReferenceSet {
	return InterMVReferenceSet{
		Nearest:  [2]motion.Vector{randomSmallVector(rng), randomSmallVector(rng)},
		Near:     [2]motion.Vector{randomSmallVector(rng), randomSmallVector(rng)},
		Residual: [2]motion.Vector{randomSmallVector(rng), randomSmallVector(rng)},
	}
}

func randomInterMotionForRequest(rng *rand.Rand, req InterMotionRequest) InterMotionResult {
	result := InterMotionResult{
		References: req.References,
		Mode:       req.Mode,
	}
	components, err := req.Mode.CompoundMode.Components()
	if err != nil {
		panic(err)
	}
	result.MV[0] = randomMotionVectorForComponent(rng, req, components.First, 0)
	result.MV[1] = randomMotionVectorForComponent(rng, req, components.Second, 1)
	return result
}

func randomMotionVectorForComponent(rng *rand.Rand, req InterMotionRequest, mode InterMode, index int) motion.Vector {
	switch mode {
	case InterModeNearestMV:
		return req.ReferenceMVs.Nearest[index]
	case InterModeNearMV:
		return req.ReferenceMVs.Near[index]
	case InterModeGlobalMV:
		return req.GlobalMVs[index]
	case InterModeNewMV:
		ref := req.ReferenceMVs.Residual[index]
		return motion.Vector{
			Row: ref.Row + int16(randomMVDiff(rng, req.Precision)),
			Col: ref.Col + int16(randomMVDiff(rng, req.Precision)),
		}
	default:
		panic("invalid inter mode")
	}
}

func randomSmallVector(rng *rand.Rand) motion.Vector {
	return motion.Vector{
		Row: int16(rng.Intn(257) - 128),
		Col: int16(rng.Intn(257) - 128),
	}
}
