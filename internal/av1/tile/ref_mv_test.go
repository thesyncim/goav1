package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

func TestReferenceMVStackDRLContextMatchesLibaomAndDav1d(t *testing.T) {
	tests := []struct {
		name    string
		weights [2]uint16
		want    int
	}{
		{name: "both category", weights: [2]uint16{640, 640}, want: 0},
		{name: "current category", weights: [2]uint16{640, 639}, want: 1},
		{name: "neither category", weights: [2]uint16{639, 1}, want: 2},
		{name: "next category only", weights: [2]uint16{0, 640}, want: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stack := ReferenceMVStack{Count: 2}
			stack.Candidates[0].Weight = tt.weights[0]
			stack.Candidates[1].Weight = tt.weights[1]
			got, err := stack.DRLContext(0)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("ctx=%d want %d", got, tt.want)
			}
		})
	}
}

func TestReferenceMVStackDRLRequestForMode(t *testing.T) {
	stack := refMVTestStack()
	req, err := stack.DRLRequestForMode(InterModeResult{Mode: InterModeNewMV})
	if err != nil {
		t.Fatal(err)
	}
	if req.Mode != InterModeNewMV || req.Compound || req.RefMVCount != stack.Count {
		t.Fatalf("req=%+v", req)
	}
	if req.Contexts != ([3]int{1, 0, 1}) {
		t.Fatalf("contexts=%v want [1 0 1]", req.Contexts)
	}

	req, err = stack.DRLRequestForMode(InterModeResult{Compound: true, CompoundMode: CompoundInterModeNewNew})
	if err != nil {
		t.Fatal(err)
	}
	if !req.Compound || req.CompoundMode != CompoundInterModeNewNew {
		t.Fatalf("compound req=%+v", req)
	}
}

func TestBestSingleReferenceMVsMatchesLibaomPrecision(t *testing.T) {
	stack := refMVTestStack()
	nearest, near, err := stack.BestSingleReferenceMVs(false, false)
	if err != nil {
		t.Fatal(err)
	}
	if nearest != (motion.Vector{Row: 2, Col: -2}) || near != (motion.Vector{Row: 6, Col: 4}) {
		t.Fatalf("low precision nearest=%+v near=%+v", nearest, near)
	}

	nearest, near, err = stack.BestSingleReferenceMVs(true, true)
	if err != nil {
		t.Fatal(err)
	}
	if nearest != (motion.Vector{Row: 0, Col: 0}) || near != (motion.Vector{Row: 8, Col: 8}) {
		t.Fatalf("integer precision nearest=%+v near=%+v", nearest, near)
	}
}

func TestResolveSingleInterMVReferencesMatchesLibaomSelection(t *testing.T) {
	stack := refMVTestStack()
	refs, err := stack.ResolveInterMVReferences(InterModeResult{Mode: InterModeNewMV}, 1, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if refs.Nearest[0] != (motion.Vector{Row: 2, Col: -2}) ||
		refs.Near[0] != (motion.Vector{Row: 6, Col: 4}) ||
		refs.Residual[0] != stack.Candidates[1].This {
		t.Fatalf("newmv refs=%+v", refs)
	}

	refs, err = stack.ResolveInterMVReferences(InterModeResult{Mode: InterModeNearMV}, 2, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if refs.Near[0] != stack.Candidates[3].This || refs.Residual[0] != refs.Nearest[0] {
		t.Fatalf("nearmv refs=%+v", refs)
	}
}

func TestResolveCompoundInterMVReferencesMatchesLibaomSelection(t *testing.T) {
	stack := refMVTestStack()
	refs, err := stack.ResolveInterMVReferences(InterModeResult{
		Compound:     true,
		CompoundMode: CompoundInterModeNewNear,
	}, 1, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if refs.Nearest != ([2]motion.Vector{{Row: 2, Col: -2}, {Row: -2, Col: 2}}) {
		t.Fatalf("nearest=%+v", refs.Nearest)
	}
	if refs.Near != ([2]motion.Vector{{Row: -2, Col: 10}, {Row: 8, Col: -8}}) {
		t.Fatalf("near=%+v", refs.Near)
	}
	if refs.Residual[0] != stack.Candidates[2].This || refs.Residual[1] != refs.Nearest[1] {
		t.Fatalf("residual=%+v", refs.Residual)
	}

	refs, err = stack.ResolveInterMVReferences(InterModeResult{
		Compound:     true,
		CompoundMode: CompoundInterModeNearNew,
	}, 1, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if refs.Residual[0] != refs.Nearest[0] || refs.Residual[1] != stack.Candidates[2].Compound {
		t.Fatalf("near-new residual=%+v", refs.Residual)
	}
}

func TestReferenceMVStackRejectsInvalidInputs(t *testing.T) {
	stack := refMVTestStack()
	if _, err := stack.DRLContext(3); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad drl ctx err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, _, err := (ReferenceMVStack{Count: 1}).BestSingleReferenceMVs(false, false); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("short best err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := (ReferenceMVStack{Count: MaxRefMVStackSize + 1}).DRLContexts(); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad count err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := stack.ResolveInterMVReferences(InterModeResult{Mode: InterModeGlobalMV}, 0, false, false); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("global mode err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := stack.ResolveInterMVReferences(InterModeResult{Compound: true, CompoundMode: CompoundInterModeGlobalGlobal}, 0, false, false); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("global compound err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := stack.ResolveInterMVReferences(InterModeResult{Mode: InterModeNewMV}, -1, false, false); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad ref idx err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := stack.DRLRequestForMode(InterModeResult{Mode: interModeCount}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad drl mode err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := (&BlockModeContext{}).BuildReferenceMVStack(ReferenceMVStackRequest{
		Size:                        BlockSize16x16,
		References:                  InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}},
		TemporalMVSampleUnavailable: true,
	}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("temporal without ref-frame-mvs err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestReferenceMVStackAllocs(t *testing.T) {
	stack := refMVTestStack()
	allocs := testing.AllocsPerRun(1000, func() {
		mode := InterModeResult{Compound: true, CompoundMode: CompoundInterModeNearNew}
		if _, err := stack.DRLRequestForMode(mode); err != nil {
			t.Fatal(err)
		}
		if _, err := stack.ResolveInterMVReferences(mode, 1, false, false); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ref mv stack allocated: %f", allocs)
	}
}

func TestBlockModeContextBuildReferenceMVStackSingle(t *testing.T) {
	var ctx BlockModeContext
	target := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	above := InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeNewMV},
		MV:         [2]motion.Vector{{Row: 3, Col: 5}},
	}
	left := InterMotionResult{
		References: InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameGolden, ReferenceFrameNone}},
		Mode:       InterModeResult{Mode: InterModeNearMV},
		MV:         [2]motion.Vector{{Row: -7, Col: 9}},
	}
	seedAboveMotion(&ctx, 0, BlockSize16x8, above)
	seedLeftMotion(&ctx, 0, BlockSize8x16, left)
	req := ReferenceMVStackRequest{
		Size:        BlockSize16x16,
		References:  target,
		X4:          0,
		Y4:          0,
		HaveTop:     true,
		HaveLeft:    true,
		GlobalMVs:   [2]motion.Vector{{Row: 1, Col: 1}},
		RefSignBias: [referenceFrameCount]bool{ReferenceFrameGolden: true},
	}
	result, err := ctx.BuildReferenceMVStack(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Stack.Count != 2 || result.NearestCount != 1 {
		t.Fatalf("counts stack=%d nearest=%d", result.Stack.Count, result.NearestCount)
	}
	if result.RowMatches != 1 || result.ColumnMatches != 0 || result.NewMVMatches != 1 {
		t.Fatalf("matches row=%d col=%d new=%d", result.RowMatches, result.ColumnMatches, result.NewMVMatches)
	}
	if result.ModeContext != uint16(2|(3<<refMVOffset)) {
		t.Fatalf("mode ctx=%d want %d", result.ModeContext, 2|(3<<refMVOffset))
	}
	if result.Stack.Candidates[0] != (ReferenceMVCandidate{This: above.MV[0], Weight: RefMVCategoryLevel + 8}) {
		t.Fatalf("nearest candidate=%+v", result.Stack.Candidates[0])
	}
	if result.Stack.Candidates[1] != (ReferenceMVCandidate{This: motion.Vector{Row: 7, Col: -9}, Weight: 2}) {
		t.Fatalf("fallback candidate=%+v", result.Stack.Candidates[1])
	}
}

func TestBuildReferenceMVStackRefFrameMVSUnavailableSetsGlobalContext(t *testing.T) {
	var ctx BlockModeContext
	target := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	req := ReferenceMVStackRequest{
		Size:                        BlockSize16x16,
		References:                  target,
		GlobalMVs:                   [2]motion.Vector{{Row: 1, Col: 1}},
		UseRefFrameMVS:              true,
		TemporalMVSampleUnavailable: true,
	}
	result, err := ctx.BuildReferenceMVStack(req)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := result.ModeContext, uint16((1 << globalMVOffset)); got != want {
		t.Fatalf("mode ctx=%d want %d", got, want)
	}
	if result.Stack.Count != 0 || !result.Stack.SingleRefValid || result.Stack.SingleRefMVs[0] != req.GlobalMVs[0] {
		t.Fatalf("stack=%+v", result.Stack)
	}
}

func TestBuildReferenceMVStackUsesTemporalRefFrameMVS(t *testing.T) {
	var ctx BlockModeContext
	field := newTemporalMotionFieldForTest(t, 16, 16)
	field.Entries[0] = TemporalMotionEntry{
		MV:             motion.Vector{Row: 64, Col: 32},
		RefFrameOffset: 4,
		Valid:          true,
	}
	req := ReferenceMVStackRequest{
		Size:       BlockSize16x16,
		References: InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}},

		MICol:            0,
		MIRow:            0,
		TileMIColEnd:     16,
		TileMIRowEnd:     16,
		TemporalMVs:      field,
		OrderHintBits:    5,
		CurrentOrderHint: 8,
		ReferenceOrderHints: [referenceFrameCount]uint32{
			ReferenceFrameLast: 4,
		},
		GlobalMVs:      [2]motion.Vector{{Row: 0, Col: 0}},
		UseRefFrameMVS: true,
	}
	result, err := ctx.BuildReferenceMVStack(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.ModeContext&(1<<globalMVOffset) == 0 {
		t.Fatalf("mode ctx=%d missing temporal globalmv signal", result.ModeContext)
	}
	if result.Stack.Count != 1 || !result.Stack.SingleRefValid {
		t.Fatalf("stack=%+v", result.Stack)
	}
	if got := result.Stack.Candidates[0]; got != (ReferenceMVCandidate{This: motion.Vector{Row: 64, Col: 32}, Weight: 2}) {
		t.Fatalf("temporal candidate=%+v", got)
	}
	nearest, near, err := result.Stack.BestSingleReferenceMVs(true, false)
	if err != nil {
		t.Fatal(err)
	}
	if nearest != (motion.Vector{Row: 64, Col: 32}) || near != (motion.Vector{}) {
		t.Fatalf("nearest=%+v near=%+v", nearest, near)
	}
}

func TestBuildReferenceMVStackTemporalCompoundProjectsBothRefs(t *testing.T) {
	var ctx BlockModeContext
	field := newTemporalMotionFieldForTest(t, 16, 16)
	field.Entries[0] = TemporalMotionEntry{
		MV:             motion.Vector{Row: 64, Col: 32},
		RefFrameOffset: 4,
		Valid:          true,
	}
	req := ReferenceMVStackRequest{
		Size: BlockSize16x16,
		References: InterReferencesResult{
			Ref:      [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameBWD},
			Compound: true,
		},

		MICol:            0,
		MIRow:            0,
		TileMIColEnd:     16,
		TileMIRowEnd:     16,
		TemporalMVs:      field,
		OrderHintBits:    5,
		CurrentOrderHint: 8,
		ReferenceOrderHints: [referenceFrameCount]uint32{
			ReferenceFrameLast: 4,
			ReferenceFrameBWD:  12,
		},
		GlobalMVs:      [2]motion.Vector{{Row: 0, Col: 0}, {Row: 0, Col: 0}},
		UseRefFrameMVS: true,
	}
	result, err := ctx.BuildReferenceMVStack(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Stack.Count != 2 {
		t.Fatalf("stack=%+v", result.Stack)
	}
	if got := result.Stack.Candidates[0]; got != (ReferenceMVCandidate{
		This:     motion.Vector{Row: 64, Col: 32},
		Compound: motion.Vector{Row: -64, Col: -32},
		Weight:   2,
	}) {
		t.Fatalf("temporal compound candidate=%+v", got)
	}
}

func TestBuildReferenceMVStackTemporalUnavailableSetsGlobalContext(t *testing.T) {
	var ctx BlockModeContext
	req := ReferenceMVStackRequest{
		Size:             BlockSize16x16,
		References:       InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}},
		TileMIColEnd:     16,
		TileMIRowEnd:     16,
		TemporalMVs:      newTemporalMotionFieldForTest(t, 16, 16),
		UseRefFrameMVS:   true,
		OrderHintBits:    5,
		CurrentOrderHint: 8,
		ReferenceOrderHints: [referenceFrameCount]uint32{
			ReferenceFrameLast: 4,
		},
	}
	result, err := ctx.BuildReferenceMVStack(req)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := result.ModeContext, uint16(1<<globalMVOffset); got != want {
		t.Fatalf("mode ctx=%d want %d", got, want)
	}
	if result.Stack.Count != 0 {
		t.Fatalf("stack=%+v want no temporal candidates", result.Stack)
	}
}

func TestBlockModeContextBuildReferenceMVStackCompound(t *testing.T) {
	var ctx BlockModeContext
	target := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameBWD}, Compound: true}
	direct := InterMotionResult{
		References: target,
		Mode:       InterModeResult{Compound: true, CompoundMode: CompoundInterModeNewNew},
		MV:         [2]motion.Vector{{Row: 4, Col: 6}, {Row: -2, Col: -8}},
	}
	fallback := InterMotionResult{
		References: InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameAltref}, Compound: true},
		Mode:       InterModeResult{Compound: true, CompoundMode: CompoundInterModeNearNear},
		MV:         [2]motion.Vector{{Row: 10, Col: 12}, {Row: 14, Col: 16}},
	}
	seedAboveMotion(&ctx, 0, BlockSize16x16, direct)
	seedLeftMotion(&ctx, 0, BlockSize16x16, fallback)
	req := ReferenceMVStackRequest{
		Size:        BlockSize16x16,
		References:  target,
		X4:          0,
		Y4:          0,
		HaveTop:     true,
		HaveLeft:    true,
		GlobalMVs:   [2]motion.Vector{{Row: 1, Col: 1}, {Row: 2, Col: 2}},
		RefSignBias: [referenceFrameCount]bool{ReferenceFrameAltref: true},
	}
	result, err := ctx.BuildReferenceMVStack(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Stack.Count != 2 || result.NearestCount != 1 {
		t.Fatalf("counts stack=%d nearest=%d", result.Stack.Count, result.NearestCount)
	}
	if result.Stack.Candidates[0] != (ReferenceMVCandidate{This: direct.MV[0], Compound: direct.MV[1], Weight: RefMVCategoryLevel + 8}) {
		t.Fatalf("direct candidate=%+v", result.Stack.Candidates[0])
	}
	if result.Stack.Candidates[1] != (ReferenceMVCandidate{This: fallback.MV[0], Compound: direct.MV[0], Weight: 2}) {
		t.Fatalf("compound fallback=%+v", result.Stack.Candidates[1])
	}
}

func TestBlockModeContextMarkInterClearsStaleMotion(t *testing.T) {
	var ctx BlockModeContext
	motionResult := InterMotionResult{
		References: InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}},
		Mode:       InterModeResult{Mode: InterModeNewMV},
		MV:         [2]motion.Vector{{Row: 3, Col: 5}},
	}
	if err := ctx.MarkInterMotion(BlockSize8x8, 0, 0, motionResult); err != nil {
		t.Fatal(err)
	}
	if ctx.AboveMotionValid[0] == 0 || ctx.LeftMotionValid[0] == 0 {
		t.Fatal("motion was not marked valid")
	}
	if err := ctx.MarkInter(BlockSize8x8, 0, 0, motionResult.References); err != nil {
		t.Fatal(err)
	}
	if ctx.AboveMotionValid[0] != 0 || ctx.LeftMotionValid[0] != 0 {
		t.Fatal("inter reference mark left stale motion valid")
	}
	if err := ctx.MarkIntra(BlockSize8x8, 0, 0, true, IntraModeDC); err != nil {
		t.Fatal(err)
	}
	if ctx.AboveMotionValid[0] != 0 || ctx.LeftMotionValid[0] != 0 {
		t.Fatal("intra mark left stale motion valid")
	}
}

func TestBuildReferenceMVStackSingleGlobalFallbackDoesNotInflateCount(t *testing.T) {
	var ctx BlockModeContext
	global := motion.Vector{Row: 11, Col: -13}
	result, err := ctx.BuildReferenceMVStack(ReferenceMVStackRequest{
		Size:       BlockSize16x16,
		References: InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}},
		GlobalMVs:  [2]motion.Vector{global},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Stack.Count != 0 || !result.Stack.SingleRefValid {
		t.Fatalf("stack count=%d single valid=%v", result.Stack.Count, result.Stack.SingleRefValid)
	}
	nearest, near, err := result.Stack.BestSingleReferenceMVs(true, false)
	if err != nil {
		t.Fatal(err)
	}
	if nearest != global || near != global {
		t.Fatalf("nearest=%+v near=%+v want global %+v", nearest, near, global)
	}
	req, err := result.Stack.DRLRequestForMode(InterModeResult{Mode: InterModeNewMV})
	if err != nil {
		t.Fatal(err)
	}
	if req.RefMVCount != 0 {
		t.Fatalf("drl ref count=%d want 0", req.RefMVCount)
	}
}

func FuzzReferenceMVStack(f *testing.F) {
	f.Add(uint8(4), uint16(640), uint16(100), uint16(700), uint16(1), uint8(InterModeNewMV), false, uint8(1), false, false)
	f.Add(uint8(2), uint16(639), uint16(1), uint16(0), uint16(0), uint8(InterModeNearMV), false, uint8(0), true, false)
	f.Add(uint8(4), uint16(640), uint16(100), uint16(700), uint16(1), uint8(CompoundInterModeNearNew), true, uint8(1), false, true)

	f.Fuzz(func(t *testing.T, rawCount uint8, w0 uint16, w1 uint16, w2 uint16, w3 uint16, rawMode uint8, compound bool, rawRefIdx uint8, allowHighPrecision bool, forceInteger bool) {
		stack := ReferenceMVStack{Count: int(rawCount % (MaxRefMVStackSize + 1))}
		weights := [4]uint16{w0, w1, w2, w3}
		for i := range stack.Candidates {
			stack.Candidates[i] = ReferenceMVCandidate{
				This:     motion.Vector{Row: int32(int16(weights[i%len(weights)])), Col: -int32(int16(weights[(i+1)%len(weights)]))},
				Compound: motion.Vector{Row: -int32(int16(weights[(i+2)%len(weights)])), Col: int32(int16(weights[(i+3)%len(weights)]))},
				Weight:   weights[i%len(weights)],
			}
		}

		mode := InterModeResult{Mode: InterMode(rawMode % uint8(interModeCount))}
		if compound {
			mode = InterModeResult{Compound: true, CompoundMode: CompoundInterMode(rawMode % uint8(compoundInterModeCount))}
		}
		req, err := stack.DRLRequestForMode(mode)
		if err != nil {
			if errors.Is(err, ErrInvalidDecodeState) {
				return
			}
			t.Fatalf("DRLRequestForMode err=%v", err)
		}
		if req.RefMVCount != stack.Count {
			t.Fatalf("ref mv count=%d want %d", req.RefMVCount, stack.Count)
		}

		refIdx := int(rawRefIdx % UsableRefMVStackSize)
		if _, err := stack.ResolveInterMVReferences(mode, refIdx, allowHighPrecision, forceInteger); err != nil &&
			!errors.Is(err, ErrInvalidDecodeState) {
			t.Fatalf("ResolveInterMVReferences err=%v", err)
		}
	})
}

func FuzzBuildReferenceMVStack(f *testing.F) {
	f.Add(uint8(BlockSize16x16), uint8(ReferenceFrameLast), uint8(ReferenceFrameBWD), true, uint8(CompoundInterModeNewNew), int16(3), int16(5), int16(-7), int16(9), true, true)
	f.Add(uint8(BlockSize8x8), uint8(ReferenceFrameGolden), uint8(ReferenceFrameAltref), false, uint8(InterModeNewMV), int16(11), int16(-13), int16(17), int16(19), true, false)
	f.Add(uint8(BlockSize4x4), uint8(ReferenceFrameLast2), uint8(ReferenceFrameAltref2), true, uint8(CompoundInterModeNearNew), int16(-3), int16(-5), int16(7), int16(-9), false, true)

	f.Fuzz(func(t *testing.T, rawSize uint8, rawRef0 uint8, rawRef1 uint8, compound bool, rawMode uint8, aboveRow int16, aboveCol int16, leftRow int16, leftCol int16, haveTop bool, haveLeft bool) {
		size := BlockSize(rawSize % uint8(blockSizeCount))
		ref0 := ReferenceFrame(rawRef0 % uint8(referenceFrameCount))
		ref1 := ReferenceFrameNone
		if compound {
			ref1 = ReferenceFrame(rawRef1 % uint8(referenceFrameCount))
		}
		refs := InterReferencesResult{Ref: [2]ReferenceFrame{ref0, ref1}, Compound: compound}
		mode := InterModeResult{Mode: InterMode(rawMode % uint8(interModeCount))}
		if compound {
			mode = InterModeResult{Compound: true, CompoundMode: CompoundInterMode(rawMode % uint8(compoundInterModeCount))}
		}

		var ctx BlockModeContext
		above := InterMotionResult{
			References: refs,
			Mode:       mode,
			MV:         [2]motion.Vector{{Row: int32(aboveRow), Col: int32(aboveCol)}, {Row: -int32(aboveRow), Col: -int32(aboveCol)}},
		}
		left := InterMotionResult{
			References: InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameGolden, ReferenceFrameAltref}, Compound: true},
			Mode:       InterModeResult{Compound: true, CompoundMode: CompoundInterModeNearNear},
			MV:         [2]motion.Vector{{Row: int32(leftRow), Col: int32(leftCol)}, {Row: -int32(leftCol), Col: int32(leftRow)}},
		}
		seedAboveMotion(&ctx, 0, size, above)
		seedLeftMotion(&ctx, 0, size, left)

		result, err := ctx.BuildReferenceMVStack(ReferenceMVStackRequest{
			Size:        size,
			References:  refs,
			HaveTop:     haveTop,
			HaveLeft:    haveLeft,
			GlobalMVs:   [2]motion.Vector{{Row: 1, Col: 2}, {Row: -1, Col: -2}},
			RefSignBias: [referenceFrameCount]bool{ReferenceFrameGolden: true, ReferenceFrameAltref: true},
		})
		if err != nil {
			if errors.Is(err, ErrInvalidDecodeState) {
				return
			}
			t.Fatalf("BuildReferenceMVStack err=%v", err)
		}
		if result.Stack.Count < 0 || result.Stack.Count > MaxRefMVStackSize {
			t.Fatalf("bad stack count=%d", result.Stack.Count)
		}
		if _, err := AnalyzeInterModeContext(result.ModeContext, compound); err != nil {
			t.Fatalf("bad mode context=%d compound=%v err=%v", result.ModeContext, compound, err)
		}
		if !compound {
			if _, _, err := result.Stack.BestSingleReferenceMVs(false, false); err != nil {
				t.Fatalf("single ref mvs err=%v", err)
			}
		}
	})
}

func refMVTestStack() ReferenceMVStack {
	return ReferenceMVStack{
		Count: 4,
		Candidates: [MaxRefMVStackSize]ReferenceMVCandidate{
			{This: motion.Vector{Row: 3, Col: -3}, Compound: motion.Vector{Row: -3, Col: 3}, Weight: 640},
			{This: motion.Vector{Row: 7, Col: 5}, Compound: motion.Vector{Row: -5, Col: 7}, Weight: 100},
			{This: motion.Vector{Row: -3, Col: 11}, Compound: motion.Vector{Row: 9, Col: -9}, Weight: 700},
			{This: motion.Vector{Row: 13, Col: -7}, Compound: motion.Vector{Row: -13, Col: 15}, Weight: 1},
		},
	}
}

func seedAboveMotion(ctx *BlockModeContext, x4 int, size BlockSize, result InterMotionResult) {
	dims, _ := size.Dimensions()
	for i := 0; i < int(dims.W4); i++ {
		slot := x4 + i
		ctx.AboveIntra[slot] = 0
		ctx.AboveRef[0][slot] = result.References.Ref[0]
		ctx.AboveRef[1][slot] = result.References.Ref[1]
		ctx.AboveCompound[slot] = boolByte(result.References.Compound)
		ctx.AboveInterMotion[slot] = result
		ctx.AboveMotionValid[slot] = 1
		ctx.AboveBlockSize[slot] = size
	}
}

func seedLeftMotion(ctx *BlockModeContext, y4 int, size BlockSize, result InterMotionResult) {
	dims, _ := size.Dimensions()
	for i := 0; i < int(dims.H4); i++ {
		slot := y4 + i
		ctx.LeftIntra[slot] = 0
		ctx.LeftRef[0][slot] = result.References.Ref[0]
		ctx.LeftRef[1][slot] = result.References.Ref[1]
		ctx.LeftCompound[slot] = boolByte(result.References.Compound)
		ctx.LeftInterMotion[slot] = result
		ctx.LeftMotionValid[slot] = 1
		ctx.LeftBlockSize[slot] = size
	}
}
