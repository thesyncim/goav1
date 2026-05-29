package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
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

// TestCDFUpdateBlock64x60LibaomWeights pins the libaom-extracted ref_mv weights
// at frame 1 mi=(64,60) so that future fixes to the topright cross-SB-diagonal
// path can be validated against this regression. The cdf_update strict-MD5 test
// vector diverges here because goav1's BlockModeContext (with MaxBlockModeSlots=
// 32 and SB128x128 sbSizeMIB=32) cannot reach the cell at (mi_row-1,
// mi_col+W4)=(63, 64) — it lives in the SB-above-and-right, whose bottom row is
// held by carrier.Above[col_idx+1] but is never copied into the per-SB
// AboveInterMotion/SBTopInterMotion arrays (extendAboveContextFromRightCarrier
// writes slots [sbSizeMIB..sbSizeMIB+sbSizeMIB), which is fully out of range
// when sbSizeMIB equals MaxBlockModeSlots). libaom's frame-wide mi grid covers
// the cell directly via xd->mi[-stride + xd->width]. Without that candidate
// goav1's nearest_refmv_count = 2 (vs libaom 3) and the post-sort weights at
// idx=1,2 = [656, 20] (vs libaom [664, 656]), flipping av1_drl_ctx() at idx=1
// from 0 to 1 for the NEAR_NEWMV-style decode of the NEARMV iter. The fix
// requires either adding an SB-above-right snapshot (mode.go) or extending
// MaxBlockModeSlots past sbSizeMIB so extendAboveContextFromRightCarrier can
// load slots [sbSizeMIB..2*sbSizeMIB) — neither of which fits inside ref_mv.go.
func TestCDFUpdateBlock64x60LibaomWeights(t *testing.T) {
	// Libaom-extracted state for cdf_update.ivf frame 1 mi_row=64 mi_col=60
	// BLOCK_16X32 mode=NEARMV ref_frame_type=1 (LAST_FRAME single) idx=1.
	// Capture rationale: aomdec instrumented to dump xd->weight[ref_frame_type]
	// just before read_drl_idx's aom_read_symbol(drl_cdf[drl_ctx], 2). Reproduce
	// via the patch in /tmp/libaom-v3.13.1/av1/decoder/decodemv.c (or replay
	// using the cdfupdate.ivf vector and any libaom v3.13.1 build).
	const refFrameType = 1
	const cnt = 4
	wts := [8]uint16{672, 664, 656, 8, 4, 4, 8, 4}
	_ = refFrameType
	stack := ReferenceMVStack{Count: cnt}
	for i := 0; i < cnt; i++ {
		stack.Candidates[i].Weight = wts[i]
	}

	// DRL context as libaom computes it at the NEARMV iter idx=1. libaom only
	// reads the idx=1 drl bit for this block (it returns once drl_idx=0). The
	// libaom-reported drl_ctx is 0 because weights[1]=664 and weights[2]=656
	// both meet/exceed RefMVCategoryLevel.
	got, err := stack.DRLContext(1)
	if err != nil {
		t.Fatalf("DRLContext(1) err=%v", err)
	}
	if got != 0 {
		t.Fatalf("DRLContext(1) = %d, want 0 (libaom-extracted)", got)
	}

	// Sanity: the bug observed in the cdf_update strict-MD5 run is that
	// goav1 produces wts=[672, 656, 20, 8, ...] (slot 2 falls below
	// RefMVCategoryLevel because the topright cell at mi_col+W4=64 cannot be
	// reached from BlockModeContext for SB128x128). That state computes
	// DRLContext(1) = 1, which selects drl_cdf[1] instead of drl_cdf[0] at
	// the NEARMV iter idx=1 -> the entropy stream desyncs here (seq=229715
	// in goav1_trace_rng).
	bugStack := ReferenceMVStack{Count: cnt}
	bugStack.Candidates[0].Weight = 672
	bugStack.Candidates[1].Weight = 656
	bugStack.Candidates[2].Weight = 20
	bugStack.Candidates[3].Weight = 8
	bugGot, bugErr := bugStack.DRLContext(1)
	if bugErr != nil {
		t.Fatal(bugErr)
	}
	if bugGot == 0 {
		t.Fatalf("bug-state DRLContext(1) = 0, expected != 0 (the divergence)")
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

func TestResolveCompoundNewNewDoesNotRequireNearCandidate(t *testing.T) {
	stack := refMVTestStack()
	stack.Count = 2
	refs, err := stack.ResolveInterMVReferences(InterModeResult{
		Compound:     true,
		CompoundMode: CompoundInterModeNewNew,
	}, 1, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if refs.Nearest != ([2]motion.Vector{{Row: 2, Col: -2}, {Row: -2, Col: 2}}) {
		t.Fatalf("nearest=%+v", refs.Nearest)
	}
	if refs.Near != refs.Nearest {
		t.Fatalf("near=%+v want nearest", refs.Near)
	}
	if refs.Residual != ([2]motion.Vector{stack.Candidates[1].This, stack.Candidates[1].Compound}) {
		t.Fatalf("residual=%+v", refs.Residual)
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
	if result.Stack.Candidates[0] != (ReferenceMVCandidate{This: direct.MV[0], Compound: direct.MV[1], Weight: RefMVCategoryLevel + 16}) {
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
	if err := ctx.MarkInterMotion(BlockSize8x8, 0, 0, motionResult, true); err != nil {
		t.Fatal(err)
	}
	if ctx.AboveMotionValid[0] == 0 || ctx.LeftMotionValid[0] == 0 || ctx.GridMotionValid[0][0] == 0 {
		t.Fatal("motion was not marked valid")
	}
	if err := ctx.MarkInter(BlockSize8x8, 0, 0, motionResult.References, true); err != nil {
		t.Fatal(err)
	}
	if ctx.AboveMotionValid[0] != 0 || ctx.LeftMotionValid[0] != 0 {
		t.Fatal("inter reference mark left stale motion valid")
	}
	if ctx.GridMotionValid[0][0] != 0 {
		t.Fatal("inter reference mark left stale grid motion valid")
	}
	if err := ctx.MarkInterMotion(BlockSize8x8, 0, 0, motionResult, true); err != nil {
		t.Fatal(err)
	}
	if err := ctx.MarkIntra(BlockSize8x8, 0, 0, true, IntraModeDC); err != nil {
		t.Fatal(err)
	}
	if ctx.AboveMotionValid[0] != 0 || ctx.LeftMotionValid[0] != 0 {
		t.Fatal("intra mark left stale motion valid")
	}
	if ctx.GridMotionValid[0][0] != 0 {
		t.Fatal("intra mark left stale grid motion valid")
	}
}

func TestBlockModeContextMarkInterMotionFillsGrid(t *testing.T) {
	var ctx BlockModeContext
	motionResult := InterMotionResult{
		References: InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}},
		Mode:       InterModeResult{Mode: InterModeNearMV},
		MV:         [2]motion.Vector{{Row: 0, Col: -7}},
	}
	if err := ctx.MarkInterMotion(BlockSize8x8, 4, 6, motionResult, true); err != nil {
		t.Fatal(err)
	}
	for y := 6; y < 8; y++ {
		for x := 4; x < 6; x++ {
			if ctx.GridMotionValid[y][x] == 0 {
				t.Fatalf("grid valid[%d][%d]=0", y, x)
			}
			if ctx.GridInterMotion[y][x] != motionResult || ctx.GridBlockSize[y][x] != BlockSize8x8 {
				t.Fatalf("grid[%d][%d] motion=%+v size=%v", y, x, ctx.GridInterMotion[y][x], ctx.GridBlockSize[y][x])
			}
		}
	}
	if ctx.GridMotionValid[5][4] != 0 || ctx.GridMotionValid[6][3] != 0 {
		t.Fatal("grid fill leaked outside the block")
	}
}

func TestBuildReferenceMVStackUsesOuterGridCandidateForNearMV(t *testing.T) {
	var ctx BlockModeContext
	target := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	left := InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeNearMV},
		MV:         [2]motion.Vector{{Row: 0, Col: -6}},
	}
	outer := InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeNearMV},
		MV:         [2]motion.Vector{{Row: 0, Col: -7}},
	}
	seedLeftMotion(&ctx, 0, BlockSize8x8, left)
	dims, _ := BlockSize8x8.Dimensions()
	ctx.markGridInterMotion(BlockSize8x8, 0, 0, outer, dims)

	result, err := ctx.BuildReferenceMVStack(ReferenceMVStackRequest{
		Size:       BlockSize8x8,
		References: target,
		X4:         4,
		Y4:         0,
		MICol:      4,
		MIRow:      0,
		HaveLeft:   true,
		GlobalMVs:  [2]motion.Vector{{Row: -1, Col: -8}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Stack.Count != 2 || result.NearestCount != 1 {
		t.Fatalf("counts stack=%d nearest=%d stack=%+v", result.Stack.Count, result.NearestCount, result.Stack)
	}
	nearest, near, err := result.Stack.BestSingleReferenceMVs(true, false)
	if err != nil {
		t.Fatal(err)
	}
	if nearest != left.MV[0] || near != outer.MV[0] {
		t.Fatalf("nearest=%+v near=%+v want %+v %+v", nearest, near, left.MV[0], outer.MV[0])
	}
	if result.Stack.Candidates[1] != (ReferenceMVCandidate{This: outer.MV[0], Weight: 4}) {
		t.Fatalf("outer candidate=%+v", result.Stack.Candidates[1])
	}
}

// TestBuildReferenceMVStackIntraNearestAdvancesProcessedRows mirrors the
// cdf_update frame 1 block (mi_row=20, mi_col=0) ref-MV stack search:
//
//   - The block is BLOCK_8X16 (W4=2, H4=4) at SB-local Y4=20, X4=0.
//   - The nearest above row holds an intra BLOCK_8X16 neighbor; the left
//     edge is unavailable (frame boundary).
//   - A NewMV top-right candidate occupies the diagonal slot.
//   - The outer row at row_offset=-5 holds the BLOCK_8X32 NewMV candidate
//     that completes libaom's stack.
//
// libaom's scan_row_mbmi sets processed_rows from the neighbor's block size
// regardless of whether the neighbor is inter or intra, so row_offset=-3 is
// skipped and row_offset=-5 supplies the second stack entry. The earlier
// goav1 path gated the processed_rows update on AboveMotionValid which kept
// processedRows=0, scanned row_offset=-3, advanced processedRows on a
// duplicate sample, and then suppressed the row_offset=-5 add. The result
// was stack.Count=1, no DRL bit, and NEWMV/NEARMV mode drift two blocks
// downstream.
func TestBuildReferenceMVStackIntraNearestAdvancesProcessedRows(t *testing.T) {
	var ctx BlockModeContext
	target := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	topRight := InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeNewMV},
		MV:         [2]motion.Vector{{Row: -5, Col: -69}},
	}
	outer := InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeNewMV},
		MV:         [2]motion.Vector{{Row: -10, Col: -69}},
	}

	// Above row: an intra BLOCK_8X16 neighbor whose AboveBlockSize must
	// still be populated so libaom's processed_rows stride is honoured.
	if err := ctx.MarkIntra(BlockSize8x16, 0, 16, true, IntraModeDC); err != nil {
		t.Fatal(err)
	}
	// Top-right diagonal slot at (Y4=19, X4=2) lives in the prior block.
	dims8x16, _ := BlockSize8x16.Dimensions()
	ctx.markGridInterMotion(BlockSize8x16, 2, 16, topRight, dims8x16)
	// Outer row=-5 candidate at SB-local (Y4=15, X4=1) corresponds to a
	// BLOCK_8X32 inter neighbor decoded earlier in the SB.
	dims8x32, _ := BlockSize8x32.Dimensions()
	ctx.markGridInterMotion(BlockSize8x32, 0, 8, outer, dims8x32)

	result, err := ctx.BuildReferenceMVStack(ReferenceMVStackRequest{
		Size:           BlockSize8x16,
		References:     target,
		X4:             0,
		Y4:             20,
		MICol:          0,
		MIRow:          20,
		HaveTop:        true,
		HaveLeft:       false,
		HaveTopRight:   true,
		TileMIColEnd:   64,
		TileMIRowEnd:   64,
		TileMIColStart: 0,
		TileMIRowStart: 0,
		GlobalMVs:      [2]motion.Vector{{Row: -4, Col: -70}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Stack.Count != 2 {
		t.Fatalf("stack count=%d want 2 stack=%+v", result.Stack.Count, result.Stack)
	}
	if result.Stack.Candidates[0].This != topRight.MV[0] {
		t.Fatalf("nearest candidate=%+v want %+v", result.Stack.Candidates[0], topRight.MV[0])
	}
	if result.Stack.Candidates[1].This != outer.MV[0] {
		t.Fatalf("outer candidate=%+v want %+v", result.Stack.Candidates[1], outer.MV[0])
	}
}

func TestBuildReferenceMVStackUsesTopRightBeforeNearestCutoff(t *testing.T) {
	var ctx BlockModeContext
	target := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	above := InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeNearMV},
		MV:         [2]motion.Vector{{Row: 0, Col: -7}},
	}
	left := InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeNearMV},
		MV:         [2]motion.Vector{{Row: -1, Col: -8}},
	}
	topRight := InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeNearMV},
		MV:         [2]motion.Vector{{Row: -1, Col: -7}},
	}
	seedAboveMotion(&ctx, 6, BlockSize8x8, above)
	seedLeftMotion(&ctx, 8, BlockSize8x4, left)
	dims, _ := BlockSize8x8.Dimensions()
	ctx.markGridInterMotion(BlockSize8x8, 8, 6, topRight, dims)

	result, err := ctx.BuildReferenceMVStack(ReferenceMVStackRequest{
		Size:           BlockSize8x4,
		References:     target,
		X4:             6,
		Y4:             8,
		MICol:          6,
		MIRow:          8,
		HaveTop:        true,
		HaveLeft:       true,
		HaveTopRight:   true,
		TileMIColEnd:   64,
		TileMIRowEnd:   64,
		TileMIColStart: 0,
		TileMIRowStart: 0,
		GlobalMVs:      [2]motion.Vector{{Row: 0, Col: 0}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.NearestCount != 3 {
		t.Fatalf("nearest count=%d stack=%+v", result.NearestCount, result.Stack)
	}
	nearest, near, err := result.Stack.BestSingleReferenceMVs(true, false)
	if err != nil {
		t.Fatal(err)
	}
	if nearest != above.MV[0] || near != topRight.MV[0] {
		t.Fatalf("nearest=%+v near=%+v want %+v %+v", nearest, near, above.MV[0], topRight.MV[0])
	}
	if result.Stack.Candidates[2].This != left.MV[0] {
		t.Fatalf("left candidate should sort after top-right: stack=%+v", result.Stack)
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

// TestBuildReferenceMVStackGlobalMVSubstitution verifies that a GLOBALMV
// neighbor whose corresponding warp model is non-translational contributes the
// gm_mv evaluated at the CURRENT decode block's position to the ref-MV stack
// instead of the candidate's own stored MV. This mirrors libaom's
// is_global_mv_block() substitution inside add_ref_mv_candidate and is the
// missing link that caused the cdf_update frame 1 block (12,4) DRL bit to
// diverge: before this fix the left BLOCK_16X16 GLOBALMV neighbor and the
// above BLOCK_16X8 NEARESTMV neighbor both contributed the candidate's stored
// MV (-9,-67), so the outer_blk[-1,-1] (-10,-67) candidate that libaom keeps
// distinct deduped against an entry that should have been (-9,-66) — leaving
// goav1's stack with 3 entries instead of 4 and short-circuiting the NEARMV
// DRL loop one bit early.
func TestBuildReferenceMVStackGlobalMVSubstitution(t *testing.T) {
	var ctx BlockModeContext
	target := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	candidateMV := motion.Vector{Row: -9, Col: -67}
	// Seed an above neighbor whose stored prediction mode is GLOBALMV. libaom
	// substitutes gm_mv_candidates[0] for this neighbor in the stack.
	seedAboveMotion(&ctx, 4, BlockSize16x8, InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeGlobalMV},
		MV:         [2]motion.Vector{candidateMV},
	})
	// Seed a NEARESTMV left neighbor that contributes its own stored MV
	// unchanged so the substitution effect is observable in isolation.
	leftMV := motion.Vector{Row: -11, Col: -66}
	seedLeftMotion(&ctx, 4, BlockSize16x16, InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeNearestMV},
		MV:         [2]motion.Vector{leftMV},
	})
	gmMV := motion.Vector{Row: -9, Col: -66}
	req := ReferenceMVStackRequest{
		Size:             BlockSize8x8,
		References:       target,
		X4:               4,
		Y4:               4,
		HaveTop:          true,
		HaveLeft:         true,
		GlobalMVs:        [2]motion.Vector{gmMV},
		GlobalMotionType: [2]parser.GlobalMotionType{parser.GlobalMotionRotZoom},
	}
	result, err := ctx.BuildReferenceMVStack(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Stack.Count != 2 {
		t.Fatalf("stack count=%d want 2", result.Stack.Count)
	}
	// libaom's stack has the above neighbor at slot 0 (added first, then the
	// REF_CAT_LEVEL boost) carrying the substituted gm_mv, not the candidate
	// stored MV. Sorting may or may not reorder; both orderings are valid as
	// long as one slot is the gm_mv and the other is the left neighbor MV.
	gotGM := false
	gotLeft := false
	for i := 0; i < result.Stack.Count; i++ {
		switch result.Stack.Candidates[i].This {
		case gmMV:
			gotGM = true
		case leftMV:
			gotLeft = true
		case candidateMV:
			t.Fatalf("stack[%d] should not carry the candidate's stored MV=%+v when gm substitution applies; got=%+v", i, candidateMV, result.Stack.Candidates[i])
		}
	}
	if !gotGM {
		t.Fatalf("stack missing substituted gm_mv=%+v; got=%+v", gmMV, result.Stack.Candidates[:result.Stack.Count])
	}
	if !gotLeft {
		t.Fatalf("stack missing left neighbor mv=%+v; got=%+v", leftMV, result.Stack.Candidates[:result.Stack.Count])
	}
}

// TestBuildReferenceMVStackTranslationOnlySkipsGlobalMVSubstitution checks that
// a GLOBALMV neighbor whose warp model is identity or translation-only does
// NOT trigger the gm_mv substitution; libaom restricts the substitution to
// non-translational warps via the `type > TRANSLATION` guard.
func TestBuildReferenceMVStackTranslationOnlySkipsGlobalMVSubstitution(t *testing.T) {
	var ctx BlockModeContext
	target := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	candidateMV := motion.Vector{Row: -9, Col: -67}
	seedAboveMotion(&ctx, 4, BlockSize16x8, InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeGlobalMV},
		MV:         [2]motion.Vector{candidateMV},
	})
	req := ReferenceMVStackRequest{
		Size:             BlockSize8x8,
		References:       target,
		X4:               4,
		Y4:               0,
		HaveTop:          true,
		GlobalMVs:        [2]motion.Vector{{Row: -9, Col: -66}},
		GlobalMotionType: [2]parser.GlobalMotionType{parser.GlobalMotionTranslation},
	}
	result, err := ctx.BuildReferenceMVStack(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Stack.Count != 1 || result.Stack.Candidates[0].This != candidateMV {
		t.Fatalf("stack=%+v want single entry with candidate mv=%+v", result.Stack.Candidates[:result.Stack.Count], candidateMV)
	}
}

// TestBuildReferenceMVStackSmallGlobalMVBlockSkipsSubstitution checks that the
// gm_mv substitution requires the candidate's minimum side to be at least 8
// luma samples (W4>=2 and H4>=2), matching libaom's block_size_allowed gate.
func TestBuildReferenceMVStackSmallGlobalMVBlockSkipsSubstitution(t *testing.T) {
	var ctx BlockModeContext
	target := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	candidateMV := motion.Vector{Row: -9, Col: -67}
	seedAboveMotion(&ctx, 4, BlockSize8x4, InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeGlobalMV},
		MV:         [2]motion.Vector{candidateMV},
	})
	req := ReferenceMVStackRequest{
		Size:             BlockSize8x8,
		References:       target,
		X4:               4,
		Y4:               4,
		HaveTop:          true,
		GlobalMVs:        [2]motion.Vector{{Row: -9, Col: -66}},
		GlobalMotionType: [2]parser.GlobalMotionType{parser.GlobalMotionRotZoom},
	}
	result, err := ctx.BuildReferenceMVStack(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Stack.Count != 1 || result.Stack.Candidates[0].This != candidateMV {
		t.Fatalf("stack=%+v want single entry with candidate mv=%+v (W=8 H=4 fails min>=8 gate)", result.Stack.Candidates[:result.Stack.Count], candidateMV)
	}
}

// TestScanOuterBlockReferenceMVUsesSBLeftHistory verifies that the
// scanGridBlockReferenceMV(-1, -1) outer scan picks up the cross-SB neighbor
// in the column immediately to the left of the current superblock. Without the
// fallback to SBLeftInterMotionGrid the (x4=-1) lookup returns nothing because
// gridInterMotion only addresses cells inside the current SB, so goav1
// undercounted RowMatches relative to libaom's frame-wide mi grid; that
// undercount steered the refmv mode_context one cell off libaom and caused
// frame-1 cdf_update DRL_INDEX (seq 194166) to read from the wrong CDF cell.
// Mirrors block (mi_col=32, mi_row=30) on libaom_av1_8-bit_cdf_update at the
// left edge of SB(col=32, row=0) whose left neighbor (mi_col=31, mi_row=29)
// in the prior SB(col=0, row=0) is the only candidate the outer (-1, -1) scan
// can reach.
func TestScanOuterBlockReferenceMVUsesSBLeftHistory(t *testing.T) {
	var ctx BlockModeContext
	target := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	leftMV := motion.Vector{Row: 7, Col: -11}
	candidate := InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeNearestMV},
		MV:         [2]motion.Vector{leftMV},
	}
	// Seed the rightmost column of the SB to the left at y4=29 (the cell
	// immediately above the target block's left neighbor). depth 0 of
	// SBLeftInterMotionGrid carries the rightmost column of the prior SB.
	ctx.SBLeftInterMotionGrid[0][29] = candidate
	ctx.SBLeftMotionValidGrid[0][29] = 1
	ctx.SBLeftBlockSizeGrid[0][29] = BlockSize8x8
	ctx.SBLeftBlockSizeVisitedGrid[0][29] = 1

	var matches, newMatches int
	stack := ReferenceMVStack{}
	req := ReferenceMVStackRequest{
		Size:       BlockSize8x8,
		References: target,
		X4:         0,
		Y4:         30,
		HaveTop:    true,
		HaveLeft:   true,
	}
	ctx.scanGridBlockReferenceMV(req, -1, -1, &matches, &newMatches, &stack)
	if matches == 0 {
		t.Fatalf("expected RowMatches to count the SBLeft cross-SB neighbor, got 0")
	}
	if stack.Count == 0 {
		t.Fatalf("expected stack to gain one candidate from SBLeft, count=%d", stack.Count)
	}
	if stack.Candidates[0].This != leftMV {
		t.Fatalf("candidate=%+v want this=%+v", stack.Candidates[0], leftMV)
	}
}

// TestScanOuterBlockReferenceMVSkipsSBLeftWithoutHaveLeft verifies that the
// fallback is gated by HaveLeft so the outer (-1, -1) scan does not pick up
// a stale SBLeft snapshot when the current SB is the leftmost in the tile.
func TestScanOuterBlockReferenceMVSkipsSBLeftWithoutHaveLeft(t *testing.T) {
	var ctx BlockModeContext
	target := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	ctx.SBLeftInterMotionGrid[0][29] = InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeNearestMV},
		MV:         [2]motion.Vector{{Row: 7, Col: -11}},
	}
	ctx.SBLeftMotionValidGrid[0][29] = 1
	ctx.SBLeftBlockSizeGrid[0][29] = BlockSize8x8
	ctx.SBLeftBlockSizeVisitedGrid[0][29] = 1

	var matches, newMatches int
	stack := ReferenceMVStack{}
	req := ReferenceMVStackRequest{
		Size:       BlockSize8x8,
		References: target,
		X4:         0,
		Y4:         30,
		HaveTop:    true,
		HaveLeft:   false,
	}
	ctx.scanGridBlockReferenceMV(req, -1, -1, &matches, &newMatches, &stack)
	if matches != 0 || stack.Count != 0 {
		t.Fatalf("expected no candidates without HaveLeft, matches=%d stack.Count=%d", matches, stack.Count)
	}
}

// TestScanOuterBlockReferenceMVSkipsSBTopAtTileBoundary verifies the SBTop
// fallback in scanGridBlockReferenceMV gates on the candidate cell's mi_row
// staying within the tile. When the block sits in the topmost SB row of its
// tile, the (-1, -1) lookup at (x>=0, y<0) crosses the tile boundary and the
// fallback must return without adding a candidate, mirroring libaom's
// scan_blk_mbmi() is_inside() check.
func TestScanOuterBlockReferenceMVSkipsSBTopAtTileBoundary(t *testing.T) {
	var ctx BlockModeContext
	target := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	mv := InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeNearestMV},
		MV:         [2]motion.Vector{{Row: 1, Col: 2}},
	}
	ctx.SBTopInterMotionGrid[0][15] = mv
	ctx.SBTopMotionValidGrid[0][15] = 1
	ctx.SBTopBlockSizeGrid[0][15] = BlockSize8x8
	ctx.SBTopBlockSizeVisitedGrid[0][15] = 1

	// y4<0 case: block at (X4=16, Y4=0) inside topmost SB row of tile.
	// mi_row=0, tile_row_start=0, so the (-1, -1) lookup at y=-1 crosses
	// the tile boundary (mi_row + y = -1 < tile_row_start=0).
	var matches, newMatches int
	stack := ReferenceMVStack{}
	reqTop := ReferenceMVStackRequest{
		Size:           BlockSize8x8,
		References:     target,
		X4:             16,
		Y4:             0,
		MIRow:          0,
		MICol:          16,
		TileMIRowStart: 0,
		TileMIRowEnd:   72,
		TileMIColStart: 0,
		TileMIColEnd:   88,
		HaveTop:        true,
		HaveLeft:       true,
	}
	ctx.scanGridBlockReferenceMV(reqTop, -1, -1, &matches, &newMatches, &stack)
	if matches != 0 || stack.Count != 0 {
		t.Fatalf("expected no candidates across tile boundary, matches=%d stack.Count=%d", matches, stack.Count)
	}
}

// TestScanOuterBlockReferenceMVUsesSBTopCorner verifies that the
// scanGridBlockReferenceMV(-1, -1) outer scan picks up the cell directly
// above the current block when that cell sits in the SB-row above the
// current SB (x4>=0, y4<0) and the tile-row check passes. libaom's
// scan_blk_mbmi accesses xd->mi[(-1)*stride + (-1)] which translates to the
// frame mi cell (mi_row-1, mi_col-1). goav1 stages the prior SB-row's
// bottom-most grid row in SBTopInterMotionGrid[0][col] so the scan can
// reach across the SB boundary.
//
// This is the missing entry that desyncs cdf_update frame 1 mi=(64,2):
// libaom's stack[1] becomes (9,-71) (the ABOVE entry) after the
// (-1, -1) corner scan finds the SB-above bottom cell (block (62,0)
// mv=(7,21)) and bumps the LEFT entry's weight, pushing the LEFT entry
// out of stack[1] via the post-scan sort. Without this fallback goav1's
// stack[1] stays (7,21) and the NEARMV decode picks the wrong MV,
// cascading into the (68,0) seq=228966 DRL/joint divergence.
func TestScanOuterBlockReferenceMVUsesSBTopCorner(t *testing.T) {
	var ctx BlockModeContext
	target := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	cornerMV := motion.Vector{Row: 7, Col: 21}
	candidate := InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeNearestMV},
		MV:         [2]motion.Vector{cornerMV},
	}
	// Block (mi_row=64, mi_col=2) inside SB(2, 0) at X4=2 Y4=0. The
	// (-1, -1) corner maps to (x=1, y=-1), which translates to frame
	// (mi_row=63, mi_col=1). That cell lives in SB(1, 0) and is staged
	// in SBTopInterMotionGrid[0][1] (depth 0 = SB-above's bottom row).
	ctx.SBTopInterMotionGrid[0][1] = candidate
	ctx.SBTopMotionValidGrid[0][1] = 1
	ctx.SBTopBlockSizeGrid[0][1] = BlockSize8x8
	ctx.SBTopBlockSizeVisitedGrid[0][1] = 1

	var matches, newMatches int
	stack := ReferenceMVStack{}
	req := ReferenceMVStackRequest{
		Size:           BlockSize8x16,
		References:     target,
		X4:             2,
		Y4:             0,
		MIRow:          64,
		MICol:          2,
		TileMIRowStart: 0,
		TileMIRowEnd:   72,
		TileMIColStart: 0,
		TileMIColEnd:   88,
		HaveTop:        true,
		HaveLeft:       true,
	}
	ctx.scanGridBlockReferenceMV(req, -1, -1, &matches, &newMatches, &stack)
	if matches != 1 || stack.Count != 1 {
		t.Fatalf("expected SBTop corner match, matches=%d stack.Count=%d", matches, stack.Count)
	}
	if stack.Candidates[0].This != cornerMV {
		t.Fatalf("stack[0].This=%+v want %+v", stack.Candidates[0].This, cornerMV)
	}

	// Sanity: gating on HaveTop. When the current SB is the topmost SB row
	// of its tile, HaveTop=false and the lookup must return without adding
	// a candidate (the cell is outside the tile).
	matches, newMatches = 0, 0
	stack = ReferenceMVStack{}
	reqNoTop := req
	reqNoTop.HaveTop = false
	ctx.scanGridBlockReferenceMV(reqNoTop, -1, -1, &matches, &newMatches, &stack)
	if matches != 0 || stack.Count != 0 {
		t.Fatalf("expected no match without HaveTop, matches=%d stack.Count=%d", matches, stack.Count)
	}

	// Sanity: when SBTopMotionValidGrid[0][1]=0 (snapshot empty for that
	// column), the lookup must return without adding a candidate.
	ctx.SBTopMotionValidGrid[0][1] = 0
	matches, newMatches = 0, 0
	stack = ReferenceMVStack{}
	ctx.scanGridBlockReferenceMV(req, -1, -1, &matches, &newMatches, &stack)
	if matches != 0 || stack.Count != 0 {
		t.Fatalf("expected no match when SBTopMotionValidGrid empty, matches=%d stack.Count=%d", matches, stack.Count)
	}
}

// TestScanOuterBlockReferenceMVUsesSBDiagonalCorner verifies the cross-SB
// diagonal lookup for the outer (-1, -1) block-MV scan when the current
// block sits at the top-left corner of an SB (X4=0, Y4=0). libaom's
// frame-wide mi grid covers the diagonally-up-left cell (which lives in
// the SB up-and-to-the-left of the current one) and contributes a
// RowMatch when its reference frame matches the request. Without the
// SBDiagonal fallback goav1 silently dropped this contribution and
// undercounted RowMatches, e.g. monochrome frame 3 mi=(16,64) where the
// (15,63) corner cell sits inside SB(3,0)'s BLOCK_64X64 with ref0=LAST
// matching the request and contributes RowMatches=1. Without the lookup
// goav1's mode_context was 51 (refCtx=3); with the lookup it now matches
// libaom's mode_context=67 (refCtx=4) and the refmv_cdf read at the
// monochrome frame 3 SB(4,1) divergence point lines up with libaom.
func TestScanOuterBlockReferenceMVUsesSBDiagonalCorner(t *testing.T) {
	var ctx BlockModeContext
	target := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	cornerMV := motion.Vector{Row: 4, Col: -6}
	candidate := InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeNearestMV},
		MV:         [2]motion.Vector{cornerMV},
	}
	// Seed depth=0/e=0 of the SB-diagonal snapshot: the cell at frame
	// position (mi_col-1, mi_row-1) which lives in the SB diagonally
	// up-and-to-the-left of the current one.
	ctx.SBDiagonalInterMotionGrid[0][0] = candidate
	ctx.SBDiagonalMotionValidGrid[0][0] = 1
	ctx.SBDiagonalBlockSizeGrid[0][0] = BlockSize64x64
	ctx.SBDiagonalBlockSizeVisitedGrid[0][0] = 1

	var matches, newMatches int
	stack := ReferenceMVStack{}
	req := ReferenceMVStackRequest{
		Size:       BlockSize64x64,
		References: target,
		X4:         0,
		Y4:         0,
		HaveTop:    true,
		HaveLeft:   true,
	}
	ctx.scanGridBlockReferenceMV(req, -1, -1, &matches, &newMatches, &stack)
	if matches != 1 || stack.Count != 1 {
		t.Fatalf("expected diagonal corner match, matches=%d stack.Count=%d", matches, stack.Count)
	}
	if stack.Candidates[0].This != cornerMV {
		t.Fatalf("stack[0].This=%+v want %+v", stack.Candidates[0].This, cornerMV)
	}

	// Sanity: when either HaveTop or HaveLeft is false the diagonal lookup
	// must be skipped (the diagonal SB sits outside the tile boundary).
	matches, newMatches = 0, 0
	stack = ReferenceMVStack{}
	reqNoTop := req
	reqNoTop.HaveTop = false
	ctx.scanGridBlockReferenceMV(reqNoTop, -1, -1, &matches, &newMatches, &stack)
	if matches != 0 || stack.Count != 0 {
		t.Fatalf("expected no match without HaveTop, matches=%d stack.Count=%d", matches, stack.Count)
	}

	matches, newMatches = 0, 0
	stack = ReferenceMVStack{}
	reqNoLeft := req
	reqNoLeft.HaveLeft = false
	ctx.scanGridBlockReferenceMV(reqNoLeft, -1, -1, &matches, &newMatches, &stack)
	if matches != 0 || stack.Count != 0 {
		t.Fatalf("expected no match without HaveLeft, matches=%d stack.Count=%d", matches, stack.Count)
	}

	// Sanity: when SBDiagonalMotionValidGrid[0][0]=0 (no captured cell),
	// the lookup must still be skipped even when HaveTop/HaveLeft hold.
	ctx.SBDiagonalMotionValidGrid[0][0] = 0
	matches, newMatches = 0, 0
	stack = ReferenceMVStack{}
	ctx.scanGridBlockReferenceMV(req, -1, -1, &matches, &newMatches, &stack)
	if matches != 0 || stack.Count != 0 {
		t.Fatalf("expected no match when SBDiagonalMotionValidGrid empty, matches=%d stack.Count=%d", matches, stack.Count)
	}
}

// TestScanGridColReferenceMVsUsesSBLeftHistory verifies that the outer
// column reference-MV scan picks up the cross-SB neighbor in the prior SB
// when colOffset reaches x4<0. Mirrors libaom_av1_8-bit_mv frame 1 block
// (mi_row=26, mi_col=36) where the outer_col[-5] candidate at (mi_row=22,
// mi_col=31) sits in the SB to the left of the current SB(mi_col=32..63).
// Without the SBLeft fallback the col scan silently dropped the candidate,
// undercounting refmv_count from 3 to 2 and steering the DRL_INDEX cdf
// one cell off libaom.
func TestScanGridColReferenceMVsUsesSBLeftHistory(t *testing.T) {
	var ctx BlockModeContext
	target := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	leftMV := motion.Vector{Row: -8, Col: 10}
	candidate := InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeNearestMV},
		MV:         [2]motion.Vector{leftMV},
	}
	// Seed depth=0 of the SB-left snapshot at y4=11 (the cell five columns
	// left of the current X4=4 in the SB to the left). With X4=4 and
	// colOffset=-5, x = -1 (depth = -(-1) - 1 = 0). With rowOffset=1
	// (|colOffset|>1) and i=0, y = Y4 + rowOffset = 11.
	ctx.SBLeftInterMotionGrid[0][11] = candidate
	ctx.SBLeftMotionValidGrid[0][11] = 1
	ctx.SBLeftBlockSizeGrid[0][11] = BlockSize16x8
	ctx.SBLeftBlockSizeVisitedGrid[0][11] = 1

	processedCols := 4
	var matches, newMatches int
	stack := ReferenceMVStack{}
	dims, ok := BlockSize8x8.Dimensions()
	if !ok {
		t.Fatal("dims lookup failed")
	}
	req := ReferenceMVStackRequest{
		Size:       BlockSize8x8,
		References: target,
		X4:         4,
		Y4:         10,
		MIRow:      26,
		MICol:      36,
		HaveTop:    true,
		HaveLeft:   true,
	}
	ctx.scanGridColReferenceMVs(req, dims, -5, -6, &processedCols, &matches, &newMatches, &stack)
	if stack.Count == 0 {
		t.Fatalf("expected SBLeft candidate to be added to stack, count=%d", stack.Count)
	}
	if stack.Candidates[0].This != leftMV {
		t.Fatalf("candidate=%+v want this=%+v", stack.Candidates[0].This, leftMV)
	}
}

// TestScanGridColReferenceMVsSkipsSBLeftWithoutHaveLeft verifies the
// SBLeft fallback in the outer column scan gates on HaveLeft so blocks
// at the leftmost SB column do not consult a stale snapshot.
func TestScanGridColReferenceMVsSkipsSBLeftWithoutHaveLeft(t *testing.T) {
	var ctx BlockModeContext
	target := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	ctx.SBLeftInterMotionGrid[0][11] = InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeNearestMV},
		MV:         [2]motion.Vector{{Row: -8, Col: 10}},
	}
	ctx.SBLeftMotionValidGrid[0][11] = 1
	ctx.SBLeftBlockSizeGrid[0][11] = BlockSize16x8
	ctx.SBLeftBlockSizeVisitedGrid[0][11] = 1

	processedCols := 4
	var matches, newMatches int
	stack := ReferenceMVStack{}
	dims, ok := BlockSize8x8.Dimensions()
	if !ok {
		t.Fatal("dims lookup failed")
	}
	req := ReferenceMVStackRequest{
		Size:       BlockSize8x8,
		References: target,
		X4:         4,
		Y4:         10,
		MIRow:      26,
		MICol:      0,
		HaveTop:    true,
		HaveLeft:   false,
	}
	ctx.scanGridColReferenceMVs(req, dims, -5, -6, &processedCols, &matches, &newMatches, &stack)
	if stack.Count != 0 {
		t.Fatalf("expected no candidate without HaveLeft, count=%d", stack.Count)
	}
}

// TestScanGridColReferenceMVsStepsPastIntraNeighborInPriorSB pins the bug
// where the outer column ref-MV scan skipped only one mi past an intra
// neighbor that lives in the prior SB (cross-SB lookup with x4<0), letting
// the scan re-enter cells the intra neighbor already covered and pick up a
// stray inter MV one row deeper. libaom's scan_col_mbmi always advances
// i += AOMMIN(xd->height, mi_size_high[candidate->bsize]) regardless of
// whether the candidate is inter or intra; the cross-SB snapshot must
// expose the intra neighbor's block size to match that stride. Mirrors
// mfmv frame 1 block mi=(64,18) outer_col[-5] where libaom visits (19,59)
// (intra bsize=BLOCK_16X8, len=2) and exits with col_match=0 but goav1
// previously stepped to (20,59) (inter, mv=(-4,23)) and inflated
// col_match to 1, pushing modeContext from 59 (case 1, ref_match=1) to
// 75 (case 1, ref_match=2) and steering refmv_cdf one cell off libaom.
func TestScanGridColReferenceMVsStepsPastIntraNeighborInPriorSB(t *testing.T) {
	var ctx BlockModeContext
	target := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	deeperMV := motion.Vector{Row: -4, Col: 23}
	// depth=4 (x4=-5, depth = -x4-1 = 4): cell five columns into the prior
	// SB. Seed the row immediately past where libaom's len=2 step would
	// land with a stray inter candidate to verify goav1 does NOT visit it.
	ctx.SBLeftInterMotionGrid[4][20] = InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeNewMV},
		MV:         [2]motion.Vector{deeperMV},
	}
	ctx.SBLeftMotionValidGrid[4][20] = 1
	ctx.SBLeftBlockSizeGrid[4][20] = BlockSize32x32
	ctx.SBLeftBlockSizeVisitedGrid[4][20] = 1
	// The intra neighbor at depth=4 y=19 has MotionValid=0 (intra clears
	// it) but BlockSizeVisited=1 with the intra block's size recorded by
	// clearGridInterMotion. The outer scan must respect that block's
	// mi_size step and skip the cell at y=20 entirely.
	ctx.SBLeftMotionValidGrid[4][19] = 0
	ctx.SBLeftBlockSizeGrid[4][19] = BlockSize16x8
	ctx.SBLeftBlockSizeVisitedGrid[4][19] = 1

	processedCols := 6
	var matches, newMatches int
	stack := ReferenceMVStack{}
	dims, ok := BlockSize16x8.Dimensions()
	if !ok {
		t.Fatal("dims lookup failed")
	}
	req := ReferenceMVStackRequest{
		Size:       BlockSize16x8,
		References: target,
		X4:         0,
		Y4:         18,
		MIRow:      18,
		MICol:      64,
		HaveTop:    true,
		HaveLeft:   true,
	}
	ctx.scanGridColReferenceMVs(req, dims, -5, -6, &processedCols, &matches, &newMatches, &stack)
	if stack.Count != 0 {
		t.Fatalf("expected outer col[-5] scan to step past intra neighbor at y=19 without entering y=20; got stack count=%d", stack.Count)
	}
	if matches != 0 {
		t.Fatalf("expected matches=0 when only an intra neighbor sits at the col offset; got matches=%d", matches)
	}
}

// TestScanGridColReferenceMVsAdvancesProcessedColsForIntra pins that the
// outer column scan updates *processedCols unconditionally from the cell's
// recorded mi_size, even when the cell is intra (or carries no matching
// inter ref). libaom's scan_col_mbmi computes weight/processed_cols from the
// candidate's bsize BEFORE calling add_ref_mv_candidate, which performs the
// inter-block / ref-frame match. Without this advance, a goav1 outer column
// scan that finds only an intra cell at col_offset=-3 would leave
// processedCols at its pre-scan value, then re-enter cells at col_offset=-5
// that libaom's gate would have skipped — inflating ref_mv_count and
// drifting the DRL CDF. Mirror of the row-side guarantee in
// TestScanGridRowReferenceMVsAdvancesProcessedRowsForIntra.
func TestScanGridColReferenceMVsAdvancesProcessedColsForIntra(t *testing.T) {
	var ctx BlockModeContext
	target := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	// Seed an intra neighbor at (x=3, y=3) within the current SB: the cell
	// at X4=6 + colOffset=-3, Y4=2 + rowOffset=1 + i=0. MotionValid=0
	// (intra) but BlockSize recorded so the scan honors libaom's mi_size
	// stride.
	ctx.GridBlockSize[3][3] = BlockSize16x16
	ctx.GridBlockSizeVisited[3][3] = 1
	ctx.GridMotionValid[3][3] = 0

	processedCols := 2
	var matches, newMatches int
	stack := ReferenceMVStack{}
	dims, ok := BlockSize16x8.Dimensions()
	if !ok {
		t.Fatal("dims lookup failed")
	}
	req := ReferenceMVStackRequest{
		Size:       BlockSize16x8,
		References: target,
		X4:         6,
		Y4:         2,
		MIRow:      2,
		MICol:      6,
		HaveTop:    true,
		HaveLeft:   true,
	}
	ctx.scanGridColReferenceMVs(req, dims, -3, -6, &processedCols, &matches, &newMatches, &stack)
	if stack.Count != 0 || matches != 0 {
		t.Fatalf("intra-only scan should not add a candidate: stack=%d matches=%d", stack.Count, matches)
	}
	// The candidate's bsize is BLOCK_16X16 (sizeW4=4). inc = AOMMIN(
	//   -maxColOffset + colOffset + 1, sizeW4) = AOMMIN(6-3+1=4, 4) = 4.
	// *processedCols = inc - colOffset - 1 = 4 - (-3) - 1 = 6. Pre-fix the
	// scan returned without touching processedCols, leaving it at its
	// pre-scan value (2) and admitting a spurious col_offset=-5 scan.
	if processedCols != 6 {
		t.Fatalf("expected processedCols=6 after stepping over intra neighbor (BLOCK_16X16); got %d", processedCols)
	}
}

// TestScanGridRowReferenceMVsUsesSBTopHistory verifies that the outer row
// reference-MV scan picks up the cross-SB neighbor in the SB above when
// rowOffset reaches y4<0. Mirrors quantizer_00 frame 1 mi=(4,34)
// BLOCK_4X8 Y4=2 where outer_row[-3] (rowOffset=-3) maps to y4=-1 inside
// the SB at mi_row=16..31, exposed via SBTopBlockSizeGrid /
// SBTopInterMotionGrid depth 0 (the SB-above's bottommost row).
// Without the SBTop fallback the row scan silently dropped the candidate,
// undercounting refmv_count and shifting the DRL CDF off libaom.
// Symmetric to TestScanGridColReferenceMVsUsesSBLeftHistory.
func TestScanGridRowReferenceMVsUsesSBTopHistory(t *testing.T) {
	var ctx BlockModeContext
	target := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	aboveMV := motion.Vector{Row: -8, Col: 10}
	candidate := InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeNearestMV},
		MV:         [2]motion.Vector{aboveMV},
	}
	// Seed depth=0 of the SB-top snapshot at x4=5 (the cell three rows
	// above the current Y4=2 in the SB above, hosted at the SB-above's
	// bottommost row). With Y4=2 and rowOffset=-3, y = -1
	// (depth = -(-1) - 1 = 0). With colOffset=1 (|rowOffset|>1) and i=0,
	// x = X4 + colOffset = 4 + 1 = 5.
	ctx.SBTopInterMotionGrid[0][5] = candidate
	ctx.SBTopMotionValidGrid[0][5] = 1
	ctx.SBTopBlockSizeGrid[0][5] = BlockSize8x8
	ctx.SBTopBlockSizeVisitedGrid[0][5] = 1

	processedRows := 0
	var matches, newMatches int
	stack := ReferenceMVStack{}
	dims, ok := BlockSize4x8.Dimensions()
	if !ok {
		t.Fatal("dims lookup failed")
	}
	req := ReferenceMVStackRequest{
		Size:       BlockSize4x8,
		References: target,
		X4:         4,
		Y4:         2,
		MIRow:      34,
		MICol:      4,
		HaveTop:    true,
		HaveLeft:   true,
	}
	ctx.scanGridRowReferenceMVs(req, dims, -3, -6, &processedRows, &matches, &newMatches, &stack)
	if stack.Count == 0 {
		t.Fatalf("expected SBTop candidate to be added to stack, count=%d", stack.Count)
	}
	if stack.Candidates[0].This != aboveMV {
		t.Fatalf("candidate=%+v want this=%+v", stack.Candidates[0].This, aboveMV)
	}
}

// TestScanGridRowReferenceMVsSkipsSBTopWithoutHaveTop verifies the SBTop
// fallback in the outer row scan gates on HaveTop so blocks at the topmost
// SB row do not consult a stale snapshot. Symmetric to
// TestScanGridColReferenceMVsSkipsSBLeftWithoutHaveLeft.
func TestScanGridRowReferenceMVsSkipsSBTopWithoutHaveTop(t *testing.T) {
	var ctx BlockModeContext
	target := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	ctx.SBTopInterMotionGrid[0][5] = InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeNearestMV},
		MV:         [2]motion.Vector{{Row: -8, Col: 10}},
	}
	ctx.SBTopMotionValidGrid[0][5] = 1
	ctx.SBTopBlockSizeGrid[0][5] = BlockSize8x8
	ctx.SBTopBlockSizeVisitedGrid[0][5] = 1

	processedRows := 0
	var matches, newMatches int
	stack := ReferenceMVStack{}
	dims, ok := BlockSize4x8.Dimensions()
	if !ok {
		t.Fatal("dims lookup failed")
	}
	req := ReferenceMVStackRequest{
		Size:       BlockSize4x8,
		References: target,
		X4:         4,
		Y4:         2,
		MIRow:      2,
		MICol:      4,
		HaveTop:    false,
		HaveLeft:   true,
	}
	ctx.scanGridRowReferenceMVs(req, dims, -3, -6, &processedRows, &matches, &newMatches, &stack)
	if stack.Count != 0 {
		t.Fatalf("expected no candidate without HaveTop, count=%d", stack.Count)
	}
}

// TestScanGridRowReferenceMVsAdvancesProcessedRowsForIntra mirrors the
// column-side guarantee for the row scan: scan_row_mbmi must update
// *processedRows from the cell's mi_size before checking ref-frame match.
func TestScanGridRowReferenceMVsAdvancesProcessedRowsForIntra(t *testing.T) {
	var ctx BlockModeContext
	target := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	// Seed an intra neighbor at (x=3, y=3) within the current SB: the cell
	// at X4=2 + colOffset=1 + i=0, Y4=6 + rowOffset=-3. MotionValid=0 (intra)
	// but BlockSize recorded so the scan honors libaom's mi_size stride.
	ctx.GridBlockSize[3][3] = BlockSize16x16
	ctx.GridBlockSizeVisited[3][3] = 1
	ctx.GridMotionValid[3][3] = 0

	processedRows := 2
	var matches, newMatches int
	stack := ReferenceMVStack{}
	dims, ok := BlockSize8x16.Dimensions()
	if !ok {
		t.Fatal("dims lookup failed")
	}
	req := ReferenceMVStackRequest{
		Size:       BlockSize8x16,
		References: target,
		X4:         2,
		Y4:         6,
		MIRow:      6,
		MICol:      2,
		HaveTop:    true,
		HaveLeft:   true,
	}
	ctx.scanGridRowReferenceMVs(req, dims, -3, -6, &processedRows, &matches, &newMatches, &stack)
	if stack.Count != 0 || matches != 0 {
		t.Fatalf("intra-only scan should not add a candidate: stack=%d matches=%d", stack.Count, matches)
	}
	if processedRows != 6 {
		t.Fatalf("expected processedRows=6 after stepping over intra neighbor (BLOCK_16X16); got %d", processedRows)
	}
}

// TestBuildReferenceMVStackSingleRefMVsReflectFinalSort verifies that
// SingleRefMVs[] mirrors the final sorted stack order, not the insertion
// order produced by extendSingleReferenceMVStack. libaom's
// av1_find_best_ref_mvs reads from av1_setup_ref_mv_list's mv_ref_list,
// which is populated FROM the post-sort ref_mv_stack. Without the sort
// before setSingleRefMVs, NEARESTMV/NEARMV decoded the unsorted stack[0]
// candidate instead of the highest-weighted entry libaom selects.
// Mirrors quantizer_00 frame 1 mi=(46,10) where outer_blk[-1,-1] inserted
// a (-1,-8) candidate weight=4 first and outer_col[-3]+outer_col[-5]
// later merged a gm-substituted (-1,-7) candidate to weight=8; libaom
// moved (-1,-7) to slot 0 via the post-extend stack sort and decoded
// NEARESTMV from there while goav1 (pre-fix) saw stack[0]=(-1,-8) via
// the stale single-ref cache, shifting reconstruction at (46,10) by one
// column pixel and chain-corrupting downstream ref-MV stacks.
func TestBuildReferenceMVStackSingleRefMVsReflectFinalSort(t *testing.T) {
	var ctx BlockModeContext
	target := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	// Direct path: build a stack with insertion order != weight order.
	stack := ReferenceMVStack{}
	light := motion.Vector{Row: -1, Col: -8}
	heavy := motion.Vector{Row: -1, Col: -7}
	stack.Candidates[0] = ReferenceMVCandidate{This: light, Weight: 2}
	stack.Candidates[1] = ReferenceMVCandidate{This: heavy, Weight: 8}
	stack.Count = 2
	sortReferenceMVStack(&stack, 0, stack.Count)
	stack.setSingleRefMVs(motion.Vector{})
	if stack.Candidates[0].This != heavy {
		t.Fatalf("post-sort stack[0]=%+v want heavy=%+v", stack.Candidates[0].This, heavy)
	}
	if stack.SingleRefMVs[0] != heavy {
		t.Fatalf("SingleRefMVs[0]=%+v want heavy=%+v (pre-sort order leaked into single-ref cache)",
			stack.SingleRefMVs[0], heavy)
	}

	// Integration path: mirror libaom's quantizer_00 mi=(46,10). The
	// outer_blk[-1,-1] cell carries a NEARESTMV candidate with mv=(-1,-8)
	// inserted first at idx 0 weight=4. outer_col[-3] adds a GLOBALMV
	// candidate whose stored mv is the same but gm-substitution rewrites
	// it to gm0=(-1,-7), appending at idx 1 weight=4. outer_col[-5] adds
	// a NEARESTMV cell with mv=(-1,-7) that merges into idx 1, doubling
	// its weight to 8. The post-extend sort then promotes the (-1,-7)
	// slot to idx 0. Pre-fix SingleRefMVs[0] stays at the insertion-order
	// (-1,-8); post-fix it mirrors the sorted slot.
	near := InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeNearestMV},
		MV:         [2]motion.Vector{{Row: -1, Col: -8}},
	}
	gm := InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeGlobalMV},
		MV:         [2]motion.Vector{{Row: -1, Col: -8}},
	}
	merge := InterMotionResult{
		References: target,
		Mode:       InterModeResult{Mode: InterModeNearestMV},
		MV:         [2]motion.Vector{{Row: -1, Col: -7}},
	}
	for _, cell := range []struct {
		x, y int
		mr   InterMotionResult
		size BlockSize
	}{
		{x: 5, y: 9, mr: near, size: BlockSize8x8},
		{x: 3, y: 11, mr: gm, size: BlockSize8x16},
		{x: 1, y: 11, mr: merge, size: BlockSize8x16},
	} {
		ctx.GridInterMotion[cell.y][cell.x] = cell.mr
		ctx.GridMotionValid[cell.y][cell.x] = 1
		ctx.GridBlockSize[cell.y][cell.x] = cell.size
		ctx.GridBlockSizeVisited[cell.y][cell.x] = 1
	}
	for i := 0; i < 8; i++ {
		ctx.AboveBlockSize[i+6] = BlockSize8x8
		ctx.AboveIntra[i+6] = 1
		ctx.GridBlockSizeVisited[9][i+6] = 1
		ctx.GridBlockSize[9][i+6] = BlockSize8x8
		ctx.LeftBlockSize[i+10] = BlockSize8x8
		ctx.LeftIntra[i+10] = 1
	}
	result, err := ctx.BuildReferenceMVStack(ReferenceMVStackRequest{
		Size:           BlockSize16x8,
		References:     target,
		X4:             6,
		Y4:             10,
		MIRow:          10,
		MICol:          46,
		TileMIColStart: 0,
		TileMIColEnd:   88,
		TileMIRowStart: 0,
		TileMIRowEnd:   72,
		HaveTop:        true,
		HaveLeft:       true,
		GlobalMVs:      [2]motion.Vector{{Row: -1, Col: -7}},
		GlobalMotionType: [2]parser.GlobalMotionType{
			0: parser.GlobalMotionRotZoom,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Stack.SingleRefValid {
		t.Fatal("SingleRefValid=false after BuildReferenceMVStack")
	}
	if result.Stack.Count >= 1 && result.Stack.SingleRefMVs[0] != result.Stack.Candidates[0].This {
		t.Fatalf("SingleRefMVs[0]=%+v want stack[0]=%+v after BuildReferenceMVStack (pre-sort cache leak)",
			result.Stack.SingleRefMVs[0], result.Stack.Candidates[0].This)
	}
}
