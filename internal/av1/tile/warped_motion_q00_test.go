package tile

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

// TestWarpProjectionWithContextQ00Frame1MI80x4 pins WarpProjectionWithContext
// against the libaom-extracted warp samples and reduced shear parameters for
// the first WARPED_CAUSAL block of libaom's quantizer_00 frame 1 (mi=(80,4),
// BLOCK_8X16). libaom's av1_findSamples scans the 4-MI-tall left column
// without the OBMC neighbor cap and reaches three 1-MI-tall left neighbors at
// y={8,40,72} subpel-offset positions. The previous goav1 implementation
// shared OverlappableNeighborSet with OBMC, which caps left neighbors at
// MaxOBMCNeighborsForSize(BLOCK_8X16)=2 and dropped the y=40 sample, leaving
// the projected affine matrix off by enough to flip 2335 luma residuals at
// the WARPED_CAUSAL block boundary on the strict-q00 vector.
//
// Inputs are the exact pts/pts_inref pairs that libaom's av1_findSamples
// produced for this block. The expected matrix/shear comes from libaom's
// av1_find_projection on those samples (matching the warp parameters the
// canonical decoder feeds to av1_warp_affine for this WARPED_CAUSAL block).
func TestWarpProjectionWithContextQ00Frame1MI80x4(t *testing.T) {
	// Sample positions in goav1 internal form (X, Y in subpel; RefX, RefY in
	// subpel). These mirror libaom's av1_findSamples output for the block, in
	// the order produced after av1_selectSamples drops the threshold-rejected
	// fifth left sample (whose MV pointed at a different reference). The
	// fixture is reused both as the projection input and as the post-select
	// reference set; the regression specifically guards the projection step
	// (the post-select reduction is exercised by selectWarpSamples tests).
	samples := []warpSample{
		{X: 24, Y: -72, RefX: 17, RefY: -73, Motion: motion.Vector{Row: -1, Col: -7}},
		{X: -24, Y: 8, RefX: -31, RefY: 7, Motion: motion.Vector{Row: -1, Col: -7}},
		{X: -24, Y: 40, RefX: -30, RefY: 39, Motion: motion.Vector{Row: -1, Col: -6}},
		{X: -24, Y: 72, RefX: -30, RefY: 71, Motion: motion.Vector{Row: -1, Col: -6}},
		{X: -40, Y: -72, RefX: -47, RefY: -73, Motion: motion.Vector{Row: -1, Col: -7}},
		{X: 88, Y: -72, RefX: 81, RefY: -73, Motion: motion.Vector{Row: -1, Col: -7}},
	}

	block := BlockVisit{
		MICol: 80,
		MIRow: 4,
		Size:  BlockSize8x16,
	}
	model, invalid, err := findWarpProjection(samples, block, motion.Vector{Row: -1, Col: -7})
	if err != nil {
		t.Fatalf("findWarpProjection: %v", err)
	}
	if invalid {
		t.Fatalf("findWarpProjection invalid")
	}

	// libaom av1_find_projection output for the same inputs.
	wantMatrix := [6]int32{42833, -10423, 65225, 12, 0, 65633}
	if model.Params.Matrix != wantMatrix {
		t.Fatalf("matrix = %v want %v", model.Params.Matrix, wantMatrix)
	}
	if model.Alpha != -320 || model.Beta != 0 || model.Gamma != 0 || model.Delta != 128 {
		t.Fatalf("shear params = (a=%d b=%d g=%d d=%d) want (-320, 0, 0, 128)", model.Alpha, model.Beta, model.Gamma, model.Delta)
	}
}

// TestCollectWarpSamplesAboveLeftBypassesOBMCCap pins the warp-only neighbor
// scan added in WarpProjectionWithContext. The fixture mirrors libaom's view
// of the q00 frame 1 mi=(80,4) BLOCK_8X16 block: an 8x16 above neighbor (one
// MI wide, single ref Last) and three 1-MI-tall left neighbors (BLOCK_4X4
// rows), all single-ref Last with the same matching MV. The OBMC neighbor
// scan caps left neighbors at MaxOBMCNeighborsForSize(BLOCK_8X16)=2, but
// av1_findSamples must reach all three to give av1_find_projection the
// samples it needs. This test fails if the scan reverts to the OBMC cap.
func TestCollectWarpSamplesAboveLeftBypassesOBMCCap(t *testing.T) {
	var ctx BlockModeContext
	mv := motion.Vector{Row: -1, Col: -7}

	// Block under test: BLOCK_8X16 (W4=2, H4=4) at (MICol=80, MIRow=4) inside
	// SB col 1 (so X4=16, Y4=4). HaveTop and HaveLeft both true.
	block := BlockVisit{
		MICol:    80,
		MIRow:    4,
		Size:     BlockSize8x16,
		X4:       16,
		Y4:       4,
		HaveTop:  true,
		HaveLeft: true,
	}

	// Above neighbor: BLOCK_8X16 (W4=2) at X4=16, single-ref Last.
	ctx.AboveBlockSize[16] = BlockSize8x16
	ctx.AboveRef[0][16] = ReferenceFrameLast
	ctx.AboveRef[1][16] = ReferenceFrameNone
	ctx.AboveInterMotion[16].MV[0] = mv
	ctx.AboveInterMotion[16].References.Ref[0] = ReferenceFrameLast
	ctx.AboveInterMotion[16].References.Ref[1] = ReferenceFrameNone
	ctx.AboveMotionValid[16] = 1

	// Left neighbors: three 1-MI-tall (BLOCK_4X4) blocks at Y4=4,5,6, single
	// ref Last. The 4th left slot (Y4=7) carries an unrelated block whose ref
	// won't match; av1_findSamples won't add it.
	for slot := 4; slot <= 6; slot++ {
		ctx.LeftBlockSize[slot] = BlockSize4x4
		ctx.LeftRef[0][slot] = ReferenceFrameLast
		ctx.LeftRef[1][slot] = ReferenceFrameNone
		ctx.LeftInterMotion[slot].MV[0] = mv
		ctx.LeftInterMotion[slot].References.Ref[0] = ReferenceFrameLast
		ctx.LeftInterMotion[slot].References.Ref[1] = ReferenceFrameNone
		ctx.LeftMotionValid[slot] = 1
	}
	ctx.LeftBlockSize[7] = BlockSize4x4
	ctx.LeftRef[0][7] = ReferenceFrameGolden // does not match
	ctx.LeftRef[1][7] = ReferenceFrameNone
	ctx.LeftMotionValid[7] = 1

	var samples [maxWarpSamples]warpSample
	count, doTL, doTR, err := ctx.collectWarpSamplesAboveLeft(block, ReferenceFrameLast, &samples)
	if err != nil {
		t.Fatalf("collectWarpSamplesAboveLeft: %v", err)
	}
	if count != 4 {
		t.Fatalf("count = %d want 4 (1 above + 3 left). samples = %v", count, samples[:count])
	}
	// libaom's do_tl is cleared whenever the left scan reports row_offset < 0,
	// which only happens when blockH4 <= firstH4. Here firstH4=1 < blockH4=4,
	// so we take the "block height > above block height" branch and do_tl
	// stays true.
	_ = doTL
	_ = doTR
}
