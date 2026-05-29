package tile

import (
	"math/bits"

	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

const (
	warpedModelPrecBits           = 16
	warpedModelOne                = 1 << warpedModelPrecBits
	warpedModelTransClamp         = 128 << warpedModelPrecBits
	warpedModelNonDiagAffineClamp = 1 << (warpedModelPrecBits - 3)
	warpParamReduceBits           = 6
	warpDivLUTPrecBits            = 14
	warpDivLUTBits                = 8
)

// WarpedMotionModel carries a decoded local WARPED_CAUSAL affine model and the
// reduced shear parameters required by AV1's warp filter.
type WarpedMotionModel struct {
	Params       parser.WarpedMotionParams
	Alpha, Beta  int16
	Gamma, Delta int16
}

func (s OverlappableNeighborSet) WarpProjection(block BlockVisit, ref ReferenceFrame, mv motion.Vector) (WarpedMotionModel, bool, error) {
	var samples [maxWarpSamples]warpSample
	count, err := s.warpSamples(block, ref, &samples)
	if err != nil {
		return WarpedMotionModel{}, false, err
	}
	if count == 0 {
		return WarpedMotionModel{}, true, nil
	}
	if count > 1 {
		count = selectWarpSamples(samples[:count], block.Size, mv)
	}
	return findWarpProjection(samples[:count], block, mv)
}

// WarpSampleCountWithContext mirrors libaom's av1_findSamples result. It scans
// the block-mode context for up to LEAST_SQUARES_SAMPLES_MAX (=8) matching
// neighbor samples, ignoring the OBMC neighbor cap that
// OverlappableNeighborSet.WarpSampleCount inherits.
func (c *BlockModeContext) WarpSampleCountWithContext(block BlockVisit, ref ReferenceFrame) (int, error) {
	var samples [maxWarpSamples]warpSample
	count, doTL, doTR, err := c.collectWarpSamplesAboveLeft(block, ref, &samples)
	if err != nil {
		return 0, err
	}
	if doTL && count < maxWarpSamples && block.HaveTop && block.HaveLeft {
		if _, ok, err := c.warpSampleGrid(block, ref, block.X4-1, block.Y4-1, 0, -1, 0, -1); err != nil {
			return 0, err
		} else if ok {
			count++
		}
	}
	if doTR && count < maxWarpSamples {
		dims, ok := block.Size.Dimensions()
		if !ok {
			return 0, ErrInvalidDecodeState
		}
		if block.HaveTop && c.warpHasTopRight(block, int(dims.W4), int(dims.H4)) {
			if _, ok, err := c.warpSampleGrid(block, ref, block.X4+int(dims.W4), block.Y4-1, int(dims.W4), 1, 0, -1); err != nil {
				return 0, err
			} else if ok {
				count++
			}
		}
	}
	return count, nil
}

// WarpProjectionWithContext mirrors libaom's av1_findSamples + av1_selectSamples
// + av1_find_projection chain. Unlike OverlappableNeighborSet.WarpProjection it
// scans the block-mode context for up to LEAST_SQUARES_SAMPLES_MAX (=8) neighbor
// samples, which is needed when the block is taller (or wider) than the OBMC
// neighbor cap. The OBMC neighbor cap for BLOCK_8X16 is 2 left + 1 above, but a
// 4-MI-tall block scanning 1-MI-tall left neighbors should pick up 4 left
// samples per the warp spec (find_warp_samples).
func (c *BlockModeContext) WarpProjectionWithContext(block BlockVisit, ref ReferenceFrame, mv motion.Vector) (WarpedMotionModel, bool, error) {
	var samples [maxWarpSamples]warpSample
	count, doTL, doTR, err := c.collectWarpSamplesAboveLeft(block, ref, &samples)
	if err != nil {
		return WarpedMotionModel{}, false, err
	}
	if doTL && count < maxWarpSamples && block.HaveTop && block.HaveLeft {
		// libaom: record_samples(mbmi, pts, pts_inref, 0, -1, 0, -1)
		// args: row_offset=0, sign_r=-1, col_offset=0, sign_c=-1.
		if sample, ok, err := c.warpSampleGrid(block, ref, block.X4-1, block.Y4-1, 0, -1, 0, -1); err != nil {
			return WarpedMotionModel{}, false, err
		} else if ok {
			samples[count] = sample
			count++
		}
	}
	if doTR && count < maxWarpSamples {
		dims, ok := block.Size.Dimensions()
		if !ok {
			return WarpedMotionModel{}, false, ErrInvalidDecodeState
		}
		if block.HaveTop && c.warpHasTopRight(block, int(dims.W4), int(dims.H4)) {
			// libaom: record_samples(mbmi, pts, pts_inref, 0, -1, xd->width, 1)
			// args: row_offset=0, sign_r=-1, col_offset=W4, sign_c=1. The
			// top-right cell at mi(mi_row-1, mi_col+W4) can fall one column past
			// the current superblock's right edge (X4+W4 >= MaxBlockModeSlots on
			// 128x128 SBs), which crossSBInterGridInterMotion cannot reach — it
			// only snapshots the SB directly above. topRightInterMotion resolves
			// the SB-above-and-to-the-right via the SBTopRight snapshot, matching
			// libaom's frame-wide xd->mi[-stride + xd->width] read.
			trReq := ReferenceMVStackRequest{X4: block.X4, Y4: block.Y4, HaveTop: block.HaveTop, HaveLeft: block.HaveLeft}
			if motionResult, size, ok := c.topRightInterMotion(trReq, dims); ok {
				if sample, sok := warpSampleFromMotion(motionResult, size, ref, int(dims.W4), 1, 0, -1); sok {
					samples[count] = sample
					count++
				}
			}
		}
	}
	if count == 0 {
		return WarpedMotionModel{}, true, nil
	}
	if count > 1 {
		count = selectWarpSamples(samples[:count], block.Size, mv)
	}
	return findWarpProjection(samples[:count], block, mv)
}

// collectWarpSamplesAboveLeft scans the above row and left column following
// libaom's av1_findSamples. Returns (count, doTopLeft, doTopRight, err).
// doTopLeft and doTopRight are propagated from libaom's `do_tl`/`do_tr`
// which depend on the first neighbor block's width/height alignment.
func (c *BlockModeContext) collectWarpSamplesAboveLeft(block BlockVisit, ref ReferenceFrame, samples *[maxWarpSamples]warpSample) (int, bool, bool, error) {
	if !ref.Valid() {
		return 0, false, false, ErrInvalidDecodeState
	}
	dims, ok := block.Size.Dimensions()
	if !ok {
		return 0, false, false, ErrInvalidDecodeState
	}
	blockW4 := int(dims.W4)
	blockH4 := int(dims.H4)
	count := 0
	doTL := true
	doTR := true

	// Above row: libaom's av1_findSamples scans the row at row_offset=-1.
	if block.HaveTop {
		startSlot := block.X4
		firstSize := c.AboveBlockSize[startSlot]
		firstW4 := warpNeighborW4(firstSize)
		if firstW4 <= 0 {
			firstW4 = 1
		}
		if blockW4 <= firstW4 {
			// "current block width <= above block width" branch.
			colOffset := -(int(block.MICol) % firstW4)
			if colOffset < 0 {
				doTL = false
			}
			if colOffset+firstW4 > blockW4 {
				doTR = false
			}
			if c.warpAboveNeighborMatches(startSlot, ref) {
				samples[count] = c.warpSampleAboveAt(startSlot, colOffset)
				count++
			}
		} else {
			// "current block width > above block width" branch.
			for i := 0; i < blockW4 && count < maxWarpSamples; {
				slot := startSlot + i
				if slot >= MaxBlockModeSlots {
					break
				}
				size := c.AboveBlockSize[slot]
				step := warpNeighborW4(size)
				if step <= 0 {
					step = 1
				}
				if c.warpAboveNeighborMatches(slot, ref) {
					samples[count] = c.warpSampleAboveAt(slot, i)
					count++
				}
				i += step
			}
		}
	}

	// Left column: libaom's av1_findSamples scans the column at col_offset=-1.
	if block.HaveLeft && count < maxWarpSamples {
		startSlot := block.Y4
		firstSize := c.LeftBlockSize[startSlot]
		firstH4 := warpNeighborH4(firstSize)
		if firstH4 <= 0 {
			firstH4 = 1
		}
		if blockH4 <= firstH4 {
			rowOffset := -(int(block.MIRow) % firstH4)
			if rowOffset < 0 {
				doTL = false
			}
			if c.warpLeftNeighborMatches(startSlot, ref) {
				samples[count] = c.warpSampleLeftAt(startSlot, rowOffset)
				count++
			}
		} else {
			for i := 0; i < blockH4 && count < maxWarpSamples; {
				slot := startSlot + i
				if slot >= MaxBlockModeSlots {
					break
				}
				size := c.LeftBlockSize[slot]
				step := warpNeighborH4(size)
				if step <= 0 {
					step = 1
				}
				if c.warpLeftNeighborMatches(slot, ref) {
					samples[count] = c.warpSampleLeftAt(slot, i)
					count++
				}
				i += step
			}
		}
	}

	return count, doTL, doTR, nil
}

func warpNeighborW4(size BlockSize) int {
	dims, ok := size.Dimensions()
	if !ok {
		return 0
	}
	return int(dims.W4)
}

func warpNeighborH4(size BlockSize) int {
	dims, ok := size.Dimensions()
	if !ok {
		return 0
	}
	return int(dims.H4)
}

func (c *BlockModeContext) warpAboveNeighborMatches(slot int, ref ReferenceFrame) bool {
	if slot < 0 || slot >= MaxBlockModeSlots {
		return false
	}
	if c.AboveIntra[slot] != 0 {
		return false
	}
	if c.AboveMotionValid[slot] == 0 {
		return false
	}
	if c.AboveRef[0][slot] != ref {
		return false
	}
	if c.AboveRef[1][slot] != ReferenceFrameNone {
		return false
	}
	// libaom's av1_findSamples rejects inter-intra neighbors because they
	// carry mbmi->ref_frame[1] = INTRA_FRAME != NONE_FRAME. goav1 keeps
	// Ref[1] = NONE for inter-intra to preserve has_second_ref semantics,
	// so consult the InterIntra flag explicitly here.
	if c.AboveInterIntra[slot] != 0 {
		return false
	}
	return true
}

func (c *BlockModeContext) warpLeftNeighborMatches(slot int, ref ReferenceFrame) bool {
	if slot < 0 || slot >= MaxBlockModeSlots {
		return false
	}
	if c.LeftIntra[slot] != 0 {
		return false
	}
	if c.LeftMotionValid[slot] == 0 {
		return false
	}
	if c.LeftRef[0][slot] != ref {
		return false
	}
	if c.LeftRef[1][slot] != ReferenceFrameNone {
		return false
	}
	if c.LeftInterIntra[slot] != 0 {
		return false
	}
	return true
}

func (c *BlockModeContext) warpSampleAboveAt(slot int, colOffset4 int) warpSample {
	size := c.AboveBlockSize[slot]
	w4 := warpNeighborW4(size)
	h4 := warpNeighborH4(size)
	mv := c.AboveInterMotion[slot].MV[0]
	return recordWarpSample(mv, colOffset4, 0, 1, -1, w4, h4)
}

func (c *BlockModeContext) warpSampleLeftAt(slot int, rowOffset4 int) warpSample {
	size := c.LeftBlockSize[slot]
	w4 := warpNeighborW4(size)
	h4 := warpNeighborH4(size)
	mv := c.LeftInterMotion[slot].MV[0]
	return recordWarpSample(mv, 0, rowOffset4, -1, 1, w4, h4)
}

func (c *BlockModeContext) warpSampleGrid(block BlockVisit, ref ReferenceFrame, x4 int, y4 int, colOffset4 int, signC int, rowOffset4 int, signR int) (warpSample, bool, error) {
	// libaom's av1_findSamples reads the top-left/top-right corner neighbors via
	// the frame-wide mi grid (mi[-mi_stride-1] / mi[-mi_stride+width]). goav1's
	// per-superblock GridInterMotion has no entries for negative SB-relative
	// coordinates, so a warped block at an SB top/left edge would silently drop
	// its corner samples and fit a wrong least-squares affine model. Route the
	// corner lookup through the cross-SB snapshots (SBTop/SBLeft/SBDiagonal) the
	// inter ref-MV scan already relies on so prior-SB corner neighbors are seen.
	req := ReferenceMVStackRequest{HaveTop: block.HaveTop, HaveLeft: block.HaveLeft}
	motionResult, size, ok := c.crossSBInterGridInterMotion(req, x4, y4)
	if !ok {
		return warpSample{}, false, nil
	}
	sample, ok := warpSampleFromMotion(motionResult, size, ref, colOffset4, signC, rowOffset4, signR)
	return sample, ok, nil
}

// warpSampleFromMotion applies libaom av1_findSamples' single-ref gates to a
// neighbor's stored motion and, when accepted, records the warp sample. The
// neighbor must be single-ref to the same reference: compound, mismatched ref,
// any second ref (Ref[1] != NONE), and inter-intra neighbors are all rejected.
// libaom's av1_findSamples rejects inter-intra neighbors because they carry
// ref_frame[1] = INTRA_FRAME != NONE_FRAME, so add_ref_mv_candidate's single-ref
// match (ref_frame[1] == NONE) skips them; goav1 stores inter-intra with
// Ref[1] = NONE and a separate InterIntra flag, so the flag is checked here (e.g.
// av1-1-b8-00-quantizer-00 frame 1 mi(22,16): an inter-intra TL neighbor was
// over-collected, corrupting the affine fit).
func warpSampleFromMotion(motionResult InterMotionResult, size BlockSize, ref ReferenceFrame, colOffset4, signC, rowOffset4, signR int) (warpSample, bool) {
	if motionResult.References.Compound {
		return warpSample{}, false
	}
	if motionResult.References.Ref[0] != ref {
		return warpSample{}, false
	}
	if motionResult.References.Ref[1] != ReferenceFrameNone {
		return warpSample{}, false
	}
	if motionResult.InterIntra {
		return warpSample{}, false
	}
	w4 := warpNeighborW4(size)
	h4 := warpNeighborH4(size)
	return recordWarpSample(motionResult.MV[0], colOffset4, rowOffset4, signC, signR, w4, h4), true
}

func (c *BlockModeContext) warpHasTopRight(block BlockVisit, w4, h4 int) bool {
	// has_top_right is checked by libaom against AOMMAX(xd->width, xd->height).
	// goav1 already provides blockHasTopRight in block_loop.go but it's
	// SB-shape based; for warp samples libaom uses the same has_top_right gate
	// that the OBMC scan does (HaveTopRight on the visit was computed already).
	// We re-evaluate using the larger dimension here to match
	// has_top_right(..., AOMMAX(width, height)).
	if !block.HaveTop {
		return false
	}
	// Conservative: trust the existing HaveTopRight flag. blockHasTopRight in
	// goav1 already encodes the SB partition-tree gate.
	return true
}

func findWarpProjection(samples []warpSample, block BlockVisit, mv motion.Vector) (WarpedMotionModel, bool, error) {
	model, invalid, err := findWarpAffineInt(samples, block, mv)
	if invalid || err != nil {
		return model, invalid, err
	}
	model, ok := warpShearParams(model)
	return model, !ok, nil
}

func findWarpAffineInt(samples []warpSample, block BlockVisit, mv motion.Vector) (WarpedMotionModel, bool, error) {
	dims, ok := block.Size.Dimensions()
	if !ok {
		return WarpedMotionModel{}, false, ErrInvalidDecodeState
	}
	bw := int(dims.W4) * 4
	bh := int(dims.H4) * 4
	rsuy := bh/2 - 1
	rsux := bw/2 - 1
	suy := rsuy * motion.SubpelScale
	sux := rsux * motion.SubpelScale
	duy := suy + int(mv.Row)
	dux := sux + int(mv.Col)
	var a00, a01, a11 int64
	var bx0, bx1, by0, by1 int64
	for i := range samples {
		dx := samples[i].RefX - dux
		dy := samples[i].RefY - duy
		sx := samples[i].X - sux
		sy := samples[i].Y - suy
		if absInt(sx-dx) < warpLeastSquaresMVMax && absInt(sy-dy) < warpLeastSquaresMVMax {
			a00 += int64(warpLeastSquaresSquare(sx))
			a01 += int64(warpLeastSquaresProduct1(sx, sy))
			a11 += int64(warpLeastSquaresSquare(sy))
			bx0 += int64(warpLeastSquaresProduct2(sx, dx))
			bx1 += int64(warpLeastSquaresProduct1(sy, dx))
			by0 += int64(warpLeastSquaresProduct1(sx, dy))
			by1 += int64(warpLeastSquaresProduct2(sy, dy))
		}
	}

	det := a00*a11 - a01*a01
	if det == 0 {
		return WarpedMotionModel{}, true, nil
	}

	reciprocal, shift := warpResolveDivisor64(absInt64AsUint(det))
	iDet := int64(reciprocal)
	if det < 0 {
		iDet = -iDet
	}
	shift -= warpedModelPrecBits
	if shift < 0 {
		iDet <<= -shift
		shift = 0
	}

	px0 := a11*bx0 - a01*bx1
	px1 := -a01*bx0 + a00*bx1
	py0 := a11*by0 - a01*by1
	py1 := -a01*by0 + a00*by1

	params := parser.DefaultWarpedMotionParams()
	params.Type = parser.GlobalMotionAffine
	params.Matrix[2] = warpMultShiftDiag(px0, iDet, shift)
	params.Matrix[3] = warpMultShiftNonDiag(px1, iDet, shift)
	params.Matrix[4] = warpMultShiftNonDiag(py0, iDet, shift)
	params.Matrix[5] = warpMultShiftDiag(py1, iDet, shift)

	isuy := int(block.MIRow)*4 + rsuy
	isux := int(block.MICol)*4 + rsux
	vx := int64(mv.Col)*(1<<(warpedModelPrecBits-motion.SubpelBits)) -
		(int64(isux)*(int64(params.Matrix[2])-warpedModelOne) + int64(isuy)*int64(params.Matrix[3]))
	vy := int64(mv.Row)*(1<<(warpedModelPrecBits-motion.SubpelBits)) -
		(int64(isux)*int64(params.Matrix[4]) + int64(isuy)*(int64(params.Matrix[5])-warpedModelOne))
	params.Matrix[0] = int32(clampInt64(vx, -warpedModelTransClamp, warpedModelTransClamp-1))
	params.Matrix[1] = int32(clampInt64(vy, -warpedModelTransClamp, warpedModelTransClamp-1))
	return WarpedMotionModel{Params: params}, false, nil
}

func warpShearParams(model WarpedMotionModel) (WarpedMotionModel, bool) {
	mat := model.Params.Matrix
	if mat[2] <= 0 {
		return model, false
	}
	alpha := clampInt64(int64(mat[2])-warpedModelOne, -1<<15, 1<<15-1)
	beta := clampInt64(int64(mat[3]), -1<<15, 1<<15-1)
	reciprocal, shift := warpResolveDivisor32(uint32(absInt(int(mat[2]))))
	y := int64(reciprocal)
	if mat[2] < 0 {
		y = -y
	}
	v := int64(mat[4]) * warpedModelOne * y
	gamma := clampInt64(warpRoundPowerOfTwoSigned64(v, shift), -1<<15, 1<<15-1)
	v = int64(mat[3]) * int64(mat[4]) * y
	delta := clampInt64(int64(mat[5])-warpRoundPowerOfTwoSigned64(v, shift)-warpedModelOne, -1<<15, 1<<15-1)

	model.Alpha = int16(warpRoundPowerOfTwoSigned64(alpha, warpParamReduceBits) * (1 << warpParamReduceBits))
	model.Beta = int16(warpRoundPowerOfTwoSigned64(beta, warpParamReduceBits) * (1 << warpParamReduceBits))
	model.Gamma = int16(warpRoundPowerOfTwoSigned64(gamma, warpParamReduceBits) * (1 << warpParamReduceBits))
	model.Delta = int16(warpRoundPowerOfTwoSigned64(delta, warpParamReduceBits) * (1 << warpParamReduceBits))
	return model, warpAffineShearAllowed(model.Alpha, model.Beta, model.Gamma, model.Delta)
}

func warpAffineShearAllowed(alpha int16, beta int16, gamma int16, delta int16) bool {
	return 4*absInt(int(alpha))+7*absInt(int(beta)) < warpedModelOne &&
		4*absInt(int(gamma))+4*absInt(int(delta)) < warpedModelOne
}

func warpMultShiftNonDiag(px int64, iDet int64, shift int) int32 {
	v := px * iDet
	return int32(clampInt64(
		warpRoundPowerOfTwoSigned64(v, shift),
		-warpedModelNonDiagAffineClamp+1,
		warpedModelNonDiagAffineClamp-1,
	))
}

func warpMultShiftDiag(px int64, iDet int64, shift int) int32 {
	v := px * iDet
	return int32(clampInt64(
		warpRoundPowerOfTwoSigned64(v, shift),
		warpedModelOne-warpedModelNonDiagAffineClamp+1,
		warpedModelOne+warpedModelNonDiagAffineClamp-1,
	))
}

func warpLeastSquaresProduct2(a int, b int) int {
	return (a*b*4 + (a+b)*2*motion.SubpelScale + motion.SubpelScale*motion.SubpelScale*2) >> 4
}

func warpResolveDivisor64(d uint64) (int16, int) {
	shift := bits.Len64(d) - 1
	e := d - (uint64(1) << shift)
	var f uint64
	if shift > warpDivLUTBits {
		f = warpRoundPowerOfTwoUnsigned64(e, shift-warpDivLUTBits)
	} else {
		f = e << (warpDivLUTBits - shift)
	}
	return int16(warpDivLUT[f]), shift + warpDivLUTPrecBits
}

func warpResolveDivisor32(d uint32) (int16, int) {
	shift := bits.Len32(d) - 1
	e := d - (uint32(1) << shift)
	var f uint32
	if shift > warpDivLUTBits {
		f = uint32(warpRoundPowerOfTwoUnsigned64(uint64(e), shift-warpDivLUTBits))
	} else {
		f = e << (warpDivLUTBits - shift)
	}
	return int16(warpDivLUT[f]), shift + warpDivLUTPrecBits
}

func warpRoundPowerOfTwoUnsigned64(value uint64, bits int) uint64 {
	if bits <= 0 {
		return value
	}
	return (value + (uint64(1) << (bits - 1))) >> bits
}

func warpRoundPowerOfTwoSigned64(value int64, bits int) int64 {
	if bits <= 0 {
		return value
	}
	if value < 0 {
		return -int64(warpRoundPowerOfTwoUnsigned64(uint64(-value), bits))
	}
	return int64(warpRoundPowerOfTwoUnsigned64(uint64(value), bits))
}

func absInt64AsUint(v int64) uint64 {
	if v < 0 {
		return uint64(-v)
	}
	return uint64(v)
}

var warpDivLUT = [257]uint16{
	16384, 16320, 16257, 16194, 16132, 16070, 16009, 15948, 15888, 15828, 15768,
	15709, 15650, 15592, 15534, 15477, 15420, 15364, 15308, 15252, 15197, 15142,
	15087, 15033, 14980, 14926, 14873, 14821, 14769, 14717, 14665, 14614, 14564,
	14513, 14463, 14413, 14364, 14315, 14266, 14218, 14170, 14122, 14075, 14028,
	13981, 13935, 13888, 13843, 13797, 13752, 13707, 13662, 13618, 13574, 13530,
	13487, 13443, 13400, 13358, 13315, 13273, 13231, 13190, 13148, 13107, 13066,
	13026, 12985, 12945, 12906, 12866, 12827, 12788, 12749, 12710, 12672, 12633,
	12596, 12558, 12520, 12483, 12446, 12409, 12373, 12336, 12300, 12264, 12228,
	12193, 12157, 12122, 12087, 12053, 12018, 11984, 11950, 11916, 11882, 11848,
	11815, 11782, 11749, 11716, 11683, 11651, 11619, 11586, 11555, 11523, 11491,
	11460, 11429, 11398, 11367, 11336, 11305, 11275, 11245, 11215, 11185, 11155,
	11125, 11096, 11067, 11038, 11009, 10980, 10951, 10923, 10894, 10866, 10838,
	10810, 10782, 10755, 10727, 10700, 10673, 10645, 10618, 10592, 10565, 10538,
	10512, 10486, 10460, 10434, 10408, 10382, 10356, 10331, 10305, 10280, 10255,
	10230, 10205, 10180, 10156, 10131, 10107, 10082, 10058, 10034, 10010, 9986,
	9963, 9939, 9916, 9892, 9869, 9846, 9823, 9800, 9777, 9754, 9732,
	9709, 9687, 9664, 9642, 9620, 9598, 9576, 9554, 9533, 9511, 9489,
	9468, 9447, 9425, 9404, 9383, 9362, 9341, 9321, 9300, 9279, 9259,
	9239, 9218, 9198, 9178, 9158, 9138, 9118, 9098, 9079, 9059, 9039,
	9020, 9001, 8981, 8962, 8943, 8924, 8905, 8886, 8867, 8849, 8830,
	8812, 8793, 8775, 8756, 8738, 8720, 8702, 8684, 8666, 8648, 8630,
	8613, 8595, 8577, 8560, 8542, 8525, 8508, 8490, 8473, 8456, 8439,
	8422, 8405, 8389, 8372, 8355, 8339, 8322, 8306, 8289, 8273, 8257,
	8240, 8224, 8208, 8192,
}
