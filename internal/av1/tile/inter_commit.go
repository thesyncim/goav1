package tile

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

// markDecodedInterContext commits the adjacent motion and interpolation-filter
// updates after sub8x8 chroma has collected the previous grid cells. ReadInterMotion
// preserves the reference result, and no observer runs between the two commits.
// Keep public MarkInterMotion and MarkInterFilters independently callable.
func (c *BlockModeContext) markDecodedInterContext(size BlockSize, x4, y4 int, result InterMotionResult, filters motion.InterpFilters, hasChroma bool) error {
	if !filters.X.Valid() || !filters.Y.Valid() {
		// Preserve motion-first errors and partial state even for an invalid
		// filter pair; the decoded valid path below has no failing second phase.
		if err := c.MarkInterMotion(size, x4, y4, result, hasChroma); err != nil {
			return fmt.Errorf("mark inter motion: %w", err)
		}
		if err := c.MarkInterFilters(size, x4, y4, result.References, filters); err != nil {
			return fmt.Errorf("mark inter filters: %w", err)
		}
		return nil
	}
	if err := validateInterMotion(result); err != nil {
		return fmt.Errorf("mark inter motion: %w", err)
	}
	dims, err := c.validateInterContext(size, x4, y4, result.References)
	if err != nil {
		return fmt.Errorf("mark inter motion: %w", err)
	}
	xEnd := x4 + int(dims.W4)
	yEnd := y4 + int(dims.H4)
	clear(c.AboveIntra[x4:xEnd])
	clear(c.AboveCompGroup[x4:xEnd])
	clear(c.AboveInterIntra[x4:xEnd])
	clear(c.LeftIntra[y4:yEnd])
	clear(c.LeftCompGroup[y4:yEnd])
	clear(c.LeftInterIntra[y4:yEnd])
	if hasChroma {
		clear(c.AboveChromaIntra[x4:xEnd])
		clear(c.AboveChromaMode[x4:xEnd])
		clear(c.LeftChromaIntra[y4:yEnd])
		clear(c.LeftChromaMode[y4:yEnd])
	}
	compound := boolByte(result.References.Compound)

	for x := x4; x < xEnd; x++ {
		c.AboveRef[0][x] = result.References.Ref[0]
		c.AboveRef[1][x] = result.References.Ref[1]
		c.AboveCompound[x] = compound
		c.AboveCompIndex[x] = 1
		c.AboveBlockSize[x] = size
		c.AboveInterMotion[x] = result
		c.AboveMotionValid[x] = 1
		c.AboveInterp[x] = filters
		c.AboveInterpValid[x] = 1
	}
	for y := y4; y < yEnd; y++ {
		c.LeftRef[0][y] = result.References.Ref[0]
		c.LeftRef[1][y] = result.References.Ref[1]
		c.LeftCompound[y] = compound
		c.LeftCompIndex[y] = 1
		c.LeftBlockSize[y] = size
		c.LeftInterMotion[y] = result
		c.LeftMotionValid[y] = 1
		c.LeftInterp[y] = filters
		c.LeftInterpValid[y] = 1
	}
	// Fill one row and replicate the same complete motion/filter state in
	// one grid walk, as dav1d's final block-context and ref-MV splats do.
	m0 := c.GridInterMotion[y4][x4:xEnd]
	v0 := c.GridMotionValid[y4][x4:xEnd]
	s0 := c.GridBlockSize[y4][x4:xEnd]
	z0 := c.GridBlockSizeVisited[y4][x4:xEnd]
	i0 := c.GridInterp[y4][x4:xEnd]
	f0 := c.GridInterpValid[y4][x4:xEnd]
	fillGridRow(m0, result)
	fillGridRow(v0, 1)
	fillGridRow(s0, size)
	fillGridRow(z0, 1)
	fillGridRow(i0, filters)
	fillGridRow(f0, 1)
	for y := y4 + 1; y < yEnd; y++ {
		copy(c.GridInterMotion[y][x4:xEnd], m0)
		copy(c.GridMotionValid[y][x4:xEnd], v0)
		copy(c.GridBlockSize[y][x4:xEnd], s0)
		copy(c.GridBlockSizeVisited[y][x4:xEnd], z0)
		copy(c.GridInterp[y][x4:xEnd], i0)
		copy(c.GridInterpValid[y][x4:xEnd], f0)
	}
	return nil
}
