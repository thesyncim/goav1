package tile

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

// MarkInterMotionAndFilters commits adjacent motion and interpolation-filter
// updates together, after the caller has collected any sub8x8 chroma cells. It
// preserves the separate calls' motion-first errors and partial state. Keep the
// individual methods for callers that observe the intermediate state.
func (c *BlockModeContext) MarkInterMotionAndFilters(size BlockSize, x4, y4 int, result InterMotionResult, filters motion.InterpFilters, hasChroma bool) error {
	return c.markInterMotionAndFilters(size, x4, y4, result, filters, hasChroma, true)
}

// markInterMotionAndFilters is MarkInterMotionAndFilters; keepMotion false
// skips the neighbor-motion arrays and grids (the frame decoder keeps
// neighbor motion in the dav1d refmvs rows instead). Like dav1d's pal_sz edge
// bytes, only the palette sizes are reset: the colour caches read a
// neighbour's colours only up to its size.
func (c *BlockModeContext) markInterMotionAndFilters(size BlockSize, x4, y4 int, result InterMotionResult, filters motion.InterpFilters, hasChroma bool, keepMotion bool) error {
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
	compound := boolByte(result.References.Compound)

	// dav1d case_set: each edge array gets one store per up to eight
	// entries, from the replicated value.
	ref0 := ctxSplat(uint8(result.References.Ref[0]))
	ref1 := ctxSplat(uint8(result.References.Ref[1]))
	comp := ctxSplat(compound)
	bs := ctxSplat(uint8(size))
	const zero, one = 0, 0x0101010101010101
	switch dims.W4 {
	case 1:
		c.A.Intra[x4] = 0
		c.A.CompGroup[x4] = 0
		c.A.InterIntra[x4] = 0
		c.A.Ref[0][x4] = result.References.Ref[0]
		c.A.Ref[1][x4] = result.References.Ref[1]
		c.A.Compound[x4] = compound
		c.A.CompIndex[x4] = 1
		c.A.BlockSize[x4] = size
		c.A.InterpValid[x4] = 1
		if hasChroma {
			c.A.ChromaIntra[x4] = 0
			c.A.UVMode[x4] = 0
		}
	case 2:
		ctxPut2(c.A.Intra[x4:], zero)
		ctxPut2(c.A.CompGroup[x4:], zero)
		ctxPut2(c.A.InterIntra[x4:], zero)
		ctxPut2(c.A.Ref[0][x4:], ref0)
		ctxPut2(c.A.Ref[1][x4:], ref1)
		ctxPut2(c.A.Compound[x4:], comp)
		ctxPut2(c.A.CompIndex[x4:], one)
		ctxPut2(c.A.BlockSize[x4:], bs)
		ctxPut2(c.A.InterpValid[x4:], one)
		if hasChroma {
			ctxPut2(c.A.ChromaIntra[x4:], 0)
			ctxPut2(c.A.UVMode[x4:], 0)
		}
	case 4:
		ctxPut4(c.A.Intra[x4:], zero)
		ctxPut4(c.A.CompGroup[x4:], zero)
		ctxPut4(c.A.InterIntra[x4:], zero)
		ctxPut4(c.A.Ref[0][x4:], ref0)
		ctxPut4(c.A.Ref[1][x4:], ref1)
		ctxPut4(c.A.Compound[x4:], comp)
		ctxPut4(c.A.CompIndex[x4:], one)
		ctxPut4(c.A.BlockSize[x4:], bs)
		ctxPut4(c.A.InterpValid[x4:], one)
		if hasChroma {
			ctxPut4(c.A.ChromaIntra[x4:], 0)
			ctxPut4(c.A.UVMode[x4:], 0)
		}
	default:
		for i := x4; i < x4+int(dims.W4); i += 8 {
			ctxPut8(c.A.Intra[i:], zero)
			ctxPut8(c.A.CompGroup[i:], zero)
			ctxPut8(c.A.InterIntra[i:], zero)
			ctxPut8(c.A.Ref[0][i:], ref0)
			ctxPut8(c.A.Ref[1][i:], ref1)
			ctxPut8(c.A.Compound[i:], comp)
			ctxPut8(c.A.CompIndex[i:], one)
			ctxPut8(c.A.BlockSize[i:], bs)
			ctxPut8(c.A.InterpValid[i:], one)
			if hasChroma {
				ctxPut8(c.A.ChromaIntra[i:], 0)
				ctxPut8(c.A.UVMode[i:], 0)
			}
		}
	}
	for i := x4; i < x4+int(dims.W4); i++ {
		c.A.PaletteY[i].Size = 0
		c.A.PaletteUV[i].Size = 0
		c.A.Interp[i] = filters
	}
	switch dims.H4 {
	case 1:
		c.L.Intra[y4] = 0
		c.L.CompGroup[y4] = 0
		c.L.InterIntra[y4] = 0
		c.L.Ref[0][y4] = result.References.Ref[0]
		c.L.Ref[1][y4] = result.References.Ref[1]
		c.L.Compound[y4] = compound
		c.L.CompIndex[y4] = 1
		c.L.BlockSize[y4] = size
		c.L.InterpValid[y4] = 1
		if hasChroma {
			c.L.ChromaIntra[y4] = 0
			c.L.UVMode[y4] = 0
		}
	case 2:
		ctxPut2(c.L.Intra[y4:], zero)
		ctxPut2(c.L.CompGroup[y4:], zero)
		ctxPut2(c.L.InterIntra[y4:], zero)
		ctxPut2(c.L.Ref[0][y4:], ref0)
		ctxPut2(c.L.Ref[1][y4:], ref1)
		ctxPut2(c.L.Compound[y4:], comp)
		ctxPut2(c.L.CompIndex[y4:], one)
		ctxPut2(c.L.BlockSize[y4:], bs)
		ctxPut2(c.L.InterpValid[y4:], one)
		if hasChroma {
			ctxPut2(c.L.ChromaIntra[y4:], 0)
			ctxPut2(c.L.UVMode[y4:], 0)
		}
	case 4:
		ctxPut4(c.L.Intra[y4:], zero)
		ctxPut4(c.L.CompGroup[y4:], zero)
		ctxPut4(c.L.InterIntra[y4:], zero)
		ctxPut4(c.L.Ref[0][y4:], ref0)
		ctxPut4(c.L.Ref[1][y4:], ref1)
		ctxPut4(c.L.Compound[y4:], comp)
		ctxPut4(c.L.CompIndex[y4:], one)
		ctxPut4(c.L.BlockSize[y4:], bs)
		ctxPut4(c.L.InterpValid[y4:], one)
		if hasChroma {
			ctxPut4(c.L.ChromaIntra[y4:], 0)
			ctxPut4(c.L.UVMode[y4:], 0)
		}
	default:
		for i := y4; i < y4+int(dims.H4); i += 8 {
			ctxPut8(c.L.Intra[i:], zero)
			ctxPut8(c.L.CompGroup[i:], zero)
			ctxPut8(c.L.InterIntra[i:], zero)
			ctxPut8(c.L.Ref[0][i:], ref0)
			ctxPut8(c.L.Ref[1][i:], ref1)
			ctxPut8(c.L.Compound[i:], comp)
			ctxPut8(c.L.CompIndex[i:], one)
			ctxPut8(c.L.BlockSize[i:], bs)
			ctxPut8(c.L.InterpValid[i:], one)
			if hasChroma {
				ctxPut8(c.L.ChromaIntra[i:], 0)
				ctxPut8(c.L.UVMode[i:], 0)
			}
		}
	}
	for i := y4; i < y4+int(dims.H4); i++ {
		c.L.PaletteY[i].Size = 0
		c.L.PaletteUV[i].Size = 0
		c.L.Interp[i] = filters
	}
	if !keepMotion {
		return nil
	}
	xEnd := x4 + int(dims.W4)
	yEnd := y4 + int(dims.H4)
	for x := x4; x < xEnd; x++ {
		c.AboveInterMotion[x] = result
		c.AboveMotionValid[x] = 1
	}
	for y := y4; y < yEnd; y++ {
		c.LeftInterMotion[y] = result
		c.LeftMotionValid[y] = 1
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

// ctxSplat replicates v into every byte, the rep_macro pattern of case_set.
func ctxSplat(v uint8) uint64 { return uint64(v) * 0x0101010101010101 }

type ctxByte interface{ ~uint8 | ~int8 }

// ctxPut2, ctxPut4 and ctxPut8 store the low 2, 4 or 8 bytes of x to
// d[:2], d[:4] or d[:8], one store each.
func ctxPut2[T ctxByte](d []T, x uint64) {
	d = d[:2]
	d[0], d[1] = T(x), T(x>>8)
}

func ctxPut4[T ctxByte](d []T, x uint64) {
	d = d[:4]
	d[0], d[1], d[2], d[3] = T(x), T(x>>8), T(x>>16), T(x>>24)
}

func ctxPut8[T ctxByte](d []T, x uint64) {
	d = d[:8]
	d[0], d[1], d[2], d[3] = T(x), T(x>>8), T(x>>16), T(x>>24)
	d[4], d[5], d[6], d[7] = T(x>>32), T(x>>40), T(x>>48), T(x>>56)
}
