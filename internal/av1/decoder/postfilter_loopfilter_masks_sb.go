package decoder

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/lfmask"
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

// applyLoopFilterMaskPlaneRangeSB is the mask apply in dav1d's shape
// (src/lf_apply_tmpl.c filter_plane_{cols,rows}_{y,uv}): instead of visiting
// every set 4x4 edge cell through a callback, it hands each 4-pixel column
// (vertical edges) or row (horizontal edges) of a 128x128 mask region to the
// superblock deblock kernel with the three width-class bitmasks assembled once.
// The kernel resolves levels itself and filters sixteen 8-bit (or eight
// 10/12-bit) samples per call with every width fused; bps selects the sample
// size. Units outside the plane's MI(4)-aligned crop are masked out
// of the bitmasks, which is exactly the edge-length clamp and padding-edge
// suppression the per-cell apply performs; on 8-aligned buffers no filter width
// ever needs downgrading, so the result is byte-identical.
func (ctx *FrameWorkPostFilterContext) applyLoopFilterMaskPlaneRangeSB(result *FrameWorkLoopFilterPostFilterApplyResult, masks *threading.FrameWorkLoopFilterMasks, lc lfmask.LevelCache, sharpness uint8, plane loopfilter.Plane, dirMask, rRow0, rRow1, rCol0, rCol1 int, dst frame.Plane, bps int, bitDepth uint8, bounds frameWorkLoopFilterBounds) {
	lut := loopfilter.SBLUTFor(sharpness)
	cols := masks.Cols
	pix := dst.Pix
	stride := dst.Stride
	w4 := int(bounds.posWidth+3) >> 2
	h4 := int(bounds.posHeight+3) >> 2
	cells := lc.Cells
	var st loopfilter.SBStats

	if plane == loopfilter.PlaneY {
		if dirMask&maskDirVertical != 0 {
			for rRow := rRow0; rRow < rRow1; rRow++ {
				baseRow := rRow * 32
				nRows := minInt(32, h4-baseRow)
				if nRows <= 0 {
					continue
				}
				rowMask := sb8LaneMask(nRows)
				for rCol := rCol0; rCol < rCol1; rCol++ {
					baseCol := rCol * 32
					nCols := minInt(32, w4-baseCol)
					if nCols <= 0 {
						continue
					}
					region := &masks.Masks[rRow*masks.SB128W+rCol]
					rowOff := baseRow * 4 * stride
					lvlRow := baseRow * cols
					for pos := 0; pos < nCols; pos++ {
						col := baseCol + pos
						if col == 0 {
							continue
						}
						m := &region.Y[0][pos]
						vm := [3]uint32{
							(uint32(m[0][0]) | uint32(m[0][1])<<16) & rowMask,
							(uint32(m[1][0]) | uint32(m[1][1])<<16) & rowMask,
							(uint32(m[2][0]) | uint32(m[2][1])<<16) & rowMask,
						}
						if vm[0]|vm[1]|vm[2] == 0 {
							continue
						}
						st.Add(sbFilterVertical(bps, bitDepth, pix, rowOff+col*4*bps, stride, (dst.Width-col*4)*bps, vm, true, cells, lvlRow+col, cols, 0, lut))
					}
				}
			}
		}
		if dirMask&maskDirHorizontal != 0 {
			for rRow := rRow0; rRow < rRow1; rRow++ {
				baseRow := rRow * 32
				nRows := minInt(32, h4-baseRow)
				if nRows <= 0 {
					continue
				}
				for rCol := rCol0; rCol < rCol1; rCol++ {
					baseCol := rCol * 32
					nCols := minInt(32, w4-baseCol)
					if nCols <= 0 {
						continue
					}
					colMask := sb8LaneMask(nCols)
					region := &masks.Masks[rRow*masks.SB128W+rCol]
					rowBytes := (dst.Width - baseCol*4) * bps
					for pos := 0; pos < nRows; pos++ {
						row := baseRow + pos
						if row == 0 {
							continue
						}
						m := &region.Y[1][pos]
						vm := [3]uint32{
							(uint32(m[0][0]) | uint32(m[0][1])<<16) & colMask,
							(uint32(m[1][0]) | uint32(m[1][1])<<16) & colMask,
							(uint32(m[2][0]) | uint32(m[2][1])<<16) & colMask,
						}
						if vm[0]|vm[1]|vm[2] == 0 {
							continue
						}
						st.Add(sbFilterHorizontal(bps, bitDepth, pix, row*4*stride+baseCol*4*bps, stride, rowBytes, vm, true, cells, row*cols+baseCol, cols, 1, lut))
					}
				}
			}
		}
		frameWorkCountAppliedLoopFilterSB(result, plane, st)
		return
	}

	ssHor := masks.Layout.SSHor
	ssVer := masks.Layout.SSVer
	regionCW := 32 >> ssHor
	regionCH := 32 >> ssVer
	laneBitsV := uint(16 >> ssVer)
	laneBitsH := uint(16 >> ssHor)
	comp := 2
	if plane == loopfilter.PlaneV {
		comp = 3
	}
	if dirMask&maskDirVertical != 0 {
		for rRow := rRow0; rRow < rRow1; rRow++ {
			baseCRow := rRow * regionCH
			nCRows := minInt(regionCH, h4-baseCRow)
			if nCRows <= 0 {
				continue
			}
			rowMask := sb8LaneMask(nCRows)
			for rCol := rCol0; rCol < rCol1; rCol++ {
				baseCCol := rCol * regionCW
				nCCols := minInt(regionCW, w4-baseCCol)
				if nCCols <= 0 {
					continue
				}
				region := &masks.Masks[rRow*masks.SB128W+rCol]
				rowOff := baseCRow * 4 * stride
				lvlRow := baseCRow * cols
				for pos := 0; pos < nCCols; pos++ {
					ccol := baseCCol + pos
					if ccol == 0 {
						continue
					}
					m := &region.UV[0][pos]
					vm := [3]uint32{
						(uint32(m[0][0]) | uint32(m[0][1])<<laneBitsV) & rowMask,
						(uint32(m[1][0]) | uint32(m[1][1])<<laneBitsV) & rowMask,
						0,
					}
					if vm[0]|vm[1] == 0 {
						continue
					}
					st.Add(sbFilterVertical(bps, bitDepth, pix, rowOff+ccol*4*bps, stride, (dst.Width-ccol*4)*bps, vm, false, cells, lvlRow+ccol, cols, comp, lut))
				}
			}
		}
	}
	if dirMask&maskDirHorizontal != 0 {
		for rRow := rRow0; rRow < rRow1; rRow++ {
			baseCRow := rRow * regionCH
			nCRows := minInt(regionCH, h4-baseCRow)
			if nCRows <= 0 {
				continue
			}
			for rCol := rCol0; rCol < rCol1; rCol++ {
				baseCCol := rCol * regionCW
				nCCols := minInt(regionCW, w4-baseCCol)
				if nCCols <= 0 {
					continue
				}
				colMask := sb8LaneMask(nCCols)
				region := &masks.Masks[rRow*masks.SB128W+rCol]
				rowBytes := (dst.Width - baseCCol*4) * bps
				for pos := 0; pos < nCRows; pos++ {
					crow := baseCRow + pos
					if crow == 0 {
						continue
					}
					m := &region.UV[1][pos]
					vm := [3]uint32{
						(uint32(m[0][0]) | uint32(m[0][1])<<laneBitsH) & colMask,
						(uint32(m[1][0]) | uint32(m[1][1])<<laneBitsH) & colMask,
						0,
					}
					if vm[0]|vm[1] == 0 {
						continue
					}
					st.Add(sbFilterHorizontal(bps, bitDepth, pix, crow*4*stride+baseCCol*4*bps, stride, rowBytes, vm, false, cells, crow*cols+baseCCol, cols, comp, lut))
				}
			}
		}
	}
	frameWorkCountAppliedLoopFilterSB(result, plane, st)
}

// sb8LaneMask selects the first n (1..32) units of a 32-unit mask lane.
func sb8LaneMask(n int) uint32 {
	if n >= 32 {
		return ^uint32(0)
	}
	return uint32(1)<<uint(n) - 1
}

// frameWorkCountAppliedLoopFilterSB folds a superblock apply's counts into the
// result exactly as per-cell frameWorkCountAppliedLoopFilterMaskEdge calls would.
func frameWorkCountAppliedLoopFilterSB(result *FrameWorkLoopFilterPostFilterApplyResult, plane loopfilter.Plane, st loopfilter.SBStats) {
	if st.Applied == 0 {
		return
	}
	result.Edges += st.Applied
	result.PlaneEdges[plane] += st.Applied
	result.Applied += st.Applied
	result.PlaneApplied[plane] += st.Applied
	if st.MaxLevel > result.PlaneMaxLevel[plane] {
		result.PlaneMaxLevel[plane] = st.MaxLevel
	}
	if st.MaxLevel > result.MaxLevel {
		result.MaxLevel = st.MaxLevel
	}
}

// sbFilterVertical dispatches one superblock column to the 8-bit or the
// high-bit-depth kernel.
func sbFilterVertical(bps int, bitDepth uint8, pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *loopfilter.SBLUT) loopfilter.SBStats {
	if bps == 1 {
		return loopfilter.FilterSBVertical8(pix, off, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
	}
	return loopfilter.FilterSBVertical16(pix, off, stride, rowBytes, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
}

// sbFilterHorizontal dispatches one superblock row to the 8-bit or the
// high-bit-depth kernel.
func sbFilterHorizontal(bps int, bitDepth uint8, pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *loopfilter.SBLUT) loopfilter.SBStats {
	if bps == 1 {
		return loopfilter.FilterSBHorizontal8(pix, off, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
	}
	return loopfilter.FilterSBHorizontal16(pix, off, stride, rowBytes, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
}
