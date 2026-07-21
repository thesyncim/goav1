// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package cdef

// FilterFrameBlocksU8ByteTrusted is the complete-real-halo counterpart of
// FilterFrameBlocksU8Trusted. It keeps the interior unit tap window in bytes
// and therefore avoids both staging-time widening and half of the filter
// kernel's source traffic. Frame-boundary units retain the uint16 sentinel
// path because VeryLarge is intentionally not representable here.
func FilterFrameBlocksU8ByteTrusted(dst []byte, dstStride int, input []byte, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, params FrameFilterParams) error {
	if len(blocks) == 0 {
		return nil
	}
	if params.CoeffShift != 0 {
		return ErrInvalidCDEF
	}
	primaryStrength := int(params.Level)
	secondaryStrength := int(params.SecondaryStrength)
	damping := int(params.Damping)
	if params.Plane != PlaneY {
		damping--
	}
	if damping < 0 {
		return ErrInvalidCDEF
	}
	xDec := int(params.XDec)
	yDec := int(params.YDec)
	bwLog2 := 3 - xDec
	bhLog2 := 3 - yDec

	if params.Plane == PlaneY {
		for i := 0; i < len(blocks); {
			block := blocks[i]
			by := int(block.BY)
			bx := int(block.BX)
			srcOrigin := ((by * dstStride) << bhLog2) + (bx << bwLog2)
			if bwLog2 == 3 && i+1 < len(blocks) && blocks[i+1].BY == block.BY && blocks[i+1].BX == block.BX+1 {
				dir1, variance1, dir2, variance2 := findDirectionDualU8Unchecked(dst[srcOrigin:], dst[srcOrigin+8:], dstStride)
				directions[by][bx] = uint8(dir1)
				variances[by][bx] = variance1
				directions[by][bx+1] = uint8(dir2)
				variances[by][bx+1] = variance2
				i += 2
				continue
			}
			dir, variance := findDirectionU8Unchecked(dst[srcOrigin:], dstStride)
			directions[by][bx] = uint8(dir)
			variances[by][bx] = variance
			i++
		}
	}
	if params.Plane == PlaneU && params.XDec != params.YDec {
		if !convertChromaDirections(blocks, directions, xDec) {
			return ErrInvalidCDEF
		}
	}
	if primaryStrength == 0 && secondaryStrength == 0 {
		return nil
	}
	return filterUnitBlocksU8Byte(dst, dstStride, input, inputOrigin, blocks, directions, variances, unitFilterParams{
		primaryStrength:   primaryStrength,
		secondaryStrength: secondaryStrength,
		damping:           damping,
		bwLog2:            bwLog2,
		bhLog2:            bhLog2,
		blockWidth:        1 << bwLog2,
		blockHeight:       1 << bhLog2,
		lumaAdjust:        params.Plane == PlaneY,
	})
}

// FilterFrameBlocksU8ByteTrustedNoDirection is the byte-input form of
// FilterFrameBlocksU8TrustedNoDirection.
func FilterFrameBlocksU8ByteTrustedNoDirection(dst []byte, dstStride int, input []byte, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, params FrameFilterParams) error {
	if len(blocks) == 0 {
		return nil
	}
	if params.Plane != PlaneY || params.Level != 0 || params.CoeffShift != 0 || params.XDec != 0 || params.YDec != 0 {
		return ErrInvalidCDEF
	}
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		directions[by][bx] = 0
		variances[by][bx] = 0
	}
	secondaryStrength := int(params.SecondaryStrength)
	if secondaryStrength == 0 {
		return nil
	}
	return filterUnitBlocksU8Byte(dst, dstStride, input, inputOrigin, blocks, directions, variances, unitFilterParams{
		secondaryStrength: secondaryStrength,
		damping:           int(params.Damping),
		bwLog2:            3,
		bhLog2:            3,
		blockWidth:        8,
		blockHeight:       8,
	})
}

// filterBlockU8SecondaryBytePureGo is the canonical byte-input kernel for the
// fixed direction-zero secondary filter. Every tap is a real 8-bit pixel;
// frame-boundary units continue to use filterBlockU8PureGo and its uint16
// sentinel buffer.
func filterBlockU8SecondaryBytePureGo(dst []byte, dstStride int, dstOrigin int, input []byte, inputOrigin int, secondaryStrength int, damping int) {
	shift := constrainShift(secondaryStrength, damping)
	for row := 0; row < 8; row++ {
		base := inputOrigin + row*BStride
		dstRow := dstOrigin + row*dstStride
		for col := 0; col < 8; col++ {
			x := int(input[base+col])
			sum := 0
			sum += 2 * constrainShifted(int(input[base+col+1])-x, secondaryStrength, shift)
			sum += 2 * constrainShifted(int(input[base+col-1])-x, secondaryStrength, shift)
			sum += 2 * constrainShifted(int(input[base+col+BStride])-x, secondaryStrength, shift)
			sum += 2 * constrainShifted(int(input[base+col-BStride])-x, secondaryStrength, shift)
			sum += constrainShifted(int(input[base+col+2])-x, secondaryStrength, shift)
			sum += constrainShifted(int(input[base+col-2])-x, secondaryStrength, shift)
			sum += constrainShifted(int(input[base+col+2*BStride])-x, secondaryStrength, shift)
			sum += constrainShifted(int(input[base+col-2*BStride])-x, secondaryStrength, shift)
			dst[dstRow+col] = byte(x + ((8 + sum - boolToInt(sum < 0)) >> 4))
		}
	}
}

func filterBlockU8BytePureGo(dst []byte, dstStride int, dstOrigin int, input []byte, inputOrigin int, params BlockFilterParams) {
	primaryStrength := int(params.PrimaryStrength)
	secondaryStrength := int(params.SecondaryStrength)
	direction := int(params.Direction)
	primaryDamping := int(params.PrimaryDamping)
	secondaryDamping := int(params.SecondaryDamping)
	priTaps := cdefPrimaryTaps[primaryStrength&1]
	pri0 := int(cdefDirections[direction+2][0])
	pri1 := int(cdefDirections[direction+2][1])
	sec0a := int(cdefDirections[direction+4][0])
	sec0b := int(cdefDirections[direction+4][1])
	sec1a := int(cdefDirections[direction][0])
	sec1b := int(cdefDirections[direction][1])
	priShift := constrainShift(primaryStrength, primaryDamping)
	secShift := constrainShift(secondaryStrength, secondaryDamping)
	clipping := primaryStrength != 0 && secondaryStrength != 0
	for row := 0; row < int(params.Height); row++ {
		base := inputOrigin + row*BStride
		dstRow := dstOrigin + row*dstStride
		for col := 0; col < int(params.Width); col++ {
			i := base + col
			x := int(input[i])
			sum := 0
			minSample, maxSample := x, x
			if primaryStrength != 0 {
				p0, p1 := int(input[i+pri0]), int(input[i-pri0])
				p2, p3 := int(input[i+pri1]), int(input[i-pri1])
				sum += int(priTaps[0]) * (constrainShifted(p0-x, primaryStrength, priShift) + constrainShifted(p1-x, primaryStrength, priShift))
				sum += int(priTaps[1]) * (constrainShifted(p2-x, primaryStrength, priShift) + constrainShifted(p3-x, primaryStrength, priShift))
				if clipping {
					minSample = min4(minSample, p0, p1, p2, p3)
					maxSample = maxClip(maxSample, p0, p1, p2, p3)
				}
			}
			if secondaryStrength != 0 {
				s0, s1 := int(input[i+sec0a]), int(input[i-sec0a])
				s2, s3 := int(input[i+sec1a]), int(input[i-sec1a])
				s4, s5 := int(input[i+sec0b]), int(input[i-sec0b])
				s6, s7 := int(input[i+sec1b]), int(input[i-sec1b])
				sum += 2 * (constrainShifted(s0-x, secondaryStrength, secShift) + constrainShifted(s1-x, secondaryStrength, secShift) + constrainShifted(s2-x, secondaryStrength, secShift) + constrainShifted(s3-x, secondaryStrength, secShift))
				sum += constrainShifted(s4-x, secondaryStrength, secShift) + constrainShifted(s5-x, secondaryStrength, secShift) + constrainShifted(s6-x, secondaryStrength, secShift) + constrainShifted(s7-x, secondaryStrength, secShift)
				if clipping {
					minSample = min4(minSample, s0, s1, s2, s3)
					minSample = min4(minSample, s4, s5, s6, s7)
					maxSample = maxClip(maxSample, s0, s1, s2, s3)
					maxSample = maxClip(maxSample, s4, s5, s6, s7)
				}
			}
			y := x + ((8 + sum - boolToInt(sum < 0)) >> 4)
			if clipping {
				y = clampInt(y, minSample, maxSample)
			}
			dst[dstRow+col] = byte(y)
		}
	}
}

func filterUnitBlocksU8BytePureGo(dst []byte, dstStride int, input []byte, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams) error {
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		strength := u.primaryStrength
		if u.lumaAdjust {
			strength = adjustStrength(strength, variances[by][bx])
		}
		if strength == 0 && u.secondaryStrength == 0 {
			continue
		}
		direction := 0
		if u.primaryStrength != 0 {
			direction = int(directions[by][bx])
		}
		filterBlockU8BytePureGo(dst, dstStride, (by<<u.bhLog2)*dstStride+(bx<<u.bwLog2), input, inputOrigin+((by*BStride)<<u.bhLog2)+(bx<<u.bwLog2), BlockFilterParams{
			PrimaryStrength:   uint8(strength),
			SecondaryStrength: uint8(u.secondaryStrength),
			Direction:         uint8(direction),
			PrimaryDamping:    uint8(u.damping),
			SecondaryDamping:  uint8(u.damping),
			Width:             uint8(u.blockWidth),
			Height:            uint8(u.blockHeight),
		})
	}
	return nil
}
