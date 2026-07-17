// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package cdef

const U8ByteInputEnabled = true

// filterBlockU8ByteNEONCtx is shared verbatim with the G_* offsets in
// filter_u8_byte_neon_arm64.s. Keep field order and widths stable.
type filterBlockU8ByteNEONCtx struct {
	dst    *byte
	input  *byte
	dstStr int64
	height int64

	pri0 int64
	pri1 int64
	sec0 int64
	sec1 int64
	sec2 int64
	sec3 int64

	priTap0 int64
	priTap1 int64
	secTap0 int64
	secTap1 int64

	priStrength int64
	secStrength int64
	priShift    int64
	secShift    int64
}

//go:noescape
func cdefFilterBlock8PrimaryByteU8NEON(ctx *filterBlockU8ByteNEONCtx)

//go:noescape
func cdefFilterBlock8SecondaryGeneralByteU8NEON(ctx *filterBlockU8ByteNEONCtx)

//go:noescape
func cdefFilterBlock8FusedByteU8NEON(ctx *filterBlockU8ByteNEONCtx)

type filterBlockU8SecondaryByteNEONCtx struct {
	dst         *byte
	input       *byte
	dstStr      int64
	secStrength int64
	secShift    int64
}

//go:noescape
func cdefFilterBlock8SecondaryByteU8NEON(ctx *filterBlockU8SecondaryByteNEONCtx)

func filterUnitBlocksU8Byte(dst []byte, dstStride int, input []byte, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams) error {
	ctx := filterBlockU8ByteNEONCtx{
		dstStr:      int64(dstStride),
		height:      8,
		secTap0:     int64(cdefSecondaryTaps[0]),
		secTap1:     int64(cdefSecondaryTaps[1]),
		secStrength: int64(u.secondaryStrength),
		secShift:    int64(constrainShift(u.secondaryStrength, u.damping)),
	}
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
		priTaps := cdefPrimaryTaps[strength&1]
		ctx.dst = &dst[by*8*dstStride+bx*8]
		ctx.input = &input[inputOrigin+by*8*BStride+bx*8]
		ctx.pri0 = int64(cdefDirections[direction+2][0])
		ctx.pri1 = int64(cdefDirections[direction+2][1])
		ctx.sec0 = int64(cdefDirections[direction+4][0])
		ctx.sec1 = int64(cdefDirections[direction][0])
		ctx.sec2 = int64(cdefDirections[direction+4][1])
		ctx.sec3 = int64(cdefDirections[direction][1])
		ctx.priTap0 = int64(priTaps[0])
		ctx.priTap1 = int64(priTaps[1])
		ctx.priStrength = int64(strength)
		ctx.priShift = int64(constrainShift(strength, u.damping))
		switch {
		case strength != 0 && u.secondaryStrength == 0:
			cdefFilterBlock8PrimaryByteU8NEON(&ctx)
		case strength == 0 && u.secondaryStrength != 0:
			if direction == 0 {
				fixed := filterBlockU8SecondaryByteNEONCtx{
					dst:         ctx.dst,
					input:       ctx.input,
					dstStr:      ctx.dstStr,
					secStrength: ctx.secStrength,
					secShift:    ctx.secShift,
				}
				cdefFilterBlock8SecondaryByteU8NEON(&fixed)
			} else {
				cdefFilterBlock8SecondaryGeneralByteU8NEON(&ctx)
			}
		default:
			cdefFilterBlock8FusedByteU8NEON(&ctx)
		}
	}
	return nil
}
