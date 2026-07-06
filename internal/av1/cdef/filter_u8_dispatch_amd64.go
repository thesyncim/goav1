// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package cdef

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the AVX2-backed 8-bit-dst CDEF block filter on amd64. The kernel
// keeps the proven uint16 tap math from filter_avx2_amd64.s and applies the
// dav1d 8bpc store shape (src/x86/cdef_avx2.asm cdef_filter_*_8bpc:
// packuswb + byte stores) directly to goav1's uint8 frame plane.
func init() {
	_ = cpu.Detected
	if cpu.Detected.AVX2 {
		filterBlockU8Impl = filterBlockU8AVX2
		return
	}
	filterBlockU8Impl = filterBlockU8PureGo
}

// filterBlockU8AVX2Ctx is the asm calling context for
// filter_u8_avx2_amd64.s. Field order and sizes are part of the ABI shared
// with the asm; do not reorder.
type filterBlockU8AVX2Ctx struct {
	dst    *byte
	input  *uint16 // pointer to input[inputOrigin]
	dstStr uintptr // dst stride in bytes
	height uintptr

	pri0 int64 // direction offsets in input elements (signed)
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

	enablePrimary   int64
	enableSecondary int64
	clipping        int64
	width           uintptr
}

//go:noescape
func filterBlockU8AVX2Asm(ctx *filterBlockU8AVX2Ctx)

func filterBlockU8AVX2(dst []byte, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) {
	width := int(params.Width)
	if width != 8 && width != 4 {
		filterBlockU8PureGo(dst, dstStride, dstOrigin, input, inputOrigin, params)
		return
	}
	primaryStrength := int(params.PrimaryStrength)
	secondaryStrength := int(params.SecondaryStrength)
	direction := int(params.Direction)
	primaryDamping := int(params.PrimaryDamping)
	secondaryDamping := int(params.SecondaryDamping)
	coeffShift := int(params.CoeffShift)
	enablePrimary := primaryStrength != 0
	enableSecondary := secondaryStrength != 0
	clippingRequired := enablePrimary && enableSecondary
	priTaps := cdefPrimaryTaps[(primaryStrength>>coeffShift)&1]

	ctx := filterBlockU8AVX2Ctx{
		dst:    &dst[dstOrigin],
		input:  &input[inputOrigin],
		dstStr: uintptr(dstStride),
		height: uintptr(params.Height),

		pri0: int64(cdefDirections[direction+2][0]),
		pri1: int64(cdefDirections[direction+2][1]),
		sec0: int64(cdefDirections[direction+4][0]),
		sec1: int64(cdefDirections[direction+4][1]),
		sec2: int64(cdefDirections[direction][0]),
		sec3: int64(cdefDirections[direction][1]),

		priTap0: int64(priTaps[0]),
		priTap1: int64(priTaps[1]),
		secTap0: int64(cdefSecondaryTaps[0]),
		secTap1: int64(cdefSecondaryTaps[1]),

		priStrength: int64(primaryStrength),
		secStrength: int64(secondaryStrength),
		priShift:    int64(constrainShift(primaryStrength, primaryDamping)),
		secShift:    int64(constrainShift(secondaryStrength, secondaryDamping)),

		enablePrimary:   boolToInt64(enablePrimary),
		enableSecondary: boolToInt64(enableSecondary),
		clipping:        boolToInt64(clippingRequired),
		width:           uintptr(width),
	}
	filterBlockU8AVX2Asm(&ctx)
}
