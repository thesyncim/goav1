// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// int16 8-wide Go-native SIMD CDEF block filter. Eight columns are processed
// per Int16x8. constrainShifted becomes AbsDiff + shift + clamp + branchless
// copysign; the weighted tap sum stays in int16 (max ~5760 even at 12-bit).
// Byte-identical to filterBlockPureGo. Widths not a multiple of 8 (4x4 chroma)
// fall back to the scalar reference.
//
// This is a Go-native SIMD reference kernel: it is byte-exact and ~7x over the
// scalar path, but ~18% behind the specialized NEON asm (filterBlockNEON),
// which is what the arm64 dispatch uses. The gap is Go register-allocation
// overhead on this constant/shuffle-heavy kernel, not the algorithm. Kept for
// the pure-Go-SIMD build and as a portable fallback; not wired ahead of the asm.
//
// The tap sum and min/max use two independent accumulators each so the 12
// per-pixel accumulations pipeline instead of forming one long dependency
// chain. cdefLoad16/cdefConstrain are package-level so they inline (closures
// here compiled to real calls that pushed SIMD vectors through the stack).

package cdef

import (
	"simd/archsimd"
	"unsafe"
)

// init binds the Go-native SIMD CDEF block filter as the dispatch kernel under
// the goexperiment.simd build (the NEON asm binding in filter_dispatch_arm64.go
// is excluded there via !goexperiment.simd). Byte-identical to the pure-Go and
// NEON references.
func init() {
	filterBlockImpl = filterBlockSIMD16
}

// cdefLoad16 loads 8 uint16 samples at input[i] as int16 without a lane-type
// convert (CDEF samples <= 12-bit; the bit patterns are identical).
func cdefLoad16(input []uint16, i int) archsimd.Int16x8 {
	return archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&input[i])))
}

// cdefConstrain is constrainShifted 8-wide:
// copysign(clamp(str-(|n-x|>>shift), 0, |n-x|), n-x).
func cdefConstrain(n, x, strV, zeroV archsimd.Int16x8, shift uint64) archsimd.Int16x8 {
	diff := n.Sub(x)
	a := n.AbsDiff(x)
	lim := strV.Sub(a.ShiftAllRight(shift)).Max(zeroV).Min(a)
	sign := diff.ShiftAllRight(15)
	return lim.Xor(sign).Sub(sign)
}

func filterBlockSIMD16(dst []uint16, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) {
	width := int(params.Width)
	if width%8 != 0 {
		filterBlockPureGo(dst, dstStride, dstOrigin, input, inputOrigin, params)
		return
	}
	primaryStrength := int(params.PrimaryStrength)
	secondaryStrength := int(params.SecondaryStrength)
	direction := int(params.Direction)
	primaryDamping := int(params.PrimaryDamping)
	secondaryDamping := int(params.SecondaryDamping)
	coeffShift := int(params.CoeffShift)
	height := int(params.Height)

	enablePrimary := primaryStrength != 0
	enableSecondary := secondaryStrength != 0
	clippingRequired := enablePrimary && enableSecondary
	priTaps := cdefPrimaryTaps[(primaryStrength>>coeffShift)&1]

	pri0 := int(cdefDirections[direction+2][0])
	pri1 := int(cdefDirections[direction+2][1])
	sec0a := int(cdefDirections[direction+4][0])
	sec0b := int(cdefDirections[direction+4][1])
	sec1a := int(cdefDirections[direction][0])
	sec1b := int(cdefDirections[direction][1])
	priShift := uint64(constrainShift(primaryStrength, primaryDamping))
	secShift := uint64(constrainShift(secondaryStrength, secondaryDamping))

	priStrV := archsimd.BroadcastInt16x8(int16(primaryStrength))
	secStrV := archsimd.BroadcastInt16x8(int16(secondaryStrength))
	priTap0V := archsimd.BroadcastInt16x8(int16(priTaps[0]))
	priTap1V := archsimd.BroadcastInt16x8(int16(priTaps[1]))
	eight := archsimd.BroadcastInt16x8(8)
	zeroV := archsimd.BroadcastInt16x8(0)

	for row := 0; row < height; row++ {
		rowBase := inputOrigin + row*BStride
		dstRow := dstOrigin + row*dstStride
		for col := 0; col < width; col += 8 {
			base := rowBase + col
			x := cdefLoad16(input, base)
			// Two independent accumulators per reduction so the chains pipeline.
			s0, s1 := x.Sub(x), x.Sub(x)
			mx0, mx1 := x, x
			mn0, mn1 := x, x
			if enablePrimary {
				a := cdefLoad16(input, base+pri0)
				b := cdefLoad16(input, base-pri0)
				s0 = s0.Add(cdefConstrain(a, x, priStrV, zeroV, priShift).Mul(priTap0V))
				s1 = s1.Add(cdefConstrain(b, x, priStrV, zeroV, priShift).Mul(priTap0V))
				mx0, mx1 = mx0.Max(a), mx1.Max(b)
				mn0, mn1 = mn0.Min(a), mn1.Min(b)
				a = cdefLoad16(input, base+pri1)
				b = cdefLoad16(input, base-pri1)
				s0 = s0.Add(cdefConstrain(a, x, priStrV, zeroV, priShift).Mul(priTap1V))
				s1 = s1.Add(cdefConstrain(b, x, priStrV, zeroV, priShift).Mul(priTap1V))
				mx0, mx1 = mx0.Max(a), mx1.Max(b)
				mn0, mn1 = mn0.Min(a), mn1.Min(b)
			}
			if enableSecondary {
				a := cdefLoad16(input, base+sec0a)
				b := cdefLoad16(input, base-sec0a)
				s0 = s0.Add(cdefConstrain(a, x, secStrV, zeroV, secShift).ShiftAllLeft(1))
				s1 = s1.Add(cdefConstrain(b, x, secStrV, zeroV, secShift).ShiftAllLeft(1))
				mx0, mx1 = mx0.Max(a), mx1.Max(b)
				mn0, mn1 = mn0.Min(a), mn1.Min(b)
				a = cdefLoad16(input, base+sec1a)
				b = cdefLoad16(input, base-sec1a)
				s0 = s0.Add(cdefConstrain(a, x, secStrV, zeroV, secShift).ShiftAllLeft(1))
				s1 = s1.Add(cdefConstrain(b, x, secStrV, zeroV, secShift).ShiftAllLeft(1))
				mx0, mx1 = mx0.Max(a), mx1.Max(b)
				mn0, mn1 = mn0.Min(a), mn1.Min(b)
				a = cdefLoad16(input, base+sec0b)
				b = cdefLoad16(input, base-sec0b)
				s0 = s0.Add(cdefConstrain(a, x, secStrV, zeroV, secShift))
				s1 = s1.Add(cdefConstrain(b, x, secStrV, zeroV, secShift))
				mx0, mx1 = mx0.Max(a), mx1.Max(b)
				mn0, mn1 = mn0.Min(a), mn1.Min(b)
				a = cdefLoad16(input, base+sec1b)
				b = cdefLoad16(input, base-sec1b)
				s0 = s0.Add(cdefConstrain(a, x, secStrV, zeroV, secShift))
				s1 = s1.Add(cdefConstrain(b, x, secStrV, zeroV, secShift))
				mx0, mx1 = mx0.Max(a), mx1.Max(b)
				mn0, mn1 = mn0.Min(a), mn1.Min(b)
			}
			sum := s0.Add(s1)
			// y = x + ((8 + sum - (sum<0)) >> 4)
			neg := sum.ShiftAllRight(15)
			y := x.Add(eight.Add(sum).Add(neg).ShiftAllRight(4))
			if clippingRequired {
				y = y.Max(mn0.Min(mn1)).Min(mx0.Max(mx1))
			}
			y.StoreArray((*[8]int16)(unsafe.Pointer(&dst[dstRow+col])))
		}
	}
}
