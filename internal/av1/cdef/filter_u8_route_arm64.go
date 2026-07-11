// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && !goexperiment.simd

package cdef

// dispatchFilterBlockU8NEON routes a prepared ctx to the width- and
// strength-specialized NEON kernel, mirroring dispatchFilterBlockNEON (dav1d's
// pri/sec/pri_sec split of src/arm/64/cdef.S). Under the goexperiment.simd
// build the secondary-only cases route to the Go-native SIMD kernels instead
// (filter_u8_gosimd_arm64.go); this file is excluded there.
func dispatchFilterBlockU8NEON(ctx *filterBlockU8NEONCtx, width int, primaryStrength int, secondaryStrength int) {
	if width == 8 {
		switch {
		case primaryStrength != 0 && secondaryStrength == 0:
			cdefFilterBlock8PrimaryU8NEON(ctx)
		case primaryStrength == 0 && secondaryStrength != 0:
			cdefFilterBlock8SecondaryU8NEON(ctx)
		default:
			cdefFilterBlock8U8NEON(ctx)
		}
		return
	}
	switch {
	case primaryStrength != 0 && secondaryStrength == 0:
		cdefFilterBlock4PrimaryU8NEON(ctx)
	case primaryStrength == 0 && secondaryStrength != 0:
		cdefFilterBlock4SecondaryU8NEON(ctx)
	default:
		cdefFilterBlock4U8NEON(ctx)
	}
}
