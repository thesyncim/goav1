// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && goexperiment.simd

package cdef

// The measured uint16 Go SIMD block filter loses to the NEON kernel, so keep
// the architecture dispatch on NEON when the SIMD experiment is enabled.
func init() {
	filterBlockImpl = filterBlockNEON
}
