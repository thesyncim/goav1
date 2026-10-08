// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && !goexperiment.simd

package transform

// The default arm64 build uses the retained hybrid NEON kernels. IDTX has no
// assembly implementation after its measured Go SIMD replacement, so the
// default build uses the scalar reference.
var forwardBlock8x8ADSTDCTImpl = forwardBlock8x8ADSTDCTNEONGuarded
var forwardBlock8x8DCTADSTImpl = forwardBlock8x8DCTADSTNEONGuarded
var forwardBlock8x8ADSTADSTImpl = forwardBlock8x8ADSTADSTNEONGuarded
var forwardBlock8x8IDTXImpl = forwardBlock8x8IDTXPureGo
