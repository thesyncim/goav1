// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package prediction

// Width-4 (8-bit) NEON PAETH/SMOOTH kernels. TX_4X4 is the most frequent block
// shape on all-intra content, so the width-4 scalar fallback is hot. Following
// dav1d src/arm/64/ipred.S, the whole block is produced in one asm call that
// packs two output rows (4 columns each) per 8-lane vector, amortising the call
// overhead. The arithmetic is identical to the 8-wide kernels; the wrappers in
// intra_neon_arm64.go route width==4 blocks with an even height here (an odd
// visible height at a frame edge falls back to the pure-Go reference so the
// two-rows-per-iteration store never runs past the block).

//go:noescape
func predictPaethW4NEONAsm(ctx *paethNEONCtx)

//go:noescape
func predictSmoothW4NEONAsm(ctx *smoothNEONCtx)

//go:noescape
func predictSmoothVerticalW4NEONAsm(ctx *smooth1DNEONCtx)

//go:noescape
func predictSmoothHorizontalW4NEONAsm(ctx *smooth1DNEONCtx)
