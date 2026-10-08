// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package prediction

// High-bit-depth (10/12-bit, bytesPerSample==2) NEON PAETH/SMOOTH kernels. The
// samples are already int16-native, so these reuse the exact arithmetic of the
// 8-bit kernels in intra_neon_arm64.s (PAETH base stays within int16 for 12-bit
// inputs; the SMOOTH u32 accumulators still fit) and differ only in writing the
// uint16 results directly (st1 {v.8h}) instead of narrowing to bytes. The
// wrappers in intra_neon_arm64.go route bytesPerSample==2, width-multiple-of-8
// blocks here and keep every other shape on the pure-Go reference. See
// dav1d src/arm/64/ipred16.S for the equivalent hbd predictors.

//go:noescape
func predictPaeth16NEONAsm(ctx *paethNEONCtx)

//go:noescape
func predictSmooth16NEONAsm(ctx *smoothNEONCtx)

//go:noescape
func predictSmoothVertical16NEONAsm(ctx *smooth1DNEONCtx)

//go:noescape
func predictSmoothHorizontal16NEONAsm(ctx *smooth1DNEONCtx)
