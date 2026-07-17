// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package transform

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

type inverseIdentity16NEONCtx struct {
	dst         *int16
	dstStride   int
	coeff       *int32
	coeffStride int
	rowGroups   int
	rowMin      int32
	rowMax      int32
	colMin      int32
	colMax      int32
}

//go:noescape
func inverseIdentity16Rows4NEONAsm(ctx *inverseIdentity16NEONCtx)

func init() {
	if cpu.Detected.NEON {
		inverseIdentity16Rows4Impl = inverseIdentity16Rows4NEON
	}
}

func inverseIdentity16Rows4NEON(dst []int16, dstStride int, coeff []int32, coeffStride int, rowGroups int, rowMin int32, rowMax int32, colMin int32, colMax int32) {
	ctx := inverseIdentity16NEONCtx{
		dst:         &dst[0],
		dstStride:   dstStride,
		coeff:       &coeff[0],
		coeffStride: coeffStride,
		rowGroups:   rowGroups,
		rowMin:      rowMin,
		rowMax:      rowMax,
		colMin:      colMin,
		colMax:      colMax,
	}
	inverseIdentity16Rows4NEONAsm(&ctx)
}
