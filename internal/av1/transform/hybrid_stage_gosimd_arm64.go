// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// Go-native SIMD staging for the inverse-transform row pass: transpose the
// column-major dequant coefficient block into row-major scratch lines in 4x4
// int32 blocks, clamping to [lo, hi] (and applying the rectangular-transform
// sqrt2 scale first when rect2 is set), byte-identical with
// stageTransposeClampScalar for every int32 input.
//
// rect2 exactness without an input-envelope assumption: the scalar reference
// computes clipRange(clipInt32((int64(v)*181+128)>>8), lo, hi). rect2Scale is
// monotonic, and scale(±1<<22) = ±2966791-ish already lies outside every row
// clamp range the callers pass (bd+8 bits, so |lo|,|hi| < 1<<20 — guarded
// below), so pre-clamping v to ±1<<22 leaves the post-clamp result identical
// while keeping v*181+128 inside int32 (2^22*181 < 2^30). Inputs beyond the
// pre-clamp saturate to the same lo/hi the scalar path reaches through
// clipInt32.

package transform

import "simd/archsimd"

// stageRect2SafeBound is the row-clamp magnitude bound under which the rect2
// pre-clamp argument above holds: rect2Scale(1<<22) = (1<<22*181+128)>>8 =
// 2966528+, and every |lo|,|hi| below this bound clamps saturated inputs
// identically to the scalar clipInt32 path.
const stageRect2SafeBound = 2966528

// stageTransposeClamp is the SIMD staging kernel; shapes below one 4x4 block
// (and rect2 calls with out-of-contract clamp bounds, which production never
// produces) fall back to the scalar reference.
func stageTransposeClamp(scratch []int32, width int, coeff []int32, coeffStride int, rows int, cols int, rect2 bool, lo int32, hi int32) {
	if rows < 4 || cols < 4 || (rect2 && (hi >= stageRect2SafeBound || lo <= -stageRect2SafeBound)) {
		stageTransposeClampScalar(scratch, width, coeff, coeffStride, rows, cols, rect2, lo, hi)
		return
	}
	loV := archsimd.BroadcastInt32x4(lo)
	hiV := archsimd.BroadcastInt32x4(hi)
	rowsBlk := rows &^ 3
	colsBlk := cols &^ 3
	if rect2 {
		preLo := archsimd.BroadcastInt32x4(-(1 << 22))
		preHi := archsimd.BroadcastInt32x4(1 << 22)
		k181 := archsimd.BroadcastInt32x4(181)
		r128 := archsimd.BroadcastInt32x4(128)
		for r := 0; r < rowsBlk; r += 4 {
			for c := 0; c < colsBlk; c += 4 {
				base := c*coeffStride + r
				v0 := archsimd.LoadInt32x4Array((*[4]int32)(coeff[base:]))
				v1 := archsimd.LoadInt32x4Array((*[4]int32)(coeff[base+coeffStride:]))
				v2 := archsimd.LoadInt32x4Array((*[4]int32)(coeff[base+2*coeffStride:]))
				v3 := archsimd.LoadInt32x4Array((*[4]int32)(coeff[base+3*coeffStride:]))
				v0 = v0.Max(preLo).Min(preHi).MulAdd(k181, r128).ShiftAllRightConst(8).Max(loV).Min(hiV)
				v1 = v1.Max(preLo).Min(preHi).MulAdd(k181, r128).ShiftAllRightConst(8).Max(loV).Min(hiV)
				v2 = v2.Max(preLo).Min(preHi).MulAdd(k181, r128).ShiftAllRightConst(8).Max(loV).Min(hiV)
				v3 = v3.Max(preLo).Min(preHi).MulAdd(k181, r128).ShiftAllRightConst(8).Max(loV).Min(hiV)
				// In-register 4x4 int32 transpose: vN holds column c+N rows
				// r..r+3; tN holds row r+N cols c..c+3.
				e0 := v0.InterleaveLo(v1)
				e1 := v0.InterleaveHi(v1)
				e2 := v2.InterleaveLo(v3)
				e3 := v2.InterleaveHi(v3)
				t0 := fdctInt64AsInt32(fdctInt32AsInt64(e0).InterleaveLo(fdctInt32AsInt64(e2)))
				t1 := fdctInt64AsInt32(fdctInt32AsInt64(e0).InterleaveHi(fdctInt32AsInt64(e2)))
				t2 := fdctInt64AsInt32(fdctInt32AsInt64(e1).InterleaveLo(fdctInt32AsInt64(e3)))
				t3 := fdctInt64AsInt32(fdctInt32AsInt64(e1).InterleaveHi(fdctInt32AsInt64(e3)))
				t0.StoreArray((*[4]int32)(scratch[(r+0)*width+c:]))
				t1.StoreArray((*[4]int32)(scratch[(r+1)*width+c:]))
				t2.StoreArray((*[4]int32)(scratch[(r+2)*width+c:]))
				t3.StoreArray((*[4]int32)(scratch[(r+3)*width+c:]))
			}
		}
	} else {
		for r := 0; r < rowsBlk; r += 4 {
			for c := 0; c < colsBlk; c += 4 {
				base := c*coeffStride + r
				v0 := archsimd.LoadInt32x4Array((*[4]int32)(coeff[base:])).Max(loV).Min(hiV)
				v1 := archsimd.LoadInt32x4Array((*[4]int32)(coeff[base+coeffStride:])).Max(loV).Min(hiV)
				v2 := archsimd.LoadInt32x4Array((*[4]int32)(coeff[base+2*coeffStride:])).Max(loV).Min(hiV)
				v3 := archsimd.LoadInt32x4Array((*[4]int32)(coeff[base+3*coeffStride:])).Max(loV).Min(hiV)
				e0 := v0.InterleaveLo(v1)
				e1 := v0.InterleaveHi(v1)
				e2 := v2.InterleaveLo(v3)
				e3 := v2.InterleaveHi(v3)
				t0 := fdctInt64AsInt32(fdctInt32AsInt64(e0).InterleaveLo(fdctInt32AsInt64(e2)))
				t1 := fdctInt64AsInt32(fdctInt32AsInt64(e0).InterleaveHi(fdctInt32AsInt64(e2)))
				t2 := fdctInt64AsInt32(fdctInt32AsInt64(e1).InterleaveLo(fdctInt32AsInt64(e3)))
				t3 := fdctInt64AsInt32(fdctInt32AsInt64(e1).InterleaveHi(fdctInt32AsInt64(e3)))
				t0.StoreArray((*[4]int32)(scratch[(r+0)*width+c:]))
				t1.StoreArray((*[4]int32)(scratch[(r+1)*width+c:]))
				t2.StoreArray((*[4]int32)(scratch[(r+2)*width+c:]))
				t3.StoreArray((*[4]int32)(scratch[(r+3)*width+c:]))
			}
		}
	}
	// Column tail: cols not covered by a 4-wide block, over the blocked rows.
	if colsBlk < cols {
		for row := 0; row < rowsBlk; row++ {
			line := scratch[row*width : row*width+cols : row*width+cols]
			if rect2 {
				for col := colsBlk; col < cols; col++ {
					line[col] = clipRange(int64(rect2Scale(coeff[col*coeffStride+row])), lo, hi)
				}
			} else {
				for col := colsBlk; col < cols; col++ {
					line[col] = clipRange(int64(coeff[col*coeffStride+row]), lo, hi)
				}
			}
		}
	}
	// Row tail: the remaining rows across every column.
	if rowsBlk < rows {
		stageTransposeClampScalar(scratch[rowsBlk*width:], width, coeff[rowsBlk:], coeffStride, rows-rowsBlk, cols, rect2, lo, hi)
	}
}
