//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// buildQuarterPlaneArch box-averages 4x4 luma blocks into dst. The Go-native
// NEON body writes 16 quarter-plane samples (64 source columns) per step; the
// remaining columns use the scalar reference, matching buildQuarterPlanePureGo.
func buildQuarterPlaneArch(dst []byte, src []byte, stride, qw, qh int) {
	neonCols := qw &^ 15
	if neonCols > 0 && qh > 0 {
		_ = dst[(qh-1)*qw+neonCols-1]
		_ = src[(qh*4-1)*stride+neonCols*4-1]
		dp := unsafe.Pointer(&dst[0])
		sp := unsafe.Pointer(&src[0])
		zero := archsimd.BroadcastUint8x16(0).ReshapeToUint16s()
		eight := archsimd.BroadcastUint8x16(8).ExtendLo8ToUint16()
		scale := archsimd.BroadcastUint8x16(255).ExtendLo8ToUint16()
		roundShift := archsimd.BroadcastInt16x8(-4)
		for qy := 0; qy < qh; qy++ {
			srow := step(sp, qy*4*stride)
			drow := step(dp, qy*qw)
			for x := 0; x < neonCols; x += 16 {
				blockSrc := step(srow, x*4)
				blockDst := step(drow, x)
				s0, h0 := zero, zero
				s1, h1 := zero, zero
				s2, h2 := zero, zero
				s3, h3 := zero, zero
				for r := 0; r < 4; r++ {
					row := step(blockSrc, r*stride)
					v0 := load16(row)
					v1 := load16(step(row, 16))
					v2 := load16(step(row, 32))
					v3 := load16(step(row, 48))
					s0, h0 = quarterAcc(s0, h0, v0)
					s1, h1 = quarterAcc(s1, h1, v1)
					s2, h2 = quarterAcc(s2, h2, v2)
					s3, h3 = quarterAcc(s3, h3, v3)
				}
				q0 := s0.ConcatAddPairs(s1).Sub(h0.ConcatAddPairs(h1).Mul(scale))
				q1 := s2.ConcatAddPairs(s3).Sub(h2.ConcatAddPairs(h3).Mul(scale))
				ra := q0.Add(eight).Shift(roundShift).ReshapeToUint8s()
				rb := q1.Add(eight).Shift(roundShift).ReshapeToUint8s()
				// Each result lane is at most 255, so its low byte is the value and the
				// even bytes of ra then rb are the 16 outputs in order.
				ra.ConcatEven(rb).StoreArray((*[16]uint8)(blockDst))
			}
		}
	}
	if neonCols != qw {
		for qy := 0; qy < qh; qy++ {
			buildQuarterPlanePureGo(
				dst[qy*qw+neonCols:],
				src[qy*4*stride+neonCols*4:],
				stride,
				qw-neonCols,
				1,
			)
		}
	}
}

func quarterAcc(sum, high archsimd.Uint16x8, v archsimd.Uint8x16) (archsimd.Uint16x8, archsimd.Uint16x8) {
	return sum.Add(v.ReshapeToUint16s()), high.Add(v.ConcatOdd(v).ExtendLo8ToUint16())
}
