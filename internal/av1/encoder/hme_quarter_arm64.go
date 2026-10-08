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
		for qy := 0; qy < qh; qy++ {
			srow := step(sp, qy*4*stride)
			drow := step(dp, qy*qw)
			for x := 0; x < neonCols; x += 16 {
				quarterBlock16(step(srow, x*4), stride, step(drow, x))
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

// quarterBlock16 writes 16 box averages from the 4-row, 64-byte source block at
// sp. The four source rows are summed per byte column first, so the widened
// sums need only 16 bits (at most 4*255 per column). Adjacent-lane sums then
// fold column pairs and quads with ConcatEven/ConcatOdd, and the result is
// rounded with (sum+8)>>4 and narrowed to bytes.
func quarterBlock16(sp unsafe.Pointer, stride int, dp unsafe.Pointer) {
	// c[m] holds byte columns 8m..8m+7 summed over the four rows.
	c0 := archsimd.BroadcastUint16x8(0)
	c1 := archsimd.BroadcastUint16x8(0)
	c2 := archsimd.BroadcastUint16x8(0)
	c3 := archsimd.BroadcastUint16x8(0)
	c4 := archsimd.BroadcastUint16x8(0)
	c5 := archsimd.BroadcastUint16x8(0)
	c6 := archsimd.BroadcastUint16x8(0)
	c7 := archsimd.BroadcastUint16x8(0)
	for r := 0; r < 4; r++ {
		row := step(sp, r*stride)
		v0 := load16(row)
		v1 := load16(step(row, 16))
		v2 := load16(step(row, 32))
		v3 := load16(step(row, 48))
		c0 = c0.Add(v0.ExtendLo8ToUint16())
		c1 = c1.Add(v0.HiToLo().ExtendLo8ToUint16())
		c2 = c2.Add(v1.ExtendLo8ToUint16())
		c3 = c3.Add(v1.HiToLo().ExtendLo8ToUint16())
		c4 = c4.Add(v2.ExtendLo8ToUint16())
		c5 = c5.Add(v2.HiToLo().ExtendLo8ToUint16())
		c6 = c6.Add(v3.ExtendLo8ToUint16())
		c7 = c7.Add(v3.HiToLo().ExtendLo8ToUint16())
	}
	// Pair sums: lane j of p is column 2j plus column 2j+1 of the 16-column run.
	p0 := c0.ConcatEven(c1).Add(c0.ConcatOdd(c1))
	p1 := c2.ConcatEven(c3).Add(c2.ConcatOdd(c3))
	p2 := c4.ConcatEven(c5).Add(c4.ConcatOdd(c5))
	p3 := c6.ConcatEven(c7).Add(c6.ConcatOdd(c7))
	// Quad sums: lane x of q is the 4-column sum for output column x.
	q0 := p0.ConcatEven(p1).Add(p0.ConcatOdd(p1))
	q1 := p2.ConcatEven(p3).Add(p2.ConcatOdd(p3))
	eight := archsimd.BroadcastUint16x8(8)
	ra := q0.Add(eight).ShiftAllRight(4).ReshapeToUint8s()
	rb := q1.Add(eight).ShiftAllRight(4).ReshapeToUint8s()
	// Each result lane is at most 255, so its low byte is the value and the
	// even bytes of ra then rb are the 16 outputs in order.
	ra.ConcatEven(rb).StoreArray((*[16]uint8)(dp))
}
