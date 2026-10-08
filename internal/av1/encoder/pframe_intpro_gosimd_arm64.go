//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// pframe_intpro_gosimd_arm64.go hosts the Go-native SIMD integer projection
// kernels (libaom's aom_int_pro_row / aom_int_pro_col shape). The guards mirror
// the preconditions the retired NEON kernels accepted; outside them the scalar
// reference runs, so output bytes are unchanged.
//
// Row: 16 columns per vector. Each column sums projHeight bytes into 16-bit
// lanes; projHeight <= 257 keeps a lane at most 257*255 = 65535. Four row
// accumulators run in parallel so the adds do not form one latency chain.
// Col: four rows at a time, 16 bytes per vector widen-accumulate into 16-bit
// lanes; width <= 2048 keeps each lane at most 128*2*255 = 65280. The lanes are
// reduced with ReduceSum when the row total fits 16 bits (width <= 257),
// otherwise lane by lane in int.

func realtimeIntProRowInBoundsArch(dst []int16, ref []byte, stride, projWidth, projHeight, normFactor int) {
	if projWidth == 0 || projHeight == 0 ||
		projWidth&15 != 0 || projHeight&3 != 0 || normFactor != 5 ||
		projHeight > 257 {
		realtimeIntProRowInBoundsPureGo(dst, ref, stride, projWidth, projHeight, normFactor)
		return
	}
	_ = dst[projWidth-1]
	_ = ref[(projHeight-1)*stride+projWidth-1]
	base := unsafe.Pointer(&ref[0])
	zero := archsimd.BroadcastUint8x16(0)
	shift := uint64(normFactor)
	for x := 0; x < projWidth; x += 16 {
		p := unsafe.Add(base, x)
		loA, hiA := archsimd.BroadcastUint16x8(0), archsimd.BroadcastUint16x8(0)
		loB, hiB := archsimd.BroadcastUint16x8(0), archsimd.BroadcastUint16x8(0)
		loC, hiC := archsimd.BroadcastUint16x8(0), archsimd.BroadcastUint16x8(0)
		loD, hiD := archsimd.BroadcastUint16x8(0), archsimd.BroadcastUint16x8(0)
		for y := 0; y < projHeight; y += 4 {
			a := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, y*stride)))
			b := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, (y+1)*stride)))
			c := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, (y+2)*stride)))
			d := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, (y+3)*stride)))
			loA = loA.Add(a.ExtendLo8ToUint16())
			hiA = hiA.Add(a.InterleaveHi(zero).ReshapeToUint16s())
			loB = loB.Add(b.ExtendLo8ToUint16())
			hiB = hiB.Add(b.InterleaveHi(zero).ReshapeToUint16s())
			loC = loC.Add(c.ExtendLo8ToUint16())
			hiC = hiC.Add(c.InterleaveHi(zero).ReshapeToUint16s())
			loD = loD.Add(d.ExtendLo8ToUint16())
			hiD = hiD.Add(d.InterleaveHi(zero).ReshapeToUint16s())
		}
		loA.Add(loB).Add(loC.Add(loD)).ShiftAllRight(shift).BitsToInt16().Store(dst[x : x+8])
		hiA.Add(hiB).Add(hiC.Add(hiD)).ShiftAllRight(shift).BitsToInt16().Store(dst[x+8 : x+16])
	}
}

func realtimeIntProColInBoundsArch(dst []int16, ref []byte, stride, projWidth, projHeight, normFactor int) {
	if projWidth == 0 || projHeight == 0 ||
		projWidth&15 != 0 || projHeight&3 != 0 || normFactor != 5 ||
		projWidth > 2056 {
		realtimeIntProColInBoundsPureGo(dst, ref, stride, projWidth, projHeight, normFactor)
		return
	}
	_ = dst[projHeight-1]
	_ = ref[(projHeight-1)*stride+projWidth-1]
	base := unsafe.Pointer(&ref[0])
	zero := archsimd.BroadcastUint8x16(0)
	narrow := projWidth <= 257
	for y := 0; y < projHeight; y += 4 {
		pa := unsafe.Add(base, y*stride)
		pb := unsafe.Add(pa, stride)
		pc := unsafe.Add(pb, stride)
		pd := unsafe.Add(pc, stride)
		accA, accB := archsimd.BroadcastUint16x8(0), archsimd.BroadcastUint16x8(0)
		accC, accD := archsimd.BroadcastUint16x8(0), archsimd.BroadcastUint16x8(0)
		for x := 0; x < projWidth; x += 16 {
			a := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(pa, x)))
			b := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(pb, x)))
			c := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(pc, x)))
			d := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(pd, x)))
			accA = accA.Add(a.ExtendLo8ToUint16()).Add(a.InterleaveHi(zero).ReshapeToUint16s())
			accB = accB.Add(b.ExtendLo8ToUint16()).Add(b.InterleaveHi(zero).ReshapeToUint16s())
			accC = accC.Add(c.ExtendLo8ToUint16()).Add(c.InterleaveHi(zero).ReshapeToUint16s())
			accD = accD.Add(d.ExtendLo8ToUint16()).Add(d.InterleaveHi(zero).ReshapeToUint16s())
		}
		dst[y] = int16(colLaneSum(accA, narrow) >> uint(normFactor))
		dst[y+1] = int16(colLaneSum(accB, narrow) >> uint(normFactor))
		dst[y+2] = int16(colLaneSum(accC, narrow) >> uint(normFactor))
		dst[y+3] = int16(colLaneSum(accD, narrow) >> uint(normFactor))
	}
}

// colLaneSum totals the eight 16-bit lanes of a column accumulator. The
// 16-bit ReduceSum is exact only while the whole row total fits 16 bits.
func colLaneSum(acc archsimd.Uint16x8, narrow bool) int {
	if narrow {
		return int(acc.ReduceSum())
	}
	return int(acc.GetElem(0)) + int(acc.GetElem(1)) + int(acc.GetElem(2)) + int(acc.GetElem(3)) +
		int(acc.GetElem(4)) + int(acc.GetElem(5)) + int(acc.GetElem(6)) + int(acc.GetElem(7))
}
