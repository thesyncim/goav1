//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// pixelStatsRows16Arch is pixelStatsRows16 with NEON's widening multiply: the
// int16 differences square straight into int32 lanes (SMULL on the low and
// high halves) instead of being sign-extended first, and the signed sum
// accumulates in int16 lanes (at most 510 per 16-sample step, flushed to
// int32 every 32 steps).
func pixelStatsRows16Arch(src []byte, srcStride int, ref []byte, refStride int, w, h int) (uint32, int32) {
	if h <= 0 {
		return 0, 0
	}
	_ = src[(h-1)*srcStride+w-1]
	_ = ref[(h-1)*refStride+w-1]
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	var sse0, sse1, sum32 archsimd.Int32x4
	var sum16 archsimd.Int16x8
	steps := 0
	for r := 0; r < h; r++ {
		s := unsafe.Add(sp, r*srcStride)
		p := unsafe.Add(rp, r*refStride)
		for c := 0; c < w; c += 16 {
			a := load16(unsafe.Add(s, c))
			b := load16(unsafe.Add(p, c))
			dl := a.ExtendLo8ToUint16().BitsToInt16().Sub(b.ExtendLo8ToUint16().BitsToInt16())
			dh := widenHi8(a).BitsToInt16().Sub(widenHi8(b).BitsToInt16())
			sum16 = sum16.Add(dl).Add(dh)
			sse0 = sse0.Add(dl.MulWidenLo(dl)).Add(dh.MulWidenLo(dh))
			dl, dh = dl.HiToLo(), dh.HiToLo()
			sse1 = sse1.Add(dl.MulWidenLo(dl)).Add(dh.MulWidenLo(dh))
			if steps++; steps == 32 {
				sum32 = sum32.Add(sum16.ExtendLo4ToInt32()).Add(sum16.HiToLo().ExtendLo4ToInt32())
				sum16 = archsimd.Int16x8{}
				steps = 0
			}
		}
	}
	sum32 = sum32.Add(sum16.ExtendLo4ToInt32()).Add(sum16.HiToLo().ExtendLo4ToInt32())
	total, signed := reducePixelAcc(sse0.Add(sse1), sum32)
	return uint32(total), int32(signed)
}
