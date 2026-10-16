//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

func init() {
	realtimeAvg8x8QuadImpl = realtimeAvg8x8QuadSIMD
}

// realtimeAvg8x8QuadSIMD is realtimeAvg8x8QuadWide on 16-byte rows: the left
// and right 8x8 halves of each eight-row band accumulate their bytes in
// 16-bit lanes (at most 8*255 per lane), and a lane reduction gives each
// block's sum (at most 64*255, still within 16 bits).
func realtimeAvg8x8QuadSIMD(src []byte, stride int) (int, int, int, int) {
	_ = src[15*stride+15]
	p := unsafe.Pointer(&src[0])
	tl, tr := avg8x8BandSums(p, stride)
	bl, br := avg8x8BandSums(unsafe.Add(p, 8*stride), stride)
	return (tl + 32) >> 6, (tr + 32) >> 6, (bl + 32) >> 6, (br + 32) >> 6
}

func avg8x8BandSums(p unsafe.Pointer, stride int) (int, int) {
	var l0, r0, l1, r1 archsimd.Uint16x8
	for r := 0; r < 8; r += 2 {
		a := load16(unsafe.Add(p, r*stride))
		b := load16(unsafe.Add(p, (r+1)*stride))
		l0 = l0.Add(a.ExtendLo8ToUint16())
		r0 = r0.Add(widenHi8(a))
		l1 = l1.Add(b.ExtendLo8ToUint16())
		r1 = r1.Add(widenHi8(b))
	}
	return int(l0.Add(l1).ReduceSum()), int(r0.Add(r1).ReduceSum())
}
