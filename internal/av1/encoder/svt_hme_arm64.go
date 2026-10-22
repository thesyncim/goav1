//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// svtDownsample2DArch is svtDownsample2D on NEON: the 2x2 sums of 16 output
// samples are two row-widened adds folded by pairwise adds, then
// (sum + 2) >> 2 exactly as svt_aom_downsample_2d_c.
func svtDownsample2DArch(dst []byte, dstStride int, src []byte, srcStride, w, h int) {
	two := archsimd.BroadcastUint16x8(2)
	for y := 0; y < h; y++ {
		r0 := src[(2*y)*srcStride:]
		r1 := src[(2*y+1)*srcStride:]
		d := dst[y*dstStride : y*dstStride+w]
		x := 0
		if len(r0) >= 2*w && len(r1) >= 2*w {
			p0 := unsafe.Pointer(&r0[0])
			p1 := unsafe.Pointer(&r1[0])
			for ; x+16 <= w; x += 16 {
				a := load16(unsafe.Add(p0, 2*x))
				b := load16(unsafe.Add(p1, 2*x))
				c := load16(unsafe.Add(p0, 2*x+16))
				e := load16(unsafe.Add(p1, 2*x+16))
				lo := a.ExtendLo8ToUint16().Add(b.ExtendLo8ToUint16()).ConcatAddPairs(widenHi8(a).Add(widenHi8(b)))
				hi := c.ExtendLo8ToUint16().Add(e.ExtendLo8ToUint16()).ConcatAddPairs(widenHi8(c).Add(widenHi8(e)))
				lo = lo.Add(two).ShiftAllRight(2)
				hi = hi.Add(two).ShiftAllRight(2)
				out := lo.SaturateToUint8().ReshapeToUint64s().InterleaveLo(hi.SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s()
				out.StoreArray((*[16]uint8)(unsafe.Pointer(&d[x])))
			}
		}
		for ; x < w; x++ {
			s := uint32(r0[2*x]) + uint32(r0[2*x+1]) + uint32(r1[2*x]) + uint32(r1[2*x+1])
			d[x] = uint8((s + 2) >> 2)
		}
	}
}

// sad16xNx4 returns the SADs of the 16-wide, rows-tall source block against
// reference origins ref+0..ref+3 (svt_sad_loop_kernel's per-row position
// batch).
func sad16xNx4(src []byte, srcStride int, ref []byte, refStride, rows int) (int, int, int, int) {
	_ = src[(rows-1)*srcStride+15]
	_ = ref[(rows-1)*refStride+18]
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	var c0, c1, c2, c3 archsimd.Uint16x8
	for r := 0; r < rows; r++ {
		s := load16(unsafe.Add(sp, r*srcStride))
		p := unsafe.Add(rp, r*refStride)
		c0 = c0.Add(widen16(absDiffU8x16(s, load16(p))))
		c1 = c1.Add(widen16(absDiffU8x16(s, load16(unsafe.Add(p, 1)))))
		c2 = c2.Add(widen16(absDiffU8x16(s, load16(unsafe.Add(p, 2)))))
		c3 = c3.Add(widen16(absDiffU8x16(s, load16(unsafe.Add(p, 3)))))
	}
	return sumU16(c0), sumU16(c1), sumU16(c2), sumU16(c3)
}
