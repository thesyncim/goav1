//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// lpd1SampledSAD is the row-subsampled SAD the chroma detector uses
// (svt_nxm_sad_kernel called with stride << shift and height >> shift), as
// svt_nxm_sad_kernel_helper_neon: 16-byte rows directly, 8-byte rows two to
// a vector. The detector samples at most 8 rows of at most 32 samples, so a
// uint16 lane gains at most 16*510.
func lpd1SampledSAD(src, ref []byte, srcOff, refOff, srcStride, refStride, w, rows, step int) int {
	if rows <= 0 || w <= 0 {
		return 0
	}
	ss, rs := step*srcStride, step*refStride
	if (w != 8 && w&15 != 0) || rows > 8 || w > 32 {
		return lpd1SampledSADPureGo(src, ref, srcOff, refOff, srcStride, refStride, w, rows, step)
	}
	_ = src[srcOff+(rows-1)*ss+w-1]
	_ = ref[refOff+(rows-1)*rs+w-1]
	sp := unsafe.Pointer(&src[srcOff])
	rp := unsafe.Pointer(&ref[refOff])
	var acc archsimd.Uint16x8
	if w == 8 {
		r := 0
		for ; r+2 <= rows; r += 2 {
			acc = absAcc16(acc, pack2Rows(unsafe.Add(sp, r*ss), ss), pack2Rows(unsafe.Add(rp, r*rs), rs))
		}
		total := sumU16(acc)
		if r < rows {
			s := src[srcOff+r*ss : srcOff+r*ss+8]
			d := ref[refOff+r*rs : refOff+r*rs+8]
			for c := range s {
				diff := int(s[c]) - int(d[c])
				total += max(diff, -diff)
			}
		}
		return total
	}
	for r := range rows {
		s, d := unsafe.Add(sp, r*ss), unsafe.Add(rp, r*rs)
		for c := 0; c < w; c += 16 {
			acc = absAcc16(acc, load16(unsafe.Add(s, c)), load16(unsafe.Add(d, c)))
		}
	}
	return sumU16(acc)
}
